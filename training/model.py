from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from copy import deepcopy

from .mpnn import MPNN_Encoder, MPNN_Decoder_OS, MPNN_Decoder_AR, MPNN_Decoder_EU
from einops import repeat, rearrange
from copy import copy, deepcopy
from training.model_utils import *
import pandas as pd


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        only_c_alpha=False,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.only_c_alpha = only_c_alpha

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        num_dist = 25 if not only_c_alpha else 1
        _, edge_in = 6, num_positional_embeddings + num_rbf * num_dist
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        if not self.only_c_alpha:
            RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
            RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
            RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
            RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
            RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)  # + 24

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        args,
        encoder_type,
        decoder_type,
        hidden_dim=128,
        vocab=21,
        k_neighbors=32,
        augment_eps=0.2,
        noise_unknown=None,
        noise_unknown_internal=None,
        embedding_dim=128,
        mask_attention="none",
        ignore_unknown=False,
        node_features_type="zeros",
        only_c_alpha: bool = False,
        n_cycles: int = 1,
        no_sequence_in_encoder: bool = False,
        double_sequence_features: bool = False,
        separate_modules_num: int = 1,
    ):
        super(ProteinMPNN, self).__init__()
        encoders = {
            "mpnn": MPNN_Encoder,
        }
        decoders = {
            "mpnn": MPNN_Decoder_OS,
            "mpnn_auto": MPNN_Decoder_AR,
        }
        auto_decoders = ["mpnn_auto"]
        num_letters = 20 if ignore_unknown else 21

        if noise_unknown is None:
            noise_unknown = augment_eps
        if noise_unknown_internal is None:
            noise_unknown_internal = 0
        self.num_letters = num_letters
        self.attention = mask_attention
        self.node_features_type = node_features_type
        self.only_c_alpha = only_c_alpha
        self.augment_eps = augment_eps
        self.noise_unknown = noise_unknown
        self.noise_unknown_internal = noise_unknown_internal
        self.n_cycles = n_cycles
        self.no_sequence_in_encoder = no_sequence_in_encoder
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.one_shot_decoder = (decoder_type not in auto_decoders)
        self.use_sequence_in_encoder = self.one_shot_decoder if not no_sequence_in_encoder else False
        self.add_sequence_in_decoder = not self.use_sequence_in_encoder or double_sequence_features
        
        if separate_modules_num > n_cycles:
            separate_modules_num = n_cycles

        self.str_features = False
        self.seq_features = False

        self.features = ProteinFeatures(
            hidden_dim,
            top_k=k_neighbors,
            only_c_alpha=only_c_alpha,
        )
        args.edge_compute_func = self.features

        n_vectors = {"sidechain_orientation": 1}
        args.vector_dim = 4 + sum([n_vectors[x] for x in node_features_type.split("+") if x in n_vectors])

        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, embedding_dim)
        if node_features_type is not None:
            d_structure = {"dihedral": 2, "secondary_structure": 3}
            d_sequence = {"chemical": 6}
            input_f_structure = sum(
                [
                    d_structure[x]
                    for x in node_features_type.split("+")
                    if x in d_structure
                ]
            )
            input_f_seq = sum(
                [
                    d_sequence[x]
                    for x in node_features_type.split("+")
                    if x in d_sequence
                ]
            )
            if input_f_structure > 0:
                self.str_features = True
            if self.use_sequence_in_encoder:

                if input_f_seq > 0:
                    input_f_structure += hidden_dim
                else:
                    input_f_structure += embedding_dim
            if input_f_structure > 0:
                self.W_v_str = nn.Linear(input_f_structure, hidden_dim, bias=True)
            if input_f_seq > 0:
                self.seq_features = True
                self.W_v_seq = nn.Linear(input_f_seq + embedding_dim, hidden_dim, bias=True)

        self.separate_modules_num = separate_modules_num
        self.encoders = nn.ModuleList([encoders[encoder_type](args)])
        if separate_modules_num > 1:
            self.encoders += nn.ModuleList([encoders[encoder_type](args) for i in range(separate_modules_num - 1)])

        # Decoder layers
        in_dim = hidden_dim # edge features
        if self.add_sequence_in_decoder or not self.one_shot_decoder:
            if not self.seq_features:
                in_dim += embedding_dim
            else:
                in_dim += hidden_dim
        args.in_dim = in_dim

        self.decoders = nn.ModuleList([decoders[decoder_type](args) for i in range(n_cycles)])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_e_cycle = None

        if self.attention != "none":
            self.att_layer = MaskAttLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                include_nodes=(node_features_type != "zeros"),
                num_heads=3,
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def noise_coords(self, X, V_structure, vector_node_struct, chain_labels):

        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        return X, V_structure, vector_node_struct

    def generate_embeddings(self, X, S, mask, residue_idx, chain_encoding_all, V_struct):

        E, E_idx = self.features(X[:, :, : 4], mask, residue_idx, chain_encoding_all)

        if not self.str_features:
            h_V = torch.zeros((E.shape[0], E.shape[1], self.hidden_dim), device=E.device) # node embeddings = zeros
        else:
            h_V = self.W_v_str(V_struct)
        
        h_E = self.W_e(E)
        return h_V, h_E, E_idx
    
    def find_chains_idx(self, residue_idx):

        diffs = residue_idx[:, 1 : ] - residue_idx[:,  : -1]
        idxs = torch.nonzero(diffs > 1)
        idxs = [np.array([w.item() for w in idxs[idxs[:, 0] == k][:, 1]] + [len(diffs[k][diffs[k] > 0])]) for k in range(residue_idx.shape[0])]
        return idxs
    
    def random_unit_vectors_like(self, tensor):
        rand_vecs = torch.randn_like(tensor)
        norms = torch.norm(rand_vecs, dim=-1, keepdim=True)
        return rand_vecs / norms

    def initialize_sequence(self, seq, V_sequence, chain_M, mask, residue_idx, vector_node_seq):
        if self.one_shot_decoder:
            if V_sequence is not None:
                V_sequence[chain_M.bool()] = 0
        seq[chain_M.bool()] = 0
        if self.one_shot_decoder and vector_node_seq is not None:
            vector_node_seq[chain_M.bool()] = self.random_unit_vectors_like(vector_node_seq[chain_M.bool()])
        
        return seq, V_sequence, vector_node_seq
        
    def extract_features(self, seq, chain_M, optional_features, mask, residue_idx, chain_encoding_all, X, cycle):
        vector_node_seq = optional_features.get("vector_node_seq")
        vector_node_struct = optional_features.get("vector_node_struct")
        V_sequence = optional_features["scalar_seq"]
        V_structure = optional_features["scalar_struct"]
        if cycle == 0:
            seq, V_sequence, vector_node_seq = self.initialize_sequence(seq, V_sequence, chain_M, mask, residue_idx, vector_node_seq)
            X, V_structure, vector_node_struct = self.noise_coords(X, V_structure, vector_node_struct, chain_M)
        h_S = self.W_s(seq)
        if self.seq_features:
            h_S = torch.cat([V_sequence, h_S], -1)
            h_S = self.W_v_seq(h_S)

        if self.use_sequence_in_encoder:
            if self.str_features:
                V_structure = torch.cat([V_structure, h_S], -1)
            else:
                V_structure = h_S
        
        if vector_node_seq is not None:
            X = torch.cat([X, vector_node_seq], 2)
        if vector_node_struct is not None:
            X = torch.cat([X, vector_node_struct], 2)

        # Prepare node and edge embeddings
        h_V, h_E, E_idx = self.generate_embeddings(X, seq, mask, residue_idx, chain_encoding_all, V_structure)
        return h_V, h_E, E_idx, X, h_S

    def forward(
        self,
        X,
        S,
        mask,
        mask_original,
        chain_M,
        residue_idx,
        chain_encoding_all,
        optional_features,
        test=False,
    ):
        """Graph-conditioned sequence model"""

        output = []
        seq = deepcopy(S)
        coords = deepcopy(X)
        global_context = None
        for cycle in range(self.n_cycles):
            h_V, h_E, E_idx, coords, h_S = self.extract_features(seq, chain_M, optional_features, mask, residue_idx, chain_encoding_all, coords, cycle)

            h_V, h_E, coords, E_idx = self.encoders[min(cycle, len(self.encoders) - 1)](
                h_V, h_E, E_idx, mask, coords, residue_idx, chain_encoding_all, global_context
            )

            decoder_module = self.decoders[min(cycle, len(self.decoders) - 1)]
            if self.one_shot_decoder:
                if self.add_sequence_in_decoder:
                    h_E = cat_neighbors_nodes(
                        h_S, h_E, E_idx
                    )
                h_V, h_E, coords, E_idx = decoder_module(
                    h_V, h_E, E_idx, mask, coords, residue_idx, chain_encoding_all, global_context
                )
            else:
                h_V, h_E, coords, E_idx = decoder_module(h_V, h_E, h_S, E_idx, mask, coords, chain_M, test=test, out=self.W_out, out_seq=self.W_s, num_letters=self.num_letters, seq=seq.clone())

            out = {}
            logits = self.W_out(h_V)
            if self.W_e_cycle is not None:
                h_E = self.W_e_cycle(h_E)
            else:
                seq = seq.clone()
                seq[chain_M.bool()] = torch.max(logits.detach(), -1)[1][chain_M.bool()]
                if self.num_letters == 20:
                    seq[chain_M.bool()] += 1
                out["seq"] = logits.clone()
            output.append(out)

        return output