from torch import nn
import torch
import torch.utils.checkpoint
from .base_models import Encoder, Decoder, Decoder_AR
from .model_utils import *
import torch.utils.checkpoint
from copy import deepcopy
from .model_utils import pna_aggregate


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff, num_out=None):
        if num_out is None:
            num_out = num_hidden
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_out, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nn.Module):
    def __init__(
        self,
        num_E_dim,
        num_V_dim,
        num_hidden,
        dropout=0.1,
        scale=30,
        no_edge_update=False,
        pna=False,
    ):
        super(EncLayer, self).__init__()
        self.scale = scale
        self.pna = pna
        self.no_edge_update = no_edge_update
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_E_dim)

        self.W1 = nn.Linear(num_E_dim + num_V_dim * 2, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_V_dim, bias=True)
        self.W11 = nn.Linear(num_E_dim + num_V_dim * 2, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_E_dim, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden * 4 if self.pna else num_hidden, num_hidden * 4, num_hidden)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # edge + gathered neighbor features
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # + original vertex
        h_message = self.W3(
            self.act(self.W2(self.act(self.W1(h_EV))))
        )  # 3 linear layers
        if mask_attend is not None:
            h_message = (
                mask_attend.unsqueeze(-1) * h_message
            )  # 0 if at least one vertex is unknown

        if self.pna:
            dh = pna_aggregate(h_message, mask_attend)
            dh = self.dense(torch.cat([h_V, dh], -1))  # 3 linear layers
        else:
            dh = torch.sum(h_message, -2) / self.scale  # sum over neighbors
            h_V = self.norm1(
                h_V + self.dropout1(dh)
            )  # update known vertices with sum of output over neighbors
            dh = self.dense(h_V)

        h_V = self.norm2(
            h_V + self.dropout2(dh)
        )  # update vertices with output of a dense network
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V  # set unknown to 0

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # edge + gathered neighbor features
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # + original vertex
        h_message = self.W13(
            self.act(self.W12(self.act(self.W11(h_EV))))
        )  # 3 linear layers
        if not self.no_edge_update:
            h_E = self.norm3(h_E + self.dropout3(h_message))  # update edges with output
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(
        self, num_hidden, num_in, dropout=0.1, scale=30, pna=False,
    ):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.pna = pna
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden * 4 if self.pna else num_hidden, num_hidden * 4, num_hidden)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)  # edge and vertex features

        h_message = self.W3(
            self.act(self.W2(self.act(self.W1(h_EV))))
        )  # 3 linear layers
        if mask_attend is not None:
            h_message = (
                mask_attend.unsqueeze(-1) * h_message
            )  # set to 0 if model doesn't see the vertex
        if self.pna:
            dh = self.dense(torch.cat([h_V, pna_aggregate(h_message, mask_attend)], -1))
        else:
            dh = torch.sum(h_message, -2) / self.scale
            h_V = self.norm1(h_V + self.dropout1(dh))
            dh = self.dense(h_V)

        h_V = self.norm2(h_V + self.dropout2(dh))  # update vertex features again

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V  # set non-existing to 0
        return h_V


class MPNN_Decoder_EU(Encoder):
    def __init__(self, args) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncLayer(
                    num_V_dim=args.hidden_dim,
                    num_E_dim=args.in_dim,
                    num_hidden=args.hidden_dim,
                    dropout=args.dropout,
                    no_edge_update=args.no_edge_update,
                    pna=args.use_pna_in_decoder,
                )
                for _ in range(args.num_decoder_layers)
            ]
        )
        self.use_attn = args.use_attention_in_decoder
        if self.use_attn:
            self.self_attn = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(args.hidden_dim, 4, dropout=0.01, batch_first=True) for _ in range(args.num_decoder_layers)
                ]
            )
        else:
            self.self_attn = [None for _ in range(args.num_decoder_layers)]

        
    def forward(self, h_V, h_E, E_idx, mask, X, residue_idx, chain_encoding_all, global_context):
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer, attn in zip(self.layers, self.self_attn):
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend
            )
            if self.use_attn:
                h_V = attn(
                    query=h_V, 
                    key=h_V, 
                    value=h_V,
                    key_padding_mask=~mask.bool(),
                    need_weights=False
                )[0]
                h_V[mask.bool()] = 0
        return h_V, h_E, X, E_idx


class MPNN_Encoder(Encoder):
    def __init__(self, args) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncLayer(
                    num_V_dim=args.hidden_dim,
                    num_E_dim=args.hidden_dim,
                    num_hidden=args.hidden_dim,
                    dropout=args.dropout,
                    no_edge_update=args.no_edge_update,
                    pna=args.use_pna_in_encoder,
                )
                for _ in range(args.num_encoder_layers)
            ]
        )
        self.use_attn = args.use_attention_in_encoder
        if self.use_attn:
            self.self_attn = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(args.hidden_dim, 4, dropout=0.01, batch_first=True) for _ in range(args.num_encoder_layers)
                ]
            )
        else:
            self.self_attn = [None for _ in range(args.num_encoder_layers)]

        
    def forward(self, h_V, h_E, E_idx, mask, X, residue_idx, chain_encoding_all, global_context):
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer, attn in zip(self.layers, self.self_attn):
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend
            )
            if self.use_attn:
                h_V = attn(
                    query=h_V, 
                    key=h_V, 
                    value=h_V,
                    key_padding_mask=~mask.bool(),
                    need_weights=False
                )[0]
                h_V[mask.bool()] = 0
        return h_V, h_E, X, E_idx


class MPNN_Decoder_AR(Decoder_AR):
    def __init__(self, args) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecLayer(
                    args.hidden_dim, args.in_dim + args.hidden_dim, dropout=args.dropout, pna=args.use_pna_in_decoder,
                )
                for _ in range(args.num_decoder_layers)
            ]
        )

    def get_autoregression_masks(self, chain_M, mask, E_idx):
        device = chain_M.device
        chain_M = chain_M * mask  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device)))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )  # in each line, a_{ij} = 1 if i is earlier in the list than j, 0 otherwise
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(
            -1
        )  # do the neighbors come earlier than the vertex?
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend  # do the neighbors come earlier and also exist?
        mask_fw = mask_1D * (
            1.0 - mask_attend
        )  # do the neighbors come later and also exist?
        return mask_bw, mask_fw

    def forward(self, h_V, h_E, h_S, E_idx, mask, X, chain_M, test=False, out=None, seq=None, out_seq=None, num_letters=21):
        # Get autoregression masks
        mask_bw, mask_fw = self.get_autoregression_masks(chain_M, mask, E_idx)

        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = cat_neighbors_nodes(
            h_S, h_E, E_idx
        )  # edge + neigbour residue features

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(h_S), h_E, E_idx
        )  # edge features + zeros
        h_EXV_encoder = cat_neighbors_nodes(
            h_V, h_EX_encoder, E_idx
        )  # + vertex features

        if not test:
            h_EXV_encoder_fw = (
                mask_fw * h_EXV_encoder
            )  # edge features + zeros + vertex features * exist & come later
            for layer in self.layers:
                h_ESV = cat_neighbors_nodes(
                    h_V, h_ES, E_idx
                )  # vertex (updated) + edge + neighbor residue features
                h_ESV = (
                    mask_bw * h_ESV + h_EXV_encoder_fw
                )  # for neighbors that come earlier, keep (true) residues, for neighbors that come later, set zeros
                h_V = torch.utils.checkpoint.checkpoint(
                    layer, h_V, h_ESV, mask
                )  # update vertex features with h_ESV
        else:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=chain_M.device)))
            )
            max_len = int(chain_M.sum(1).max())
            decoding_order = decoding_order[:,  -max_len:]
            masked_res = deepcopy(chain_M * mask) # 1 if masked and exists
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            output = h_V.clone()
            for i in range(decoding_order.shape[1]):
                chain_M_gathered = gather_nodes((1 - masked_res).unsqueeze(-1), E_idx) # 0 if masked, 1 if not masked
                mask_bw = mask_1D * chain_M_gathered # 1 if exists and not masked
                mask_fw = mask_1D * (1 - chain_M_gathered) # 1 if exists and masked
                idx = [(k, j) for k, j in enumerate(decoding_order[:, i]) if masked_res[k, j] == 1] # only if exists and masked
                idx = list(zip(*idx))
                h_V_start = h_V.clone()
                h_EXV_encoder_fw = (
                    mask_fw * h_EXV_encoder
                )
                for layer in self.layers:
                    h_ESV = cat_neighbors_nodes(
                        h_V_start, h_ES, E_idx
                    )
                    h_ESV = (
                        mask_bw * h_ESV + h_EXV_encoder_fw
                    )
                    h_V_start = torch.utils.checkpoint.checkpoint(
                        layer, h_V_start, h_ESV, mask
                    )
                output[idx] = h_V_start[idx]
                masked_res[idx] = 0
                logits = out(h_V_start)
                seq[idx] = torch.max(logits.detach(), -1)[1][idx]
                if num_letters == 20:
                    seq[idx] += 1
                h_S = out_seq(seq)
                h_ES = cat_neighbors_nodes(
                    h_S, h_E, E_idx
                )
            h_V = output
        return h_V, h_E, X, E_idx


class MPNN_Decoder_OS(Decoder):
    def __init__(self, args) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecLayer(
                    args.hidden_dim, args.in_dim + args.hidden_dim, dropout=args.dropout, pna=args.use_pna_in_decoder,
                )
                for _ in range(args.num_decoder_layers)
            ]
        )
        self.use_attn = args.use_attention_in_decoder
        if self.use_attn:
            self.self_attn = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(args.hidden_dim, 4, dropout=0.01, batch_first=True) for _ in range(args.num_encoder_layers)
                ]
            )
        else:
            self.self_attn = [None for _ in range(args.num_encoder_layers)]
    
    def forward(self, h_V, h_E, E_idx, mask, X, residue_idx, chain_encoding_all, global_context):
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer, attn in zip(self.layers, self.self_attn):
            h_EV = cat_neighbors_nodes(
                h_V, h_E, E_idx
            )  # vertex (updated) + edge + neighbor residue features
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_EV, mask, mask_attend)
            if attn is not None:
                h_V = attn(query=h_V, key=h_V, value=h_V,
                                key_padding_mask=~(mask.bool()),
                                need_weights=False)[0]
                h_V[mask.bool()] = 0
        return h_V, h_E, X, E_idx
