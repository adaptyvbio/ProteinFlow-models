import argparse
import os.path
import time, os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os.path
from training.model import (
    ProteinMPNN,
)
import random
from tqdm import tqdm
from training.aws_utils import get_trained_model_path
import optuna
from training.model_utils import *
import sys
from copy import deepcopy
from scipy.stats import norm
from math import sqrt
from proteinflow import ProteinLoader


def get_mse_loss(att_mse, mask):
    if att_mse is not None:
        new, old, E_idx = att_mse
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        loss_att_mse = (new - old) ** 2
        return loss_att_mse[mask_attend.bool()].mean()
    return torch.zeros(1).to(mask.device)

def compute_vec_angles(vecs1, vecs2):

    inner_product = (vecs1 * vecs2).sum(dim=-1)
    cos = inner_product / (torch.norm(vecs1, dim=-1) * torch.norm(vecs2, dim=-1) + 1e-6)
    cos -= 1e-6 * torch.sign(cos)
    return torch.acos(cos)

def get_orientation_loss(vecs1, vecs2, mask):
    
    if vecs1 is None or vecs2 is None:
        return 0
    angles = compute_vec_angles(vecs1[mask.bool()], vecs2[mask.bool()])
    return torch.mean(torch.abs((angles - np.pi) / np.pi)).to(mask.device)

def get_orientation_accuracy(vecs1, vecs2, mask):

    if vecs1 is None or vecs2 is None:
        return torch.tensor(0.)
    angles = compute_vec_angles(vecs1[:, :, 0], vecs2[:, :, 0])
    return torch.sum((torch.abs(angles - np.pi) <= (np.pi / 6)) * mask)

def norm_pdf(sigma, x):
    return (1 / (sigma * sqrt(2 * torch.pi))) * torch.exp(-(x ** 2) / (2 * sigma ** 2))

def get_coors_loss(X_pred, X_true, mask, loss_type):
    if loss_type == "mse":
        diff = torch.sum((X_pred[:, :, 2, :][mask.bool()] - X_true[:, :, 2, :][mask.bool()]) ** 2, dim=-1)
        return diff.mean()
    elif loss_type == "huber":
        delta = 1
        diff = torch.sum((X_pred[:, :, 2, :][mask.bool()] - X_true[:, :, 2, :][mask.bool()]) ** 2, dim=-1)
        root = torch.sqrt(diff)
        loss = torch.zeros_like(diff)
        greater_mask = (root > delta)
        loss[~greater_mask] = (1 / 2) * diff[~greater_mask]
        loss[greater_mask] = delta * (root[greater_mask] - delta / 2)
        return loss.mean()
    elif loss_type == "relaxed_mse":
        scale = 0.3
        diff = torch.sum((X_pred[:, :, 2, :] - X_true[:, :, 2, :]) ** 2, dim=-1)
        coef = (norm_pdf(scale, torch.tensor(0)) - norm_pdf(scale, torch.sqrt(diff))) ** 2
        return (coef * diff)[mask.bool()].mean()


def initialize_sequence(seq, chain_M, seq_init_mode):
    if seq_init_mode == "zeros":
        seq[chain_M.bool()] = 0
    elif seq_init_mode == "random":
        seq[chain_M.bool()] = torch.randint(size=seq[chain_M.bool()].shape, low=1, high=22)
    return seq

def compute_loss(model_args, args, model, sidechain_net=None):
    S = model_args["S"]
    X = deepcopy(model_args["X"])
    vecs_gt = deepcopy(model_args["optional_features"]["vector_node_seq"])
    mask = model_args["mask"]
    chain_M = model_args["chain_M"]
    mask_for_loss = mask * chain_M
    if args.ignore_unknown_residues:
        mask_for_loss *= (S != 0)
    
    if args.use_sidechain_pretrained:
        with torch.no_grad():
            orientations = sidechain_net(**model_args)[0]
            model_args["optional_features"]["vector_node_seq"][chain_M.bool()] = orientations[chain_M.bool()]
    output = model(**model_args, test=args.test)
    seq_loss = torch.tensor(0.).to(args.device)
    mse_loss = torch.tensor(0.).to(args.device)
    struct_loss = torch.tensor(0.).to(args.device)
    orientation_loss = torch.tensor(0.).to(args.device)
    total_loss = torch.tensor(0.).to(args.device)
    for out in output:
        if "seq" in out:
            seq_loss += loss_smoothed(
                S,
                out["seq"],
                mask_for_loss,
                no_smoothing=args.no_smoothing,
                ignore_unknown=args.ignore_unknown_residues,
            )
        if "att_mse" in out:
            mse_loss += 0.1 * get_mse_loss(out["att_mse"], mask)
        if "coords" in out:
            struct_loss += (
                args.struct_loss_weight
                * get_coors_loss(out["coords"], X, mask_for_loss, loss_type=args.struct_loss_type)
                / (3.5**2)
            )
        if "node_vecs" in out:
            orientation_loss += get_orientation_loss(out["node_vecs"][:, :, 0], vecs_gt[:, :, 0], mask_for_loss.bool())

    true_false, rmsd, pp = loss_nll(
        S, out.get("seq"), mask_for_loss, X, out.get("coords"), ignore_unknown=args.ignore_unknown_residues
    )
    acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
    
    acc_orientation = get_orientation_accuracy(out.get("node_vecs"), vecs_gt, mask_for_loss.bool()).cpu().data.numpy()
    
    weights = torch.sum(mask_for_loss).cpu().data.numpy()
    total_loss = (seq_loss + mse_loss + struct_loss + orientation_loss)
    return (
        total_loss,
        struct_loss.detach().clone(),
        seq_loss.detach().clone(),
        orientation_loss.detach().clone(),
        rmsd,
        acc,
        acc_orientation,
        pp,
        weights,
    )

def get_loss(batch, optimizer, args, model, sidechain_net=None):
    device = args.device
    optional_feature_names = {
        "scalar_seq": ["chemical", "chem_topological"],
        "scalar_struct": ["dihedral", "topological", "mask_feature", "secondary_structure"],
        "vector_node_seq": ["sidechain"],
        "vector_node_struct": [],
        "vector_edge_seq": [], # not implemented
        "vector_edge_struct": [], # not implemented
    }
    model_args = {}
    model_args["chain_M"] = get_masked_sequence(
        batch,
        lower_limit=args.lower_masked_limit,
        upper_limit=args.upper_masked_limit,
        mask_whole_chains=args.mask_whole_chains,
        force_binding_sites_frac=args.train_force_binding_sites_frac,
        mask_frac=args.masking_probability,
    ).to(dtype=torch.float32, device=device)
    model_args["X"] = batch["X"].to(dtype=torch.float32, device=device)
    model_args["S"] = batch["S"].to(dtype=torch.long, device=device)
    model_args["optional_features"] = {}
    for k, v in optional_feature_names.items():
        if k.startswith("scalar"):
            model_args["optional_features"][k] = torch.cat([batch[x] for x in v if x in batch], dim=2).to(
                dtype=torch.float32, device=device
            ) if any([x in batch for x in v]) else None
        elif k.startswith("vector"):
            model_args["optional_features"][k] = torch.stack([batch[x] for x in v if x in batch], dim=2).to(
                dtype=torch.float32, device=device
            ) if any([x in batch for x in v]) else None
    model_args["residue_idx"] = batch["residue_idx"].to(dtype=torch.long, device=device)
    model_args["chain_encoding_all"] = batch["chain_encoding_all"].to(
        dtype=torch.long, device=device
    )
    model_args["mask"] = batch["mask"].to(dtype=torch.float32, device=device)
    model_args["mask_original"] = batch["mask_original"].to(
        dtype=torch.float32, device=device
    )
    model_args["global_context"] = batch["global_context"].to(
        dtype=torch.float32, device=device
    )

    optimizer.zero_grad()

    if not args.no_mixed_precision:
        with torch.cuda.amp.autocast():
            loss, struct_loss, seq_loss, orientation_loss, rmsd, acc, acc_orientation, pp, weights = compute_loss(
                model_args, args, model, sidechain_net
            )
    else:
        loss, struct_loss, seq_loss, orientation_loss, rmsd, acc, acc_orientation, pp, weights = compute_loss(
            model_args, args, model, sidechain_net
        )

    return loss, struct_loss, seq_loss, orientation_loss, acc, acc_orientation, rmsd, pp, weights

def main(args, trial=None):
    # torch.autograd.set_detect_anomaly(True)

    scaler = torch.cuda.amp.GradScaler()

    args.device = torch.device(args.device)

    base_folder = time.strftime(args.output_path, time.localtime())

    if base_folder[-1] != "/":
        base_folder += "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ["model_weights"]
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = ""
    if args.load_experiment is not None:
        PATH = get_trained_model_path(args.load_experiment, args.load_epoch_mode)

    logfile = base_folder + "log.txt"
    if not PATH:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    if args.train_for_sidechain_orientation:
        args.use_orientation_loss = True
    
    if args.use_orientation_loss or args.use_sidechain_pretrained:
        args.use_sidechain_orientation = True

    DATA_PARAM = {
        "features_folder": args.features_path,
        "max_length": args.max_protein_length,
        "min_length": args.min_protein_length,
        "rewrite": args.force,
        "debug": args.debug,
        "load_to_ram": args.load_to_ram,
        "interpolate": args.interpolate,
        "node_features_type": args.node_features,
        "use_global_topological_context": args.use_global_topological_context,
        "batch_size": args.batch_size,
    }

    training_dict = (
        None
        if args.clustering_dict_path is None
        else os.path.join(args.clustering_dict_path, "train.pickle")
    )
    validation_dict = (
        None
        if args.clustering_dict_path is None
        else os.path.join(args.clustering_dict_path, "valid.pickle")
    )
    test_dict = (
        None
        if args.clustering_dict_path is None
        else os.path.join(args.clustering_dict_path, "test.pickle")
    )

    print("\nDATA LOADING")
    if not args.test:
        train_loader = ProteinLoader(
            dataset_folder=os.path.join(args.dataset_path, "training"),
            clustering_dict_path=training_dict,
            shuffle_clusters=not args.not_shuffle_clusters,
            **DATA_PARAM,
        )
        valid_loader = ProteinLoader(
            dataset_folder=os.path.join(args.dataset_path, "validation"),
            clustering_dict_path=validation_dict,
            shuffle_clusters=False,
            **DATA_PARAM,
        )
    else:
        test_loader = ProteinLoader(
            dataset_folder=os.path.join(args.dataset_path, "test"),
            clustering_dict_path=test_dict,
            shuffle_clusters=False,
            **DATA_PARAM,
        )

    if args.mask_attention != "none" and args.interpolate == "none":
        args.interpolate = "zeros"

    if args.use_sidechain_pretrained:

        args_pretrained = argparse.Namespace()
        args_pretrained.in_dim = 128
        args_pretrained.hidden_dim = 128
        args_pretrained.vector_dim = 5
        args_pretrained.num_encoder_layers = 3
        args_pretrained.num_decoder_layers = 3
        args_pretrained.num_encoder_mpnn_layers = 2
        args_pretrained.num_decoder_mpnn_layers = 2
        args_pretrained.num_neighbors = 32
        args_pretrained.dropout = 0.1

        sidechain_net = ProteinMPNN(
            args_pretrained,
            encoder_type="gvp",
            decoder_type="gvp",
            use_esm_embeddings=False,
            k_neighbors=32,
            augment_eps=0.2,
            embedding_dim=128,
            ignore_unknown=False,
            mask_attention="none",
            node_features_type="zeros",
            only_c_alpha=False,
            predict_structure=False,
            noise_unknown=None,
            n_cycles=1,
            no_sequence_in_encoder=False,
            cycle_over_embedding=False,
            seq_init_mode="zeros",
            double_sequence_features=False,
            use_global_feats_attention_dim=False,
            hidden_dim=128,
            predict_node_vectors="only_vectors",
        )
        sidechain_net.load_state_dict(torch.load(args.sidechain_net_path)["model_state_dict"])
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            sidechain_net = nn.DataParallel(sidechain_net)
        sidechain_net.to(args.device)
        sidechain_net.eval()
    
    else:
        sidechain_net = None

    predict_node_vectors = "no_prediction"
    if args.use_sidechain_orientation:
        predict_node_vectors = "with_others"
    if args.train_for_sidechain_orientation:
        predict_node_vectors = "only_vectors"

    model = ProteinMPNN(
        args,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        k_neighbors=args.num_neighbors,
        augment_eps=args.backbone_noise,
        embedding_dim=args.embedding_dim,
        ignore_unknown=args.ignore_unknown_residues,
        mask_attention=args.mask_attention,
        node_features_type=args.node_features,
        only_c_alpha=args.only_c_alpha,
        predict_structure=args.predict_structure,
        noise_unknown=args.noise_unknown,
        n_cycles=args.n_cycles,
        no_sequence_in_encoder=args.no_sequence_in_encoder,
        cycle_over_embedding=args.cycle_over_embedding,
        seq_init_mode=args.seq_init_mode,
        double_sequence_features=args.double_sequence_features,
        use_global_feats_attention_dim=args.use_global_topological_context,
        hidden_dim=args.hidden_dim,
        separate_modules_num=args.separate_modules_num,
        predict_node_vectors=predict_node_vectors,
        only_cycle_over_decoder=args.only_cycle_over_decoder,
        random_connections_frac=args.random_connections_frac,
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint["step"]  # write total_step from the checkpoint
        epoch = checkpoint["epoch"]  # write epoch from the checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step, lr=args.lr)

    if PATH:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if not args.test:
        print("\nTRAINING")

        best_res = 0
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0.0, 0.0
            struct_loss_sum, seq_loss_sum, orientation_loss_sum = 0.0, 0.0, 0.0
            train_acc = 0.0
            train_acc_orientation = 0.0
            train_rmsd = 0.0
            train_pp = 0.0
            if args.skip_tqdm:
                loader = train_loader
            else:
                loader = tqdm(train_loader)
            for batch in loader:
                with torch.autograd.set_detect_anomaly(True):
                    loss, struct_loss, seq_loss, orientation_loss, acc, acc_orientation, rmsd, pp, weights = get_loss(
                        batch, optimizer, args, model, sidechain_net
                    )
                if not args.no_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_sum += loss.detach()
                struct_loss_sum += struct_loss.detach()
                seq_loss_sum += seq_loss.detach()
                orientation_loss_sum += orientation_loss.detach()
                train_acc += acc
                train_acc_orientation += acc_orientation
                train_weights += weights
                train_rmsd += rmsd
                train_pp += pp

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0.0, 0.0
                validation_acc = 0.0
                validation_acc_orientation = 0.0
                valid_rmsd = 0.0
                valid_pp = 0.0
                if args.skip_tqdm:
                    loader = valid_loader
                else:
                    loader = tqdm(valid_loader)
                for batch in loader:
                    loss, struct_loss, seq_loss, orientation_loss, acc, acc_orientation, rmsd, pp, weights = get_loss(
                        batch, optimizer, args, model, sidechain_net
                    )
                    validation_sum += loss.detach()
                    validation_acc += acc
                    validation_acc_orientation += acc_orientation
                    valid_rmsd += rmsd
                    valid_pp += pp
                    validation_weights += weights

            train_accuracy = train_acc / train_weights
            validation_accuracy = validation_acc / validation_weights
            train_accuracy_orientation = train_acc_orientation / train_weights
            validation_accuracy_orientation = validation_acc_orientation / validation_weights
            train_rmsd = train_rmsd / len(train_set)
            valid_rmsd = valid_rmsd / len(valid_set)
            train_pp = train_pp / len(train_set)
            valid_pp = valid_pp / len(valid_set)
            train_loss = float(train_sum / len(train_set))
            struct_loss = float(struct_loss_sum / len(train_set))
            seq_loss = float(seq_loss_sum / len(train_set))
            orientation_loss = float(orientation_loss_sum / len(train_set))
            validation_loss = float(validation_sum / len(valid_set))

            train_accuracy_ = np.format_float_positional(
                np.float32(train_accuracy), unique=False, precision=3
            )
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3
            )
            train_accuracy_orientation_ = np.format_float_positional(
                np.float32(train_accuracy_orientation), unique=False, precision=3
            )
            validation_accuracy_orientation_ = np.format_float_positional(
                np.float32(validation_accuracy_orientation), unique=False, precision=3
            )

            t1 = time.time()
            dt = np.format_float_positional(
                np.float32(t1 - t0), unique=False, precision=1
            )
            epoch_string = f"epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss:.2f}, valid: {validation_loss:.2f}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, train_rmsd: {train_rmsd:.2f}, valid_rmsd: {valid_rmsd:.2f}, train_pp: {train_pp:.2f}, valid_pp: {valid_pp:.2f}\n"

            with open(logfile, "a") as f:
                f.write(epoch_string)
            print(epoch_string)

            if trial is not None:  # optuna trial for hyperparameter optimization
                if args.train_for_sidechain_orientation:
                    trial.report(validation_accuracy_orientation, e)
                trial.report(validation_accuracy, e)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            checkpoint_filename_last = (
                base_folder + "model_weights/epoch_last.pt".format(e + 1, total_step)
            )
            torch.save(
                {
                    "epoch": e + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_filename_last,
            )

            best_epoch = False
            if args.predict_structure:
                if valid_rmsd < best_res:
                    best_epoch = True
                    best_res = valid_rmsd
            elif validation_accuracy > best_res:
                best_epoch = True
                best_res = validation_accuracy
            if best_epoch:
                checkpoint_filename_best = (
                    base_folder
                    + "model_weights/epoch_best.pt".format(e + 1, total_step)
                )
                torch.save(
                    {
                        "epoch": e + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_filename_best,
                )


            if (e + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = (
                    base_folder
                    + "model_weights/epoch{}_step{}.pt".format(e + 1, total_step)
                )
                torch.save(
                    {
                        "epoch": e + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_filename,
                )
        
        if args.train_for_sidechain_orientation:
            return validation_accuracy_orientation
        
        return validation_accuracy
    
    else:
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            validation_acc_orientation = 0.0
            valid_rmsd = 0.0
            valid_pp = 0.0
            for batch in tqdm(test_loader):
                loss, struct_loss, seq_loss, orientation_loss, acc, acc_orientation, rmsd, pp, weights = get_loss(
                        batch, optimizer, args, model, sidechain_net
                )
                validation_sum += loss.detach()
                validation_acc += acc
                valid_rmsd += rmsd
                valid_pp += pp
                validation_weights += weights

            validation_accuracy = validation_acc / validation_weights
            valid_rmsd = valid_rmsd / len(test_set)
            valid_pp = valid_pp / len(test_set)
            validation_loss = float(validation_sum / len(test_set))

            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3
            )

            if args.train_for_sidechain_orientation:
                print(f"test_acc: {validation_accuracy_orientation_}, test_rmsd: {valid_rmsd:.2f}, test_pp: {valid_pp:.2f}")
            
            print(f"test_acc: {validation_accuracy_}, test_rmsd: {valid_rmsd:.2f}, test_pp: {valid_pp:.2f}")

def parse(command = None):
    if command is not None:
        sys.argv = command.split()
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/ubuntu/ml-4/data/best_prot",
        help="path for loading training data (a folder with training, test and validation subfolders)",
    )
    argparser.add_argument(
        "--features_path",
        type=str,
        default="/home/ubuntu/ml-4/data/tmp_features",
        help="path where ProteinMPNN features will be saved",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="./exp_020",
        help="path for logs and model weights",
    )
    argparser.add_argument(
        "--clustering_dict_path",
        type=str,
        default="/home/ubuntu/ml-4/data/best_prot/splits_dict",
        help="path to a folder containing train.pickle, valid.pickle and test.pickle clustering files",
    )
    argparser.add_argument(
        "--load_experiment",
        type=str,
        default=None,
        help="path for previous model weights, e.g. file.pt",
    )
    argparser.add_argument(
        "--load_epoch_mode",
        choices=["last", "best"],
        default="last",
        help="the mode for loading the model weights",
    )
    argparser.add_argument(
        "--num_epochs", type=int, default=100, help="number of epochs to train for"
    )
    argparser.add_argument(
        "--save_model_every_n_epochs",
        type=int,
        default=10,
        help="save model weights every n epochs",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=8, help="number of tokens for one batch"
    )
    argparser.add_argument(
        "--min_protein_length",
        type=int,
        default=30,
        help="minimum length of the protein complex",
    )
    argparser.add_argument(
        "--max_protein_length",
        type=int,
        default=2000,
        help="maximum length of the protein complex",
    )
    argparser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden model dimension"
    )
    argparser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="number of encoder layers"
    )
    argparser.add_argument(
        "--num_decoder_layers", type=int, default=3, help="number of decoder layers"
    )
    argparser.add_argument(
        "--num_encoder_mpnn_layers",
        type=int,
        default=1,
        help="Number of stacked GVPs to use in one aggregation GVP layer inside the encoder (need encoder_type='gvp' to be used)"
    )
    argparser.add_argument(
        "--num_decoder_mpnn_layers",
        type=int,
        default=1,
        help="Number of stacked GVPs to use in one aggregation GVP layer inside the decoder (need decoder_type='gvp' to be used)"
    )
    argparser.add_argument(
        "--num_neighbors",
        type=int,
        default=32,
        help="number of neighbors for the sparse graph",
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout"
    )
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.2,
        help="amount of noise added to backbone during training",
    )
    # argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument(
        "--gradient_norm",
        type=float,
        default=-1.0,
        help="clip gradient norm, set to negative to omit clipping",
    )
    argparser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="train without mixed precision",
    )

    # New parameters
    argparser.add_argument(
        "--lower_masked_limit",
        type=int,
        default=15,
        help="The minimum number of residues to mask in each protein (ignored if mask_whole_chains is true or masking_probability is specified)",
    )
    argparser.add_argument(
        "--upper_masked_limit",
        type=int,
        default=100,
        help="The maximum number of residues to mask in each protein (ignored if mask_whole_chains is true or masking_probability is specified)",
    )
    argparser.add_argument(
        "--masking_probability",
        type=float,
        help="The probability of masking a residue (if specified, overrides lower_masked_limit and upper_masked_limit)",
    )
    argparser.add_argument(
        "--mask_whole_chains",
        action="store_true",
        help="if true, lower_masked_limit, upper_masked_limit, masking_probability are ignored and entire chains are masked",
    )
    argparser.add_argument(
        "--force", action="store_true", help="If true, rewrite existing feature files"
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="The name of the torch device"
    )
    argparser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="The dimension of the residue type embedding",
    )
    argparser.add_argument(
        "--no_smoothing",
        action="store_true",
        help="Use a regular cross-entropy loss without smooting the one-hot encoding",
    )
    argparser.add_argument(
        "--use_mean",
        action="store_true",
        help="Use mean over existing neighbors for aggregation instead of sum",
    )
    argparser.add_argument(
        "--use_egnn_for_refinement",
        action="store_true",
        help="Use an additional EGNN pre-processing layer for sructure refinement",
    )
    argparser.add_argument(
        "--predict_structure",
        action="store_true",
        help="Predict the structure of the protein instead of the sequence",
    )
    argparser.add_argument(
        "--small_dataset", action="store_true", help="Use 0.1 of the training clusters"
    )
    argparser.add_argument(
        "--train_force_binding_sites_frac",
        type=float,
        default=0.15,
        help="If > 0, this fraction of regions sampled in polymer chains will be forced to be around the binding sites (in training)",
    )
    argparser.add_argument(
        "--val_force_binding_sites_frac",
        type=float,
        default=0.15,
        help="If > 0, this fraction of regions sampled in polymer chains will be forced to be around the binding sites (in validation)",
    )
    argparser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set instead of training (make sure to set previous_checkpoint)",
    )
    argparser.add_argument(
        "--ignore_unknown_residues",
        action="store_true",
        help="Predict 20 aminoacids; if the residue type is unknown in the ground truth, mask it in loss calculation",
    )
    argparser.add_argument(
        "--debug", action="store_true", help="Only process 1000 files per subset"
    )
    argparser.add_argument(
        "--load_to_ram",
        action="store_true",
        help="Load the data to RAM (use with caution! make sure the RAM is big enough!)",
    )
    argparser.add_argument(
        "--mask_attention",
        choices=["none", "only_missing_values", "use_loss", "vanilla"],
        default="none",
        help="Apply an attention layer before the encoder",
    )
    argparser.add_argument(
        "--interpolate",
        choices=["none", "only_middle", "all"],
        default="none",
        help="Choose none for no interpolation, only_middle for only linear interpolation in the middle, all for linear interpolation + ends generation",
    )
    argparser.add_argument(
        "--node_features",
        default="zeros",
        help='The node features type; choices = ["zeros", "dihedral", "chemical", "topological", "mask", "chem_topological", "esm"] and combinations (e.g. "chemical+sidechain")',
    )
    argparser.add_argument(
        "--use_sidechain_orientation",
        action="store_true",
        help="Use sidechain orientation unit vectors",
    )
    argparser.add_argument(
        "--use_global_topological_context",
        action="store_true",
        help="Use topological context features",
    )
    argparser.add_argument(
        "--only_c_alpha",
        action="store_true",
        help="Use less features for the edges"
    )
    argparser.add_argument(
        "--no_edge_update", action="store_true", help="Skip edge update in encoder"
    )
    argparser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If None, NoamOpt is used, otherwise Adam with this starting learning rate",
    )
    argparser.add_argument(
        "--struct_loss_weight",
        type=float,
        default=1,
        help="Weight used to balance the structure loss",
    )
    argparser.add_argument(
        "--debug_file",
        default=None,
        type=str,
        help="If not None, open a specific file instead of loading the dataset",
    )
    argparser.add_argument(
        "--noise_unknown",
        default=None,
        type=float,
        help="The noise level to apply to masked structure (by default same as backbone_noise)",
    )
    argparser.add_argument(
        "--no_sequence_in_encoder",
        action="store_true",
        help="If true, one-shot decoders will only use sequence features in the decoder, like autoregressive do"
    )
    argparser.add_argument(
        "--n_cycles",
        default=1,
        type=int,
        help="Number of refinement cycles (1 = only prediction, no refinement)"
    )
    argparser.add_argument(
        "--seq_init_mode",
        choices=["zeros", "random", "esm_one_hot", "esm_probabilities", "uniform_probabilities"],
        default="zeros",
        help="The sequence initialization mode for one-shot decoders"
    )
    argparser.add_argument(
        "--cycle_over_embedding",
        action="store_true",
        help="Run cycles at the embedding level instead of the sequence level"
    )
    argparser.add_argument(
        "--double_sequence_features",
        action="store_true",
        help="If true, sequence is used both in the encoder and in the decoder"
    )
    argparser.add_argument(
        "--decoder_type",
        choices=["mpnn", "mpnn_auto", "egnn", "gps", "gvp", "mpnn_enc"],
        default="mpnn_auto"
    )
    argparser.add_argument(
        "--encoder_type",
        choices=["mpnn", "egnn", "gps", "gvp"],
        default="mpnn"
    )
    argparser.add_argument(
        "--use_attention_in_encoder",
        action="store_true",
        help="If true, add global node attention to (MPNN) encoder layers"
    )
    argparser.add_argument(
        "--use_attention_in_decoder",
        action="store_true",
        help="If true, add global node attention to (MPNN) decoder layers"
    )
    argparser.add_argument(
        "--separate_modules_num",
        default=1,
        type=int,
        help="The number of separate modules to use for recycling (if n_cycles > separate_modules_num, the last module is used for all remaining cycles)"
    )
    argparser.add_argument(
        "--struct_loss_type",
        choices=["mse", "huber", "relaxed_mse"],
        help="The type of the structure loss",
        default="mse"
    )
    argparser.add_argument(
        "--use_edges_for_structure",
        action="store_true",
        help="In case structure is predicted with force updates, use edge features in computation"
    )
    argparser.add_argument(
        "--use_orientation_loss",
        action="store_true",
        help="If True, add a loss on sidechain orientation during training"
    )
    argparser.add_argument(
        "--train_for_sidechain_orientation",
        action="store_true",
        help="If True, the model is trained to only predict sidechain orientation"
    )
    argparser.add_argument(
        "--use_sidechain_pretrained",
        action="store_true",
        help="If True, uses a model that has been pre-trained to predict sidechain orientations"
    )
    argparser.add_argument(
        "--sidechain_net_path",
        type=str,
        help="The path to the pre-trained model that predicts sidechain orientations",
        default="/home/ubuntu/proteinmpnn/outputs/test_gvp_for_sidechains/model_weights/epoch_last.pt"
    )
    argparser.add_argument(
        "--skip_tqdm",
        action="store_true",
        help="Skip drawing the tqdm progressbars for epoch progress"
    )
    argparser.add_argument(
        "--only_cycle_over_decoder",
        action="store_true",
        help="If true (and cycle_over_embedding is true, and n_cycles > 1), only cycle over the decoder, not the encoder"
    )
    argparser.add_argument(
        "--not_shuffle_clusters",
        action="store_true",
        help="Use a fixed representative for each cluster instead of shuffling them"
    )
    argparser.add_argument(
        "--use_pna_in_encoder",
        action="store_true",
        help="Use PNA aggregation in the encoder"
    )
    argparser.add_argument(
        "--use_pna_in_decoder",
        action="store_true",
        help="Use PNA aggregation in the decoder"
    )
    argparser.add_argument(
        "--random_connections_frac",
        type=float,
        default=0.0,
        help="The fraction of random connections to add to the graph"
    )

    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(args)