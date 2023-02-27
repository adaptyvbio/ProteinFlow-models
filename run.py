import argparse
import os.path
import time, os
import numpy as np
import torch
import torch.nn as nn
import os.path
from training.model import (
    ProteinMPNN,
)
from tqdm import tqdm
from training.model_utils import *
import sys
from proteinflow import ProteinLoader


def get_trained_model_path(load_experiment, epoch_mode):
    if epoch_mode == "last":
        filename = "epoch_last.pt"
    elif epoch_mode == "best":
        filename = "epoch_best.pt"
    model_path = os.path.join("experiments", load_experiment, "model_weights", filename)
    return model_path

def initialize_sequence(seq, chain_M, seq_init_mode):
    if seq_init_mode == "zeros":
        seq[chain_M.bool()] = 0
    elif seq_init_mode == "random":
        seq[chain_M.bool()] = torch.randint(size=seq[chain_M.bool()].shape, low=1, high=22)
    return seq

def compute_loss(model_args, args, model):
    S = model_args["S"]
    mask = model_args["mask"]
    chain_M = model_args["chain_M"]
    mask_for_loss = mask * chain_M
    
    output = model(**model_args, test=args.test)
    seq_loss = torch.tensor(0.).to(args.device)
    for out in output:
        seq_loss += loss_smoothed(
            S,
            out["seq"],
            mask_for_loss,
            no_smoothing=False,
            ignore_unknown=False,
        )

    true_false, pp = loss_nll(
        S, out.get("seq"), mask_for_loss, ignore_unknown=False
    )
    pp = pp.cpu().data.numpy()
    acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy()
    
    weights = torch.sum(mask_for_loss).cpu().data.numpy()
    return (
        seq_loss,
        acc,
        pp,
        weights,
    )

def get_loss(batch, optimizer, args, model):
    device = args.device
    optional_feature_names = {
        "scalar_seq": ["chemical"],
        "scalar_struct": ["dihedral", "secondary_structure"],
        "vector_node_seq": ["sidechain_orientation"],
        "vector_node_struct": [],
        "vector_edge_seq": [], # not implemented
        "vector_edge_struct": [], # not implemented
    }
    model_args = {}
    model_args["chain_M"] = batch["masked_res"].to(dtype=torch.long, device=device)
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

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        loss, acc, pp, weights = compute_loss(
            model_args, args, model
        )
    # else:
    #     loss, acc, pp, weights = compute_loss(
    #         model_args, args, model, sidechain_net
        # )

    return loss, acc, pp, weights

def main(args):
    # torch.autograd.set_detect_anomaly(True)
    args.output_path = os.path.join("experiments", args.experiment_name)

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
    with open(logfile, "w") as f:
        f.write("Epoch\tTrain\tValidation\n")

    DATA_PARAM = {
        "features_folder": args.features_path,
        "max_length": args.max_protein_length,
        "rewrite": True,
        "debug": args.debug,
        "load_to_ram": False,
        "interpolate": False,
        "node_features_type": args.node_features,
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
            dataset_folder=os.path.join(args.dataset_path, "train"),
            clustering_dict_path=training_dict,
            shuffle_clusters=not args.not_shuffle_clusters,
            **DATA_PARAM,
        )
        valid_loader = ProteinLoader(
            dataset_folder=os.path.join(args.dataset_path, "valid"),
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

    model = ProteinMPNN(
        args,
        encoder_type="mpnn",
        decoder_type=args.decoder_type,
        k_neighbors=args.num_neighbors,
        augment_eps=args.backbone_noise,
        embedding_dim=128,
        ignore_unknown=False,
        mask_attention=False,
        node_features_type=args.node_features,
        only_c_alpha=False,
        noise_unknown=None,
        n_cycles=args.n_cycles,
        no_sequence_in_encoder=True,
        double_sequence_features=False,
        hidden_dim=128,
        separate_modules_num=args.separate_modules_num,
    )
    if torch.cuda.device_count() > 1 and args.device == "cuda":
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

    optimizer = get_std_opt(model.parameters(), 128, total_step, lr=args.lr)

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
            train_acc = 0.0
            train_pp = 0.0
            loader = tqdm(train_loader)
            for batch in loader:
                with torch.autograd.set_detect_anomaly(True):
                    loss, acc, pp, weights = get_loss(
                        batch, optimizer, args, model
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # else:
                #     loss.backward()
                #     optimizer.step()

                train_sum += loss.detach()
                train_acc += acc
                train_weights += weights
                train_pp += pp

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0.0, 0.0
                validation_acc = 0.0
                valid_pp = 0.0
                loader = tqdm(valid_loader)
                for batch in loader:
                    loss, acc, pp, weights = get_loss(
                        batch, optimizer, args, model
                    )
                    validation_sum += loss.detach()
                    validation_acc += acc
                    valid_pp += pp
                    validation_weights += weights

            length_train = len(train_loader.dataset)
            length_valid = len(valid_loader.dataset)
            train_accuracy = train_acc / train_weights
            validation_accuracy = validation_acc / validation_weights
            train_pp = train_pp / length_train
            valid_pp = valid_pp / length_valid
            train_loss = float(train_sum / length_train)
            validation_loss = float(validation_sum / length_valid)

            train_accuracy_ = np.format_float_positional(
                np.float32(train_accuracy), unique=False, precision=3
            )
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3
            )

            t1 = time.time()
            dt = np.format_float_positional(
                np.float32(t1 - t0), unique=False, precision=1
            )
            epoch_string = f"epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_loss:.2f}, valid: {validation_loss:.2f}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, train_pp: {train_pp:.2f}, valid_pp: {valid_pp:.2f}\n"

            with open(logfile, "a") as f:
                f.write(epoch_string)
            print(epoch_string)

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
            if validation_accuracy > best_res:
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


            if (e + 1) % 10 == 0:
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
        
        return validation_accuracy
    
    else:
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            valid_pp = 0.0
            for batch in tqdm(test_loader):
                loss, acc, pp, weights = get_loss(
                        batch, optimizer, args, model
                )
                validation_sum += loss.detach()
                validation_acc += acc
                valid_pp += pp
                validation_weights += weights

            length_test = len(test_loader.dataset)
            validation_accuracy = validation_acc / validation_weights
            valid_pp = valid_pp / length_test
            validation_loss = float(validation_sum / length_test)

            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3,
            )
            
            print(f"test_acc: {validation_accuracy_}, test_pp: {valid_pp:.2f}")

def parse(command = None):
    if command is not None:
        sys.argv = command.split()
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/proteinflow_20230102_stable/",
        help="path for loading training data (a folder with training, test and validation subfolders)",
    )
    argparser.add_argument(
        "--features_path",
        type=str,
        default="./data/tmp_features/",
        help="path where ProteinMPNN features will be saved",
    )
    argparser.add_argument(
        "--experiment_name",
        type=str,
        default="test",
        help="tag for the experiment, used for naming the output folder",
    )
    argparser.add_argument(
        "--clustering_dict_path",
        type=str,
        default="./data/proteinflow_20230102_stable/splits_dict",
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
        "--batch_size", type=int, default=8, help="number of tokens for one batch"
    )
    argparser.add_argument(
        "--max_protein_length",
        type=int,
        default=2000,
        help="maximum length of the protein complex",
    )
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.2,
        help="amount of noise added to backbone during training",
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="The name of the torch device"
    )
    argparser.add_argument(
        "--small_dataset", action="store_true", help="Use 0.1 of the training clusters"
    )
    argparser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set instead of training (make sure to set previous_checkpoint)",
    )
    argparser.add_argument(
        "--debug", action="store_true", help="Only process 1000 files per subset"
    )
    argparser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If None, NoamOpt is used, otherwise Adam with this starting learning rate",
    )
    argparser.add_argument(
        "--n_cycles",
        default=1,
        type=int,
        help="Number of refinement cycles (1 = only prediction, no refinement)"
    )
    argparser.add_argument(
        "--decoder_type",
        choices=["mpnn", "mpnn_auto"],
        default="mpnn"
    )
    argparser.add_argument(
        "--separate_modules_num",
        default=1,
        type=int,
        help="The number of separate modules to use for recycling (if n_cycles > separate_modules_num, the last module is used for all remaining cycles)"
    )
    argparser.add_argument(
        "--not_shuffle_clusters",
        action="store_true",
        help="Use a fixed representative for each cluster instead of shuffling them"
    )
    argparser.add_argument(
        "--num_neighbors",
        type=int,
        default=32,
        help="number of neighbors for the sparse graph",
    )
    argparser.add_argument(
        "--node_features",
        default=None,
        help='The node features type; choices = ["dihedral", "chemical", "sidechain_orientation", "secondary_structure"] and combinations (e.g. "chemical+sidechain_orientation")',
    )

    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(args)
