# ProteinMPNN
Forked from: https://github.com/dauparas/ProteinMPNN

## Training
To train the original model, run (recommended parameters):
```
python training/run_bestprot.py --mask_whole_chains --use_mean --no_smoothing --ignore_unknown_residues --force ...
```

For debugging on a small dataset on CPU:
```
python run_bestprot.py --debug --max_protein_length 100 --device cpu --no_mixed_precision --force --features_path /home/ubuntu/ml-4/data/tmp --output_path ./exp_tmp ...
```

To run and log the experiment:
```
python run_and_log.py --run_on_spot --experiment_name ... --key ...
```

With the `--run_on_spot` option it will run on an AWS spot instance, otherwise locally. In any instance the results will be logged and uploaded to AWS S3.
**Note that for running on spot you need to set the path to the relevant key file (by default `/home/ubuntu/ml-4-keypair.pem`)**

## Hyperparameter search
To run a hyperparameter search, use the `search_and_log.py` script. It will run the search and log the results on S3, in a similar fashion to `run_and_log.py`. You can use `--search_name`, `--rewrite_experiment`, `--key`, `--run_on_spot` parameters in the same way here.

In addition, there are two new arguments: `--n_trials` is the number of optimization trials and unless `--no_pruning` is true, unsuccessful trials will be stopped early.

To set constant arguments for training, just provide the options as usual. To define the search space, use the corresponding arguments this way:
- to vary a flag, use it with `_bool_`, e.g. `--normalize _bool_`,
- to define a categorical choice, use `_list_` followed by the arguments to choose from, e.g. `--node_features _list_ topological dihedral`,
- to define an integer range, just use the argument followed by low and high boundaries, e.g. `--num_neighbors 30 50`,
- to define a float range, do the same but keep in mind that the first argument needs to either contain a `.` or an `e`, e.g. `--struct_loss_weight 1e-3 1` or `--struct_loss_weight 1. 10`,
- to define an integer or a float range on a logarithmic scale, add `_log_` before the lower bound, e.g. `--lr _log_ 1e-5 1e-2`.

You can optimize as many parameters as you want in parallel. Here is an example of a search command optimizing for structure loss and the number of encoder layers. 
```
python search_and_log.py --force --small_dataset --use_structure_loss _bool_ --num_encoder_layers 3 6 --search_name test --run_on_spot --n_trials 100 --prune
```

Run `training.aws_utils.get_search_log()` to check the search history.

## Evaluation
To evaluate a model on the test set, use the same command with `--test` and `--previous_checkpoint` options.
```
python run_bestprot.py --mask_whole_chains --dataset_path /path/to/dataset --test --load_experiment experiment_name
```

Get a `pandas` table with all experiments by running `training.aws_utils.get_experiment_log()`.

For more information, check out the `--help` option.
The dataset is assumed to be in BestProt format, in training, validation and test folders located at the dataset path.