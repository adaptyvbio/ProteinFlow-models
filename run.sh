# Download data
# proteinflow download --tag 20230102_stable

# Run experiments
python run.py --experiment_name autoregressive --decoder_type mpnn_auto
python run.py --experiment_name one_shot
python run.py --experiment_name one_shot_no_shuffle --not_shuffle_cluster
python run.py --experiment_name one_shot_multi_dihedral --n_cycles 3 --separate_modules_num 2 --node_features dihedral

# Test models
python run.py --load_experiment autoregressive --decoder_type mpnn_auto --test --load_epoch_mode best
python run.py --load_experiment one_shot --test --load_epoch_mode best
python run.py --load_experiment one_shot_no_shuffle --not_shuffle_clusters --test --load_epoch_mode best
python run.py --load_experiment one_shot_multi_dihedral --n_cycles 3 --separate_modules_num 2 --node_features dihedral --test --load_epoch_mode best