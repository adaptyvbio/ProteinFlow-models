# Training
You can train a model with `python run.py`. Check out the `--help` option for a list of available parameters. Run `run.sh` to download the data, run some experiments and test the results.

Install the necessary requirements with `python -m pip install -r requirements.txt` first.

# Evaluation
To evaluate a model on the test set, use the same command with `--test` and `--load_experiment` options.
```bash
python run.py  --test --load_experiment experiment_name
```

For more information, check out the `--help` option.
The dataset is expected to be in ProteinFlow format, in training, validation and test folders located at the dataset path.

# References
This code was adapted from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
