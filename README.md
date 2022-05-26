# Multi-task Classification using Denoise Auto-Encoder (MCDAE)
A PyTorch implementation of the MCDAE

Below is the dataset used in this repo... After downloading the datasets, you can put them in the folder `data/elec/`

* [Classification of power quality for power facility failure response](https://aihub.or.kr/opendata/66053/download)

# How to use
First of all, download the data from the data source above.    
and then, run main.py. Below describes how to train a model using main.py.

```bash
usage: main.py [-h] [--dataset_path DATASET_PATH] [--test_size TEST_SIZE]
               [--val_size VAL_SIZE] [--normalize] [--batch_size BATCH_SIZE]
               [--epoch EPOCH] [--lr LR] [--recon_penalty RECON_PENALTY]
               [--patience PATIENCE] [--delta DELTA] [--model_path MODEL_PATH]
               [--print_log_option PRINT_LOG_OPTION] [--model_type MODEL_TYPE]
               [--in_features IN_FEATURES] [--latent_dim LATENT_DIM]
               [--n_targets N_TARGETS] [--n_labels N_LABELS] [--drop_p DROP_P]
               [--noise_scale NOISE_SCALE]
```

# Optional arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        dataset directory path: data/elec/train.csv
  --test_size TEST_SIZE
                        test size when splitting the data (float type)
  --val_size VAL_SIZE   validation size when splitting the data (float type)
  --normalize           normailze data
  --batch_size BATCH_SIZE
                        input batch size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --recon_penalty RECON_PENALTY
                        penalty term of the reconstruction error
  --patience PATIENCE   patience of early stopping condition
  --delta DELTA         significant improvement to update a model
  --model_path MODEL_PATH
                        a path to sava a model
  --print_log_option PRINT_LOG_OPTION
                        print training loss every print_log_option
  --model_type MODEL_TYPE
                        model type: 0 = single encoder layer, 1 = double
                        encoder layer
  --in_features IN_FEATURES
                        input features
  --latent_dim LATENT_DIM
                        latent vector dimension
  --n_targets N_TARGETS
                        the number of targets (tasks)
  --n_labels N_LABELS   list of the number of labels of each target (task)
  --drop_p DROP_P       drop out rate in the encoder and decoder layer
  --noise_scale NOISE_SCALE
                        scale noise
```

Please run the "example_usage.ipynb" if you find anything hard to understannd. 