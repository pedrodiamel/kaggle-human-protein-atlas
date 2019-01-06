# Code modified by heitorrapela based on https://github.com/igorsoldatov/human_protein_atlas/blob/master/splite_folds.py
import numpy as np
import pandas as pd
import os
import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import RepeatedKFold


def arg_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--n-splits', default=4, type=int, metavar='N',
                        help='K value of K-fold (default: 4)')
    
    parser.add_argument('--n-repeats', default=1, type=int, metavar='N',
                        help='Num of repets of K-fold (default: 1)')

    parser.add_argument('--train-file', default='train.csv', type=str,
                        help='train file input')

    parser.add_argument('--output-path', default='./folds', type=str,
                        help='output path of folds')

    parser.add_argument('--debug', default=False, type=bool,
                        help='Debug flag')


    return parser

# parameters
parser = arg_parser();
args = parser.parse_args();

n_splits = args.n_splits
n_repeats = args.n_repeats
train_file = args.train_file
output_path = args.output_path
debug = args.debug


if not os.path.exists(output_path):
    os.makedirs(output_path)


data_labels = pd.read_csv(train_file)

splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

partitions = []
fold = 0

for train_idx, test_idx in splitter.split(data_labels.index.values):
    
    partition = {}
    partition["train"] = data_labels.Id.values[train_idx]
    partition["validation"] = data_labels.Id.values[test_idx]
    partitions.append(partition)

    train_labels = data_labels.loc[data_labels.Id.isin(partition["train"])]
    valid_labels = data_labels.loc[data_labels.Id.isin(partition["validation"])]
    
    train_labels.to_csv(output_path + f'/train_{fold}.csv', index=False)
    valid_labels.to_csv(output_path + f'/valid_{fold}.csv', index=False)
    
    if(debug):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        print("TRAIN:", len(train_idx), "TEST:", len(test_idx))
    
    fold = fold + 1