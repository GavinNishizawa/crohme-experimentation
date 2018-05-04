#!/usr/bin/env python3
import os
from glob import glob
import pandas as pd
import pickle

from progress_bar import do_with_progress_bar
from read_inkml import *
from feature_extraction import *
from balance_split import *


def preprocess_dir(fdir, process, get_symbols, ratio):
    pickle_fn = fdir + str(round(ratio*100)) + "-" + \
            str(round(100*(1-ratio))) + ".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    print("Getting train/test splits...")
    train_files, test_files = \
            split_dir(fdir, get_symbols, ratio)

    data = {}
    print("Processing training data...")
    data['train'] = preprocess_files(train_files, process)
    print("Processing test data...")
    data['test'] = preprocess_files(test_files, process)


    print("\nSaving preprocessed data to disk...")
    pickle.dump(data, open(pickle_fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("\tDone!")
    return data


def preprocess_files(flist, process):
    processed = []

    func = lambda f: processed.extend(process(f))
    file_to_str = lambda f: os.path.basename(f)

    print("\nExtracting features...")
    do_with_progress_bar(flist, func, file_to_str)

    print("\nPerforming additional preprocessing...")
    columns= ["aspect","n_traces","width","height", \
            "fn","symbol"]
    data = pd.DataFrame(processed)
    lc = len(columns)
    ldc = len(data.columns)
    d_columns = {v:columns[v-ldc+lc] for v in data.columns[-lc:]}
    data.rename(columns=d_columns, inplace=True)

    return data


def test_iso():
    data_dir_fn = os.path.join("data","trainingSymbols")
    gt_fn = os.path.join("data","trainingSymbols","iso_GT.txt")

    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    process = get_process_iso(df)
    get_symbols = get_get_symbols_iso(df)
    data = preprocess_dir(data_dir_fn, process, get_symbols, 0.7)
    print(len(data['train']), len(data['test']))


def test_multi():
    #fn = os.path.join("data","TrainINKML","expressmatch","101_alfonso.inkml")
    #symbols = process_inkml(fn)

    dir_fn = os.path.join("data","TrainINKML","extension")
    data = preprocess_dir(dir_fn, process_inkml, get_symbols_inkml, 0.7)
    print(len(data['train']), len(data['test']))


if __name__=="__main__":
    test_iso()
    test_multi()

