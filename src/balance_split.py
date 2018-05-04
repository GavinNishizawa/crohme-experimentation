#!/usr/bin/env python3
import os
from glob import glob
import pandas as pd
import pickle

from progress_bar import do_with_progress_bar
from read_inkml import *


def measure_error(split_counts, target_ratio):
    total_err = 0
    for k in split_counts.keys():
        # total number of k symbols so far
        total = sum(split_counts[k])

        # number of k symbols in split A
        n = split_counts[k][0]

        # current k symbol ratio
        ratio = n/total if total > 0 else 0

        # error as difference from target ratio
        err = target_ratio - ratio

        # add the squared error
        total_err += err**2
    return total_err


'''
Place file f into either split A or split B, based on the
placement which results in the lower distance to the
target ratio.
'''
def place_file_in_split(f, get_symbols, splits, split_counts, ratio):
    symbols = get_symbols(f)

    # check to see the error if placed in split A
    for s in symbols:
        if not split_counts.__contains__(s):
            split_counts[s] = [0,0]
        # update the symbol counts
        split_counts[s][0] += 1

    # calculate the error after adding the symbols to split A
    errA = measure_error(split_counts, ratio)

    # check to see the error if placed in split B
    for s in symbols:
        # update the symbol counts
        split_counts[s][0] -= 1
        split_counts[s][1] += 1
    errB = measure_error(split_counts, ratio)

    # place the file into the split that gives the lower error
    if errA < errB:
        splits[0].append(f)
        # update counts after checking spilt B error
        for s in symbols:
            split_counts[s][0] += 1
            split_counts[s][1] -= 1
    else:
        splits[1].append(f)


'''
Split files into two groups, using a greedy solution,
such that the ratio of symbol counts in each group
approximately matches the provided ratio.

e.g. If ratio is 0.5, approximately half the 'a' symbols
should be in each group, and likewise for every symbol.
'''
def split_dir(fdir, get_symbols, ratio=0.7):
    pickle_fn = fdir + str(round(ratio*100)) + "-" + \
            str(round(100*(1-ratio))) + ".symbols.pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    splits = [[] for i in range(2)]
    split_counts = {}

    func = lambda f: place_file_in_split(f, \
            get_symbols, splits, split_counts, ratio)
    to_process = glob( \
            os.path.join(fdir, "**","*.inkml"), recursive=True)
    file_to_str = lambda f: os.path.basename(f)

    print("\nPlacing files into a", \
            round(ratio*100),round(100*(1-ratio)),"split...")
    do_with_progress_bar(to_process, func, file_to_str)

    total_err = measure_error(split_counts, ratio)
    print(split_counts)
    print("Average error:",total_err/len(split_counts.keys()))

    print("\nSaving splits to disk...")
    pickle.dump(splits, open(pickle_fn, 'wb'), \
            protocol=pickle.HIGHEST_PROTOCOL)
    print("\tDone!")
    return splits


def test_iso():
    data_dir_fn = os.path.join("data","trainingSymbols")
    gt_fn = os.path.join("data","trainingSymbols","iso_GT.txt")

    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    get_symbols = get_get_symbols_iso(df)
    data = split_dir(data_dir_fn, get_symbols, 0.7)
    print(len(data[0]),len(data[1]))


def test_multi():
    #fn = os.path.join("data","TrainINKML","expressmatch","101_alfonso.inkml")
    #symbols = process_inkml(fn)

    dir_fn = os.path.join("data","TrainINKML","extension")
    data = split_dir(dir_fn, get_symbols_inkml, 0.7)
    print(len(data[0]),len(data[1]))


if __name__=="__main__":
    test_multi()
    test_iso()

