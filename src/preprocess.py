#!/usr/bin/env python3
import os
from glob import glob
import pandas as pd
import argparse
import pickle
from bs4 import BeautifulSoup

from feature_extraction import *


# convert trace string to a list of coordinates
def process_trace_str(s):
    to_int = lambda v: round(1000*float(v))
    to_coord = lambda s: tuple(map(to_int, s.strip().split()))
    trace =  [ to_coord(p) for p in s.strip().split(',')]
    return trace


# convert filename of xml file to beautiful soup object
def xml_to_soup(fn):
    soup = None
    with open(fn, encoding="latin-1") as f:
        soup = BeautifulSoup(f, "xml")

    return soup


# get the filename from inkml soup
def get_soup_fn(soup):
    # find the tag containing the filename
    select_fn = lambda tag: \
            tag.name == 'annotation' and tag['type'] == "UI"

    return soup.find(select_fn).string.replace('"','')


def get_GT_symbol(fn, gt_df):
    #print("FILE:",fn,"\n", gt_df[gt_df["fn"] == fn])
    return gt_df[gt_df["fn"] == fn]['gt'].values[0]


def get_process_iso(gt_df):
    return lambda fn: process_iso_inkml(fn, gt_df)


def process_iso_inkml(fn, gt_df):
    soup = xml_to_soup(fn)

    sfn = get_soup_fn(soup)
    symbol = get_GT_symbol(sfn, gt_df)
    #print(fn,":",sfn,":", symbol)

    # convert traces into (id, coordinate list) pairs
    ts = soup.find_all('trace')

    get_trace_id = lambda t: int(t['id'].replace('"',''))
    get_trace_pts = lambda t: process_trace_str(t.string)
    id_traces = { get_trace_id(t): get_trace_pts(t) for t in ts}
    trace_ids = list(id_traces.keys())
    trace_ids.sort()

    traces = [id_traces[k] for k in trace_ids]
    feat_arr = process_traces(traces)

    feat_arr.extend((fn, symbol))
    return [feat_arr]


def preprocess_dir(fdir, process):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    print("\nExtracting features...")
    processed = []
    columns= ["aspect","n_traces","width","height", \
            "fn","symbol"]
    to_process = glob(os.path.join(fdir, "**","*.inkml"), recursive=True)
    n_to_process = len(to_process)
    n_done = 0
    for f in to_process:
        processed.extend(process(f))
        n_done += 1
        p_done = n_done/n_to_process
        p20 = round(p_done*20)
        print("\r[{0}] {2}/{3} ({1}%) {4}".format( \
                '#'*p20+' '*(20-p20), round(p_done*100), \
                n_done, n_to_process, os.path.basename(f)), \
                end='')

    print("\nPerforming additional preprocessing...")
    data = pd.DataFrame(processed)
    lc = len(columns)
    ldc = len(data.columns)
    d_columns = {v:columns[v-ldc+lc] for v in data.columns[-lc:]}
    data.rename(columns=d_columns, inplace=True)

    print("\nSaving preprocessed data to disk...")
    pickle.dump(data, open(pickle_fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("\tDone!")
    return data


def process_inkml(fn):
    soup = xml_to_soup(fn)

    # get traces
    traces = soup.select("trace")
    # get trace groups
    tgs = set([v.parent for v in soup.select("traceGroup traceView")])

    # helper functions
    get_truth = lambda t: \
            t.name == 'annotation' and t['type'] == 'truth'
    get_get_trace = lambda k: lambda t:\
            t.name == 'trace' and t['id'] == k

    # collect all symbols with traceGroup data
    # e.g., {'id':'28','truth':'S','traces':[[(1,2), ..], ..]}
    feat_arrs = []
    for tg in tgs:
        # get the traceGroup id
        tg_ = {'id':tg['xml:id']}

        # get the ground truth value
        tg_['truth'] = str(tg.find(get_truth).string)

        # build the list of traces
        tvs = tg.find_all('traceView')
        tg_['traces'] = []
        for tv in tvs:
            get_trace = get_get_trace(tv['traceDataRef'])
            trc = process_trace_str(soup.find(get_trace).string)
            tg_['traces'].append(trc)

        #symbols.append(tg_)

        # extract features from traces
        feat_arr = process_traces(tg_['traces'])
        feat_arr.extend((fn, tg_['truth']))
        feat_arrs.append(feat_arr)

    return feat_arrs


def test_iso():
    data_dir_fn = os.path.join("data","trainingSymbols")
    gt_fn = os.path.join("data","trainingSymbols","iso_GT.txt")

    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    process = get_process_iso(df)
    data = preprocess_dir(data_dir_fn, process)
    print(data.head())


def test_multi():
    #fn = os.path.join("data","TrainINKML","expressmatch","101_alfonso.inkml")
    #symbols = process_inkml(fn)

    dir_fn = os.path.join("data","TrainINKML","extension")
    data = preprocess_dir(dir_fn, process_inkml)
    print(data.head())


if __name__=="__main__":
    test_multi()

