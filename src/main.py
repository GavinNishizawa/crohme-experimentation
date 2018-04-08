#!/usr/bin/env python3
#import os
import numpy as np
import pandas as pd
import argparse
#import PIL
from bs4 import BeautifulSoup


# convert trace string to a list of coordinates
def process_trace_str(s):
    to_int = lambda v: round(float(v))
    to_coord = lambda s: tuple(map(to_int, s.strip().split()))
    trace =  np.array([ to_coord(p) for p in s.strip().split(',')])
    return trace


# convert traces into (id, coordinate list) pairs
def process_traces(ts):
    to_id_trace = lambda t: (t['id'], process_trace_str(t.string))
    traces = [ to_id_trace(t) for t in ts]
    return traces


# convert filename of xml file to beautiful soup object
def xml_to_soup(fn):
    soup = None
    with open(fn) as f:
        soup = BeautifulSoup(f, "xml")

    return soup


# get the filename from inkml soup
def get_soup_fn(soup):
    # find the tag containing the filename
    select_fn = lambda tag: \
            tag.name == 'annotation' and tag['type'] == "UI"

    return soup.find(select_fn).string


def get_GT_symbol(fn, gt_df):
    return gt_df[gt_df["fn"] == fn]['gt'].values[0]


def process_inkml(fn, gt_df):
    soup = xml_to_soup(fn)

    sfn = get_soup_fn(soup)
    print(sfn,":", get_GT_symbol(sfn, gt_df))

    traces = process_traces(soup.find_all('trace'))
    for t in traces:
        print("Trace",t[0],":\n", t[1] )


def parse_args():
    parser = argparse.ArgumentParser(description="Process an inkml file using a ground truth reference file.")
    parser.add_argument("file", type=str, help=".inkml file")
    parser.add_argument("gt", type=str, help="ground truth file")

    return parser.parse_args()


def main():
    print("hello world")
    args = parse_args()

    # read in the ground truth file
    df = pd.read_csv(args.gt, names=["fn","gt"])
    # print(df.head)

    process_inkml(args.file, df)


if __name__=="__main__":
    main()

