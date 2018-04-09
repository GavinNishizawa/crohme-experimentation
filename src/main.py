#!/usr/bin/env python3
import os
from glob import glob
import numpy as np
import pandas as pd
import argparse
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup

IMG_SIZE=100


# convert trace string to a list of coordinates
def process_trace_str(s):
    to_int = lambda v: round(float(v))
    to_coord = lambda s: tuple(map(to_int, s.strip().split()))
    trace =  [ to_coord(p) for p in s.strip().split(',')]
    return trace


# convert traces into (id, coordinate list) pairs
def process_traces(ts):
    to_id_trace = lambda t: (t['id'].replace('"',''), process_trace_str(t.string))
    id_traces = [ to_id_trace(t) for t in ts]

    # collect points from all traces
    pts = []
    for t in id_traces:
        pts.extend(t[1])

    # find min and max x and y values
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    ar = (max_y-min_y)/float(max_x-min_x+1)
    print(ar)

    for id_t in id_traces:
        # center each point in the trace
        for i in range(len(id_t[1])):
            pt = id_t[1][i]

            # center points
            scale = lambda a,b,c: abs(IMG_SIZE*float(a-b)/(c-b+1))
            id_t[1][i] = (scale(pt[0], min_x, max_x), scale(pt[1], min_y, max_y))

    return id_traces


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

    return soup.find(select_fn).string.replace('"','')


def get_GT_symbol(fn, gt_df):
    #print("FILE:",fn,"\n", gt_df[gt_df["fn"] == fn])
    return gt_df[gt_df["fn"] == fn]['gt'].values[0]


def process_inkml(fn, gt_df):
    soup = xml_to_soup(fn)

    sfn = get_soup_fn(soup)
    print(fn,":",sfn,":", get_GT_symbol(sfn, gt_df))

    traces = process_traces(soup.find_all('trace'))
    '''
    for t in traces:
        print("Trace",t[0],":\n", t[1] )
    '''

    im = Image.new('1', (IMG_SIZE,IMG_SIZE))
    draw = ImageDraw.Draw(im)
    for t in traces:
        last = None
        for c in t[1]:
            if None != last:
                last.extend(c)
                draw.line(last, fill=128)
            last = list(c)
    #im.show()
    im.save(fn+".bmp")



def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of inkml files using a ground truth reference file.")
    parser.add_argument("fdir", type=str, help="directory containing .inkml files")
    parser.add_argument("gt", type=str, help="ground truth file")

    return parser.parse_args()


def main():
    print("hello world")
    args = parse_args()

    # read in the ground truth file
    df = pd.read_csv(args.gt, names=["fn","gt"])
    # print(df.head)

    for f in glob(os.path.join(args.fdir, "*.inkml")):
        process_inkml(f, df)


if __name__=="__main__":
    main()

