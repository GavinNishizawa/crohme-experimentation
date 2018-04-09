#!/usr/bin/env python3
import os
from glob import glob
import numpy as np
import pandas as pd
import argparse
import pickle
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup

IMG_SIZE=1000
SM_IMG_SIZE=28


# convert trace string to a list of coordinates
def process_trace_str(s):
    to_int = lambda v: round(1000*float(v))
    to_coord = lambda s: tuple(map(to_int, s.strip().split()))
    trace =  [ to_coord(p) for p in s.strip().split(',')]
    return trace


def calc_scale_fn(pts):
    # find min and max x and y values
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    x_range = max_x-min_x
    y_range = max_y-min_y
    max_range = max(x_range, y_range)
    scale_factor = IMG_SIZE/(float(max_range)+1)
    x_offset = (max_range-x_range)/2-min_x
    y_offset = (max_range-y_range)/2-min_y

    # scale point
    scale = lambda x,y: ( \
            scale_factor*(x+x_offset), \
            scale_factor*(y+y_offset))

    return scale


# convert traces into (id, coordinate list) pairs
def process_traces(ts):
    to_id_trace = lambda t: (t['id'].replace('"',''), process_trace_str(t.string))
    id_traces = [ to_id_trace(t) for t in ts]

    # collect points from all traces
    pts = []
    for t in id_traces:
        pts.extend(t[1])

    # scaling function
    scale = calc_scale_fn(pts)

    for id_t in id_traces:
        # scale each point in the trace
        for i in range(len(id_t[1])):
            pt = id_t[1][i]

            id_t[1][i] = scale(pt[0],pt[1])

    midpoint = lambda a,b: ( \
            (a[0]+b[0])/2, \
            (a[1]+b[1])/2)

    for id_t in id_traces:
        trace = id_t[1]
        n_pts = len(trace)
        midpoints = [None]*n_pts

        # calc midpoints of consecutive points in the trace
        midpoints[0] = trace[0]
        for i in range(1,n_pts):
            last_pt = trace[i-1]
            pt = trace[i]

            midpoints[i] = midpoint(last_pt, pt)

        '''
        aug_trace = []
        for i in range(n_pts):
            aug_trace.append(midpoints[i])
            #aug_trace.append(trace[i])
        '''

        id_t[1].clear()
        id_t[1].extend(midpoints)


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


def get_ellipse_pts(point_pair):
    p1 = point_pair[0]
    p2 = point_pair[1]

    x_range = abs(p1[0]-p2[0])
    y_range = abs(p1[1]-p2[1])
    x_mid = (p1[0]+p2[0])/2
    y_mid = (p1[1]+p2[1])/2

    return None


def process_inkml(fn, gt_df):
    soup = xml_to_soup(fn)

    sfn = get_soup_fn(soup)
    symbol = get_GT_symbol(sfn, gt_df)
    print(fn,":",sfn,":", symbol)

    traces = process_traces(soup.find_all('trace'))
    '''
    for t in traces:
        print("Trace",t[0],":\n", t[1] )
    '''
    halve = lambda l: list(map(lambda v: v/2, l))

    im = Image.new('1', (IMG_SIZE,IMG_SIZE))
    draw = ImageDraw.Draw(im)
    for t in traces:
        last = None
        for c in t[1]:
            if None != last:
                last.extend(c)
                width = 3 + round(IMG_SIZE/SM_IMG_SIZE)
                draw.line(last, fill=128, width=width)
            last = list(c)
    #im.show()
    im = im.resize((SM_IMG_SIZE-3,SM_IMG_SIZE-3), Image.LANCZOS)
    im = im.resize((SM_IMG_SIZE,SM_IMG_SIZE), Image.BICUBIC)
    im.save(fn+".bmp")

    return (fn, sfn, symbol, np.array(im))



def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of inkml files using a ground truth reference file.")
    parser.add_argument("fdir", type=str, help="directory containing .inkml files")
    parser.add_argument("gt", type=str, help="ground truth file")

    return parser.parse_args()


def preprocess(fdir, gt_df):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    processed = []
    for f in glob(os.path.join(fdir, "*.inkml")):
        processed.append(process_inkml(f, gt_df))

    columns=["fn","symbol_fn","symbol","image"]
    data = pd.DataFrame(processed, columns=columns)

    pickle.dump(data, open(pickle_fn, 'wb'))
    return data


def main():
    print("hello world")
    args = parse_args()

    # read in the ground truth file
    df = pd.read_csv(args.gt, names=["fn","gt"])
    # print(df.head)

    data = preprocess(args.fdir, df)
    print(data.head)



if __name__=="__main__":
    main()

