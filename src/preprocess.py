#!/usr/bin/env python3
import os
import math
from glob import glob
import numpy as np
import pandas as pd
import argparse
import pickle
from PIL import Image, ImageDraw, ImageFilter
from bs4 import BeautifulSoup

IMG_SIZE=1000
SM_IMG_SIZE=3
IMG_SIZES=[3,5,7,10,13,21]


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
    scale_pt = lambda p: scale(p[0], p[1])
    scale = lambda x,y: ( \
            scale_factor*(x+x_offset), \
            scale_factor*(y+y_offset))

    return scale_pt, y_range/(x_range+1)


# convert traces into (id, coordinate list) pairs
def process_traces(ts):
    get_trace_id = lambda t: int(t['id'].replace('"',''))
    get_trace_pts = lambda t: process_trace_str(t.string)
    id_traces = { get_trace_id(t): get_trace_pts(t) for t in ts}
    trace_ids = list(id_traces.keys())
    trace_ids.sort()

    # collect points from all traces
    pts = []
    for tid in trace_ids:
        pts.extend( id_traces[tid] )

    e_dist = lambda a,b: (\
            (a[0]-b[0])**2 + \
            (a[1]-b[1])**2)**(1/2)

    # compute the total length of the strokes in the symbol
    total_symbol_length = 0
    for i in range(1,len(pts)):
        total_symbol_length += e_dist(pts[i], pts[i-1])

    # scaling function
    scale, ar = calc_scale_fn(pts)

    # scale each point
    for i in range(len(pts)):
        pts[i] = scale(pts[i])

    midpoint = lambda a,b: ( \
            (a[0]+b[0])/2, \
            (a[1]+b[1])/2)

    # convert to midpoints
    for i in range(len(pts)-1):
        pts[i] = midpoint(pts[i],pts[i+1])

    '''
    sz = 1000
    im = Image.new('1', (sz,sz))
    draw = ImageDraw.Draw(im)
    sv = IMG_SIZE/sz
    width = round(1/sz)
    draw.line(pts, fill=128, width=width)
    #im.save("my_symbol.bmp")
    '''

    angle = lambda a,b: (\
            math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))

    n_pts = len(pts)
    r_angles = [0]*n_pts
    a_angles = [0]*n_pts
    dists = [0]*n_pts
    for i in range(0, n_pts-1):
        pass
        # absolute angles
        a_angles[i] = angle(pts[i], pts[i-1])

        # relative angles
        r_angles[i] = a_angles[i] - a_angles[i-1]

        # distance
        dists[i] = e_dist(pts[i], pts[i-1])

    # average distance between points
    avg_dist = sum(dists)/len(dists)

    return pts, a_angles, r_angles, avg_dist, ar, len(trace_ids)


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


def scale_angle_pts(pts, n_bins):
    saps = [0]*n_bins
    pts.sort()
    max_dist = pts[-1][0]
    for p in pts:
        ind = round(p[0]*(n_bins-1)/(max_dist+1))
        saps[ind] = (saps[ind] + p[1])%360

    return saps


def process_inkml(fn, gt_df):
    soup = xml_to_soup(fn)

    sfn = get_soup_fn(soup)
    symbol = get_GT_symbol(sfn, gt_df)
    #print(fn,":",sfn,":", symbol)

    pts, a_angles, r_angles, avg_dist, ar, n_traces = \
            process_traces(soup.find_all('trace'))

    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))

    # draw image from traces at different scales
    im_arr = []
    for sz in IMG_SIZES:
        im = Image.new('1', (sz,sz))
        draw = ImageDraw.Draw(im)
        sv = IMG_SIZE/sz
        width = round(1/sz)
        s_pts = [scale(p,sv) for p in pts]
        draw.line(s_pts, fill=128, width=width)
        #im.save(fn+str(sz)+".bmp")
        im_arr.extend(np.array(im).flatten().tolist())

    # bin relative and absolute angles
    n_bins = 8
    r_angle_bins = [0]*n_bins
    for a in r_angles:
        ind = round(n_bins*(a%360)/360)%n_bins
        r_angle_bins[ind] += 1

    a_angle_bins = [0]*n_bins
    for a in a_angles:
        ind = round(n_bins*(a%360)/360)%n_bins
        a_angle_bins[ind] += 1

    im_arr.extend(r_angle_bins)
    im_arr.extend(a_angle_bins)
    im_arr.extend((fn, sfn, symbol, ar, avg_dist, n_traces))
    return im_arr


def preprocess_dir(fdir, gt_df):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    processed = []
    columns= ["fn","symbol_fn","symbol","aspect", "avg_dist", "n_traces"]
    to_process = glob(os.path.join(fdir, "*.inkml"))
    n_to_process = len(to_process)
    n_done = 0
    for f in to_process:
        processed.append(process_inkml(f, gt_df))
        n_done += 1
        p_done = n_done/n_to_process
        p20 = round(p_done*20)
        print("\r[{0}] {2}/{3} ({1}%) {4}".format( \
                '#'*p20+' '*(20-p20), round(p_done*100), \
                n_done, n_to_process, os.path.basename(f)), \
                end='', flush=True)

    data = pd.DataFrame(processed)
    lc = len(columns)
    ldc = len(data.columns)
    d_columns = {v:columns[v-ldc+lc] for v in data.columns[-lc:]}
    data.rename(columns=d_columns, inplace=True)
    print("\tAdding aspect values...")

    aspect_very_high = lambda a: int(a > 3)
    aspect_high = lambda a: int(a > 1.2)
    aspect_med = lambda a: int(a < 1.2 and a > 0.8)
    aspect_low = lambda a: int(a < 0.8)
    aspect_very_low = lambda a: int(a < 0.3)
    aspect_pos = lambda a: int(a > 1)

    data['aspect_pos'] = data['aspect'].apply(aspect_pos)
    data['aspect_very_high'] = data['aspect'].apply(aspect_very_high)
    data['aspect_high'] = data['aspect'].apply(aspect_high)
    data['aspect_med'] = data['aspect'].apply(aspect_med)
    data['aspect_low'] = data['aspect'].apply(aspect_low)
    data['aspect_very_low'] = data['aspect'].apply(aspect_very_low)

    print("\nSaving preprocessed data to disk...")
    pickle.dump(data, open(pickle_fn, 'wb'))
    print("\tDone!")
    return data


def main():
    data_dir_fn = os.path.join("data","train")
    gt_fn = os.path.join("data","train","iso_GT.txt")

    # read in the ground truth file
    df = pd.read_csv(gt_fn, names=["fn","gt"])

    data = preprocess_dir(data_dir_fn, df)
    print(data.head())


if __name__=="__main__":
    main()

