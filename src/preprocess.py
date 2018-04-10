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
    scale = lambda x,y: ( \
            scale_factor*(x+x_offset), \
            scale_factor*(y+y_offset))

    return scale, y_range/(x_range+1)


# convert traces into (id, coordinate list) pairs
def process_traces(ts):
    to_id_trace = lambda t: (t['id'].replace('"',''), process_trace_str(t.string))
    id_traces = [ to_id_trace(t) for t in ts]

    # collect points from all traces
    pts = []
    for t in id_traces:
        pts.extend(t[1])

    # scaling function
    scale, ar = calc_scale_fn(pts)

    for id_t in id_traces:
        # scale each point in the trace
        for i in range(len(id_t[1])):
            pt = id_t[1][i]

            id_t[1][i] = scale(pt[0],pt[1])

    midpoint = lambda a,b: ( \
            (a[0]+b[0])/2, \
            (a[1]+b[1])/2)

    e_dist = lambda a,b: (\
            (a[0]-b[0])**2 + \
            (a[1]-b[1])**2)**(1/2)

    angle = lambda a,b: (\
            math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))

    angle_pts = []
    for id_t in id_traces:
        trace = id_t[1]
        n_pts = len(trace)
        midpoints = [None]*n_pts
        e_dists = [0]*n_pts
        angles = [0]*n_pts
        d_angles = [None]*n_pts

        # calc midpoints of consecutive points in the trace
        midpoints[0] = trace[0]
        d_angles[0] = tuple((0,0))
        for i in range(1,n_pts):
            last_pt = trace[i-1]
            pt = trace[i]

            midpoints[i] = midpoint(last_pt, pt)
            e_dists[i] = e_dist(last_pt, pt)
            angles[i] = angle(midpoints[i], midpoints[i-1]) - angles[i-1]
            t_dist = d_angles[i-1][0] + e_dists[i]
            d_angles[i] = tuple((t_dist,angles[i]))

        '''
        aug_trace = []
        for i in range(n_pts):
            aug_trace.append(midpoints[i])
            #aug_trace.append(trace[i])
        '''

        id_t[1].clear()
        id_t[1].extend(midpoints)
        angle_pts.extend(d_angles)


    # average distance between points
    ad = sum(e_dists)/len(e_dists)

    return id_traces, ar, ad, angle_pts


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
    symbol = get_GT_symbol(sfn, gt_df)
    #print(fn,":",sfn,":", symbol)

    traces, ar, ad, angle_pts = process_traces(soup.find_all('trace'))
    '''
    for t in traces:
        print("Trace",t[0],":\n", t[1] )
    '''

    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))

    im_arr = []
    for sz in IMG_SIZES:
        im = Image.new('1', (sz,sz))
        draw = ImageDraw.Draw(im)
        sv = IMG_SIZE/sz
        width = round(1/sz)
        for t in traces:
            last = None
            for c in t[1]:
                scaled = scale(c,sv)
                if None != last:
                    last.extend(scaled)
                    draw.line(last, fill=128, width=width)
                last = list(scaled)
        #im.save(fn+str(sz)+".bmp")
        im_arr.extend(np.array(im).flatten().tolist())

    # angles
    n_bins = 30
    scaled_pts = [0]*n_bins
    angle_pts.sort()
    max_dist = angle_pts[-1][0]
    for c in angle_pts:
        ind = round(c[0]*(n_bins-1)/(max_dist+1))
        scaled_pts[ind] = (scaled_pts[ind] + c[1])%360

    im_arr.extend(scaled_pts)

    return (fn, sfn, symbol, im_arr, ar, ad, len(traces))


def preprocess_dir(fdir, gt_df):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    processed = []
    to_process = glob(os.path.join(fdir, "*.inkml"))
    n_to_process = len(to_process)
    n_done = 0
    for f in to_process:
        processed.append(process_inkml(f, gt_df))
        n_done += 1
        p_done = n_done/n_to_process
        p20 = round(p_done*20)
        print("\r[{0}] {2}/{3} ({1}%) {4}".format('#'*p20+' '*(20-p20), round(p_done*100), n_done, n_to_process, os.path.basename(f)), end='', flush=True)
    print("\nDone!")

    columns=["fn","symbol_fn","symbol","image", "aspect", "avg_dist", "n_traces"]
    pdf = pd.DataFrame(processed, columns=columns)
    data = pdf['image'].apply(pd.Series)
    data[columns] = pdf[columns]

    aspect_very_high = lambda a: int(a > 3)
    aspect_high = lambda a: int(a > 1.2)
    aspect_med = lambda a: int(a < 1.2 and a > 0.8)
    aspect_low = lambda a: int(a < 0.8)
    aspect_very_low = lambda a: int(a < 0.3)
    aspect_pos = lambda a: int(a > 1)

    data['aspect_pos'] = pdf['aspect'].apply(aspect_pos)
    data['aspect_very_high'] = pdf['aspect'].apply(aspect_very_high)
    data['aspect_high'] = pdf['aspect'].apply(aspect_high)
    data['aspect_med'] = pdf['aspect'].apply(aspect_med)
    data['aspect_low'] = pdf['aspect'].apply(aspect_low)
    data['aspect_very_low'] = pdf['aspect'].apply(aspect_very_low)

    pickle.dump(data, open(pickle_fn, 'wb'))
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

