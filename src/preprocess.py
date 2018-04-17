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
MED_IMG_SZ=10


# convert trace string to a list of coordinates
def process_trace_str(s):
    to_int = lambda v: round(1000*float(v))
    to_coord = lambda s: tuple(map(to_int, s.strip().split()))
    trace =  [ to_coord(p) for p in s.strip().split(',')]
    return trace


def calc_scale_fn(pts):
    # find min and max x and y values
    df_pts = pd.DataFrame(np.array(pts))
    max_x,max_y = df_pts[[0,1]].apply(max)
    min_x,min_y = df_pts[[0,1]].apply(min)

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


def e_dist(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)


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


def bin_angles(angles, n_bins=8):
    get_bin_ind = lambda a,n: round(n*(a%360)/360)%n
    bin_size = (360/n_bins)/3

    angle_bins = [0]*n_bins
    for a in angles:
        # index of bin for angle
        ind = get_bin_ind(a, n_bins)
        angle_bins[ind] += 1

        # index of bin for angle with offset
        # forwards of 1/3 the bin size
        a_o_f = a + bin_size
        ind = get_bin_ind(a_o_f, n_bins)
        angle_bins[ind] += 1

        # index of bin for angle with offset
        # backwards of 1/3 the bin size
        a_o_b = a - bin_size
        ind = get_bin_ind(a_o_b, n_bins)
        angle_bins[ind] += 1

    total_binned = sum(angle_bins) + 1

    # scale bin counts by number of total angles
    angle_bins = [ n/total_binned for n in angle_bins]

    return angle_bins


def extract_features(traces):
    # collect points from all scaled traces
    pts = []
    for t in traces:
        pts.extend(t)

    angle = lambda a,b: (\
            math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))

    # add first and last points and angle between
    feat_arr = [pts[0][0],pts[0][1],pts[-1][0],pts[-1][1], \
            angle(pts[-1], pts[0])]

    # add first 4 trace angles
    f4_tas = [0]*4
    for i in range(min(4,len(traces))):
        f4_tas[i] = angle(traces[i][0], traces[i][-1])
    feat_arr.extend(f4_tas)

    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))

    # draw image from traces at different scales
    img_sizes = [3,5,10,50]
    for sz in img_sizes:
        im = Image.new('1', (sz,sz))
        draw = ImageDraw.Draw(im)
        sv = IMG_SIZE/sz
        for t in traces:
            s_pts = [scale(p,sv) for p in t]
            draw.line(s_pts, fill=128, width=1)
        #im.save(fn+str(sz)+".bmp")
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))

        if sz <= MED_IMG_SZ:
            feat_arr.extend(np.array(im).flatten().tolist())

        # rotated projections
        im = im.rotate(45, resample=Image.BICUBIC)
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))

        # rotated projections
        im = im.rotate(15, resample=Image.BICUBIC)
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))


    # compute the total length of the strokes in the symbol
    total_sym_len = 0
    for i in range(1,len(pts)):
        total_sym_len += e_dist(pts[i], pts[i-1])
    feat_arr.append(total_sym_len)

    n_pts = len(pts)
    feat_arr.append(n_pts)
    r_angles = [0]*n_pts
    a_angles = [0]*n_pts
    dists = [0]*n_pts
    c_dists = [0]*n_pts
    daa_pts = [None]*n_pts
    daa_pts[0] = (0,0)
    for i in range(1, n_pts):
        # absolute angles
        a_angles[i] = angle(pts[i], pts[i-1])

        # relative angles
        r_angles[i] = a_angles[i] - a_angles[i-1]

        # distance
        dists[i] = e_dist(pts[i], pts[i-1])

        # cumulative distance
        c_dists[i] = dists[i] + c_dists[i-1]

        # (IMG_SIZE * scaled distance, abs angle) pts
        daa_pts[i] = ( \
                IMG_SIZE*c_dists[i]/(total_sym_len+1), \
                a_angles[i])

    # scaling function
    daa_scale, daa_ar = calc_scale_fn(daa_pts)

    # scale each daa point
    for i in range(len(daa_pts)):
        daa_pts[i] = daa_scale(daa_pts[i])

    # average distance between points
    avg_dist = sum(dists)/len(dists)
    feat_arr.append(avg_dist)

    img_sizes = [5,10,50]
    for sz in img_sizes:
        im = Image.new('1', (sz,sz))
        draw = ImageDraw.Draw(im)
        sv = IMG_SIZE/sz
        s_pts = [scale(p,sv) for p in daa_pts]
        draw.line(s_pts, fill=128, width=1)
        #feat_arr.extend(np.array(im).flatten().tolist())
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))

        # rotated projections
        im = im.rotate(45, resample=Image.BICUBIC)
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))

        # rotated projections
        im = im.rotate(15, resample=Image.BICUBIC)
        im_df = pd.DataFrame(np.array(im))

        # project counts on x axis
        feat_arr.extend(im_df.apply(sum))
        # project counts on y axis
        feat_arr.extend(im_df.apply(sum,axis=1))

    # bin relative and absolute angles
    bin_sizes = [2,3,4,6,8,32]
    for n_bins in bin_sizes:
        feat_arr.extend(bin_angles( \
                r_angles, n_bins))
        feat_arr.extend(bin_angles( \
                a_angles, n_bins))

    return feat_arr


def process_inkml(fn, gt_df):
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

    # collect points from all traces
    pts = []
    for tid in trace_ids:
        pts.extend( id_traces[tid] )

    # calculate symbol width and height in original space
    df_pts = pd.DataFrame(np.array(pts))
    max_x,max_y = df_pts[[0,1]].apply(max)
    min_x,min_y = df_pts[[0,1]].apply(min)
    width = max_x - min_x
    height = max_y - min_y

    '''
    # compute the total length of the strokes in the symbol
    total_sym_len = 0
    for i in range(1,len(pts)):
        total_sym_len += e_dist(pts[i], pts[i-1])
    '''

    # scaling function
    scale, ar = calc_scale_fn(pts)

    # scale each point
    for i in range(len(pts)):
        pts[i] = scale(pts[i])

    s_traces = []
    total_pts = 0
    for tid in trace_ids:
        t_len = len(id_traces[tid])
        # scaled points for this trace
        s_traces.append( \
                pts[total_pts:(total_pts+t_len)])
        total_pts += t_len

    midpoint = lambda a,b: ( \
            (a[0]+b[0])/2, \
            (a[1]+b[1])/2)

    midpoint3 = lambda a,b,c: ( \
            (a[0]+b[0]+c[0])/3, \
            (a[1]+b[1]+c[1])/3)

    # convert to midpoints of 3
    for t in s_traces:
        if len(t) > 3:
            for i in range(1,len(t)-1):
                t[i] = midpoint3(t[i-1],t[i],t[i+1])

    '''
    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))
    print(s_traces)
    sz = IMG_SIZE
    im = Image.new('1', (sz,sz))
    draw = ImageDraw.Draw(im)
    sv = IMG_SIZE/sz
    for t in s_traces:
        s_pts = [scale(p,sv) for p in t]
        draw.line(s_pts, fill=128, width=1)
    im.save("my_symbol.bmp")
    '''

    feat_arr = extract_features(s_traces)

    n_traces = len(trace_ids)
    feat_arr.extend((fn, sfn, symbol, ar, n_traces, \
            width, height))
    return feat_arr


def preprocess_dir(fdir, gt_df):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    processed = []
    columns= ["fn","symbol_fn","symbol","aspect","n_traces", \
            "width","height"]
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

    print("\nPerforming additional preprocessing...")
    data = pd.DataFrame(processed)
    lc = len(columns)
    ldc = len(data.columns)
    d_columns = {v:columns[v-ldc+lc] for v in data.columns[-lc:]}
    data.rename(columns=d_columns, inplace=True)

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

