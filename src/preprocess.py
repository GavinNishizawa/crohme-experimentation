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

    dists = []
    mp_dists = []
    r_angles = []
    a_angles = []
    angle_pts = []
    mp_angle_pts = []
    for id_t in id_traces:
        trace = id_t[1]
        n_pts = len(trace)
        midpoints = [None]*n_pts
        e_dists = [0]*n_pts
        mpe_dists = [0]*n_pts
        abs_angles = [0]*n_pts
        angles = [0]*n_pts
        mp_angles = [0]*n_pts
        d_angles = [None]*n_pts
        dmp_angles = [None]*n_pts

        # calc midpoints of consecutive points in the trace
        midpoints[0] = trace[0]
        d_angles[0] = tuple((0,0))
        dmp_angles[0] = tuple((0,0))
        for i in range(1,n_pts):
            last_pt = trace[i-1]
            pt = trace[i]

            # distance
            e_dists[i] = e_dist(last_pt, pt)

            # midpoint
            midpoints[i] = midpoint(last_pt, pt)
            # midpoint distance
            mpe_dists[i] = e_dist(midpoints[i], midpoints[i-1])
            # absolute angles
            abs_angles[i] = angle(pt, last_pt)

            # relative angles
            angles[i] = angle(pt, last_pt) - angles[i-1]
            # relative angles for midpoints
            mp_angles[i] = angle(midpoints[i], midpoints[i-1]) - mp_angles[i-1]

            # total distance
            t_dist = d_angles[i-1][0] + e_dists[i]
            # (total distance, relative angle)
            d_angles[i] = tuple((t_dist, angles[i]))

            # total midpoint distance
            tmp_dist = dmp_angles[i-1][0] + mpe_dists[i]
            # (total midpoint distance, relative midpoint angle)
            dmp_angles[i] = tuple((tmp_dist,mp_angles[i]))

        id_t[1].clear()
        id_t[1].extend(midpoints)
        dists.extend(e_dists)
        mp_dists.extend(mpe_dists)
        a_angles.extend(abs_angles)
        r_angles.extend(angles)
        angle_pts.extend(d_angles)
        mp_angle_pts.extend(dmp_angles)


    # average distance between points
    ad = sum(e_dists)/len(e_dists)

    return id_traces, ar, ad, r_angles, a_angles, angle_pts, mp_angle_pts, dists, mp_dists


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

    '''
    traces: processed traces
    ar: "aspect ratio"
    ad: average distance between points
    r_angles: relative angles
    a_angles: absolute angles
    angle_pts: (total distance, relative angle) pairs
    mp_angle_pts: (total midpoint distance, relative midpoint angle) pairs
    dists: distances
    mp_dists: midpoint distances
    '''
    traces, ar, ad, r_angles, a_angles, \
            angle_pts, mp_angle_pts, dists, mp_dists \
            = process_traces(soup.find_all('trace'))

    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))

    # draw image from traces at different scales
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

    # bin distances and midpoint distances
    n_bins = 8
    max_dist = max(max(dists),1)
    dist_bins = [0]*n_bins
    for d in dists:
        ind = round(n_bins*(d/max_dist))%n_bins
        dist_bins[ind] += 1

    max_dist = max(max(mp_dists),1)
    mp_dist_bins = [0]*n_bins
    for d in mp_dists:
        ind = round(n_bins*(d/max_dist))%n_bins
        mp_dist_bins[ind] += 1

    # bin angle points
    n_bins = 30
    scaled_angle_pts = scale_angle_pts(angle_pts, n_bins)

    # bin mp angle points
    n_bins = 30
    scaled_mp_angle_pts = scale_angle_pts(mp_angle_pts, n_bins)


    im_arr.extend(r_angle_bins)
    im_arr.extend(a_angle_bins)
    im_arr.extend(dist_bins)
    im_arr.extend(mp_dist_bins)
    im_arr.extend(scaled_angle_pts)
    im_arr.extend(scaled_mp_angle_pts)
    im_arr.extend((fn, sfn, symbol, ar, ad, len(traces)))
    return im_arr


def preprocess_dir(fdir, gt_df):
    pickle_fn = fdir+".pkl"

    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    processed = []
    data = pd.DataFrame()
    columns= ["fn","symbol_fn","symbol","aspect", "avg_dist", "n_traces"]
    to_process = glob(os.path.join(fdir, "*.inkml"))
    n_to_process = len(to_process)
    n_done = 0
    for f in to_process:
        processed.append(process_inkml(f, gt_df))
        n_done += 1
        p_done = n_done/n_to_process
        p20 = round(p_done*20)
        print("\r[{0}] {2}/{3} ({1}%) {4}".format('#'*p20+' '*(20-p20), round(p_done*100), n_done, n_to_process, os.path.basename(f)), end='', flush=True)

    print("\nPerforming additional preprocessing...")
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

