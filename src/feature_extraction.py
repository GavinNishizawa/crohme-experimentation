import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE=1000
MED_IMG_SZ=10


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

    # add first, last points and angle, dx, dy between
    feat_arr = [pts[0][0],pts[0][1],pts[-1][0],pts[-1][1], \
            angle(pts[-1], pts[0]), pts[-1][0]-pts[0][0], \
            pts[-1][1]-pts[0][1]]

    avg_x = lambda t: sum([p[0] for p in t])/(len(t)+1)
    avg_y = lambda t: sum([p[1] for p in t])/(len(t)+1)

    # add angles, dx, dy for first 4 traces
    f4t_as = [0]*4
    f4t_dxs = [0]*4
    f4t_dys = [0]*4
    f4t_ax = [0]*4
    f4t_ay = [0]*4
    for i in range(min(4,len(traces))):
        f4t_as[i] = angle(traces[i][0], traces[i][-1])
        f4t_dxs[i] = traces[i][-1][0] - traces[i][0][0]
        f4t_dys[i] = traces[i][-1][1] - traces[i][0][1]
        f4t_ax[i] = avg_x(traces[i])
        f4t_ay[i] = avg_y(traces[i])
    feat_arr.extend(f4t_as)
    feat_arr.extend(f4t_dxs)
    feat_arr.extend(f4t_dys)
    feat_arr.extend(f4t_ax)
    feat_arr.extend(f4t_ay)

    scale = lambda p,sv: tuple((p[0]/sv, p[1]/sv))

    # draw image from traces at different scales
    img_sizes = [5,10]
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

        if sz < MED_IMG_SZ:
            feat_arr.extend(np.array(im).flatten().tolist())

        # rotated projections
        im = im.rotate(45, resample=Image.BICUBIC)
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
    d_angles = [0]*n_pts
    dd_angles = [0]*n_pts
    dists = [0]*n_pts
    dxs = [0]*n_pts
    dys = [0]*n_pts
    ddxs = [0]*n_pts
    ddys = [0]*n_pts
    c_dists = [0]*n_pts
    daa_pts = [None]*n_pts
    daa_pts[0] = (0,0)
    for i in range(1, n_pts):
        # absolute angles
        a_angles[i] = angle(pts[i], pts[i-1])

        # relative angles
        r_angles[i] = a_angles[i] - a_angles[i-1]

        # change in relative angle
        d_angles[i] = r_angles[i] - r_angles[i-1]
        dd_angles[i] = d_angles[i] - d_angles[i-1]

        # distance
        dists[i] = e_dist(pts[i], pts[i-1])

        # cumulative distance
        c_dists[i] = dists[i] + c_dists[i-1]

        # (IMG_SIZE * scaled distance, abs angle) pts
        daa_pts[i] = (IMG_SIZE*c_dists[i]/(total_sym_len+1), \
                a_angles[i])

        # dx and dy
        dxs[i] = pts[i][0] - pts[i-1][0]
        dys[i] = pts[i][1] - pts[i-1][1]

        # ddx and ddy
        ddxs[i] = dxs[i] - dxs[i-1]
        ddys[i] = dys[i] - dys[i-1]


    # average distance between points
    avg_dist = sum(dists)/len(dists)
    feat_arr.append(avg_dist)

    # bin relative and absolute angles and daa pts
    bin_sizes = [5,10]
    for n_bins in bin_sizes:
        feat_arr.extend(bin_angles( \
                r_angles, n_bins))
        feat_arr.extend(bin_angles( \
                a_angles, n_bins))
        feat_arr.extend(bin_angles( \
                d_angles, n_bins))
        feat_arr.extend(bin_angles( \
                dd_angles, n_bins))

        # % acute angles
        feat_arr.append( \
                sum([int(abs(a) > 90) for a in r_angles])/(len(r_angles)+1))

        # bin daa pts
        get_bin_ind = lambda d,n: int(d*n/IMG_SIZE)
        daa_bins = [0]*n_bins
        for p in daa_pts:
            # index of bin
            ind = get_bin_ind(p[0], n_bins)
            daa_bins[ind] += p[1]

        daa_bins = [ n%360 for n in daa_bins]
        feat_arr.extend(daa_bins)

        # bin dx dy
        dx_bins = [0]*n_bins
        dy_bins = [0]*n_bins
        get_bin_ind = lambda i: int(i*n_bins/n_pts)
        for i in range(n_pts):
            # index of bin
            ind = get_bin_ind(i)
            dx_bins[ind] += dxs[i]
            dy_bins[ind] += dys[i]

        dx_sum = sum(dx_bins)
        dx_sum = dx_sum if dx_sum != 0 else 1
        dy_sum = sum(dy_bins)
        dy_sum = dy_sum if dy_sum != 0 else 1
        dx_bins = [ n/dx_sum for n in dx_bins]
        dy_bins = [ n/dy_sum for n in dy_bins]
        feat_arr.append(dx_sum)
        feat_arr.append(dy_sum)
        feat_arr.extend(dx_bins)
        feat_arr.extend(dy_bins)

        # bin ddx ddy
        ddx_bins = [0]*n_bins
        ddy_bins = [0]*n_bins
        get_bin_ind = lambda i: int(i*n_bins/n_pts)
        for i in range(n_pts):
            # index of bin
            ind = get_bin_ind(i)
            ddx_bins[ind] += ddxs[i]
            ddy_bins[ind] += ddys[i]

        ddx_sum = sum(ddx_bins)
        ddx_sum = ddx_sum if ddx_sum != 0 else 1
        ddy_sum = sum(ddy_bins)
        ddy_sum = ddy_sum if ddy_sum != 0 else 1
        ddx_bins = [ n/ddx_sum for n in ddx_bins]
        ddy_bins = [ n/ddy_sum for n in ddy_bins]
        feat_arr.append(ddx_sum)
        feat_arr.append(ddy_sum)
        feat_arr.extend(ddx_bins)
        feat_arr.extend(ddy_bins)

    return feat_arr


def process_traces(traces):
    # collect points from all traces
    pts = []
    for t in traces:
        pts.extend(t)

    # calculate symbol width and height in original space
    df_pts = pd.DataFrame(np.array(pts))
    max_x,max_y = df_pts[[0,1]].apply(max)
    min_x,min_y = df_pts[[0,1]].apply(min)
    width = max_x - min_x
    height = max_y - min_y

    #'''
    # compute the total length of the strokes in the symbol
    total_sym_len = 0
    for i in range(1,len(pts)):
        total_sym_len += e_dist(pts[i], pts[i-1])
    #'''

    # scaling function
    scale, ar = calc_scale_fn(pts)

    # scale each point
    for i in range(len(pts)):
        pts[i] = scale(pts[i])

    s_traces = []
    total_pts = 0
    for t in traces:
        t_len = len(t)
        # scaled points for this trace
        s_traces.append( \
                pts[total_pts:(total_pts+t_len)])
        total_pts += t_len

    def midpoint(a,b):
        return ( \
            (a[0]+b[0])/2, \
            (a[1]+b[1])/2)

    def midpoint3(a,b,c):
        return( \
            (a[0]+b[0]+c[0])/3, \
            (a[1]+b[1]+c[1])/3)


    feat_arr = extract_features(s_traces)

    # convert to midpoints of 3
    for t in s_traces:
        if len(t) > 3:
            for i in range(1,len(t)-1):
                t[i] = midpoint3(t[i-1],t[i],t[i+1])

    feat_arr3 = extract_features(s_traces)

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

    for i in range(len(feat_arr)):
        feat_arr[i] += feat_arr3[i]
        feat_arr[i] /= 2

    feat_arr.extend((total_sym_len, ar, len(traces), width, height))
    return feat_arr

