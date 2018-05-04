#!/usr/bin/env python3

def do_with_progress_bar(data, func, data_to_str=str):
    n_to_process = len(data)
    n_done = 0
    for d in data:
        func(d)
        n_done += 1
        p_done = n_done/n_to_process
        p20 = round(p_done*20)
        print("\r[{0}] {2}/{3} ({1}%) {4}".format( \
                '#'*p20+' '*(20-p20), round(p_done*100), \
                n_done, n_to_process, data_to_str(d)), \
                end='')
    print()


if __name__=="__main__":
    l = []
    print("Computing x^3 for x in [0,100000]")
    do_with_progress_bar(range(100000), \
            lambda x: l.append(x**3))
    print("First 10 results:",l[:10])

