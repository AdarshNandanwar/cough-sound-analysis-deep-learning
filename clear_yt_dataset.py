# -*- coding: utf-8 -*-

import json
import os

def clear_yt_dataset():
    labels = ['dry', 'wet', 'whooping', 'croup']
    for label in labels:
        mydir = os.path.join('yt_dataset', label)
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".wav") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f))

def clear_yt_downloads():
    mydir = 'yt_downloads'
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".mp4") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

    # clear JSON map file
    link_filename_map = {}
    with open('link_filename_map.json', 'w') as fp:
        json.dump(link_filename_map, fp)

if __name__ == "__main__":

    clear_yt_dataset()
    clear_yt_downloads()