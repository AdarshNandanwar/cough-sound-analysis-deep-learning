# -*- coding: utf-8 -*-

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clear_yt_dataset():
    labels = ['dry', 'wet', 'whooping', 'croup']
    for label in labels:
        mydir = os.path.join(BASE_DIR, 'yt_dataset', label)
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        filelist = [ f for f in os.listdir(mydir) if f.endswith(".wav") ]
        for f in filelist:
            os.remove(os.path.join(mydir, f))

def clear_yt_downloads():
    mydir = os.path.join(BASE_DIR, 'yt_downloads')
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".mp4") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

    # clear JSON map file
    link_filename_map = {}
    with open(os.path.join(BASE_DIR, 'link_filename_map.json'), 'w') as fp:
        json.dump(link_filename_map, fp)

if __name__ == "__main__":

    clear_yt_dataset()
    clear_yt_downloads()