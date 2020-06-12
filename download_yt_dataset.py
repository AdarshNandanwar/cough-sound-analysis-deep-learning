# -*- coding: utf-8 -*-

from pytube import YouTube
import soundfile as sf
import librosa
import json
import os
import requests

SAMPLE_RATE = 22050
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
link_filename_map = {}
curr_file_number = 0
curr_link_number = 0
try:
    with open(os.path.join(BASE_DIR, 'link_filename_map.json'), 'r') as fp:
        link_filename_map = json.load(fp)
    curr_link_number = len(link_filename_map)
except Exception as e:
    pass

def download_youtube_audio(label, link, start_time=0.0, duration = 5.0):
    global link_filename_map
    global curr_link_number
    global curr_file_number
    download_path=os.path.join(BASE_DIR, 'yt_downloads')

    try:
        # download file if not downloaded
        filename = link_filename_map.get(link)
        if filename == None:
            filename = str(curr_link_number)
            link_filename_map[link] = filename
            curr_link_number += 1

        # download audio from youtube
        yt = YouTube(link)
        yt_audio = yt.streams.get_audio_only()
        yt_audio.download(output_path=download_path, filename=filename)

        # crop and save audio file
        file_path = os.path.join(BASE_DIR, download_path, filename+'.mp4')
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, offset=start_time, duration=duration)
        cur_dir = os.path.join(BASE_DIR, 'yt_dataset', label)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        sf.write(os.path.join(cur_dir, str(curr_file_number)+'.wav'), signal, SAMPLE_RATE)
        curr_file_number += 1

    except Exception as e:
        print(e)


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
    # clear_yt_downloads()

    labels_count = 4
    labels = ['dry', 'wet', 'whooping', 'croup']

    for i in range(labels_count):
        google_sheets_data = requests.get('https://spreadsheets.google.com/feeds/list/18BCme4ZxUIGwpzmzTnCSsAxV3kOm_OH8yAHJlDRDDMQ/'+str(i+1)+'/public/full?alt=json')
        google_sheets_data = google_sheets_data.json()['feed'].get('entry')
        if google_sheets_data:
            for entry in google_sheets_data:
                download_youtube_audio(label=labels[i], link=entry['gsx$link']['$t'], start_time=float(entry['gsx$starttime']['$t']), duration=float(entry['gsx$duration']['$t']))

    with open(os.path.join(BASE_DIR, 'link_filename_map.json'), 'w') as fp:
        json.dump(link_filename_map, fp)