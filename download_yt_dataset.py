# -*- coding: utf-8 -*-

from pytube import YouTube
import soundfile as sf
import librosa
import numpy as np
import json
import os
import requests
from clear_yt_dataset import clear_yt_dataset, clear_yt_downloads

SAMPLE_RATE = 22050
DATASET_AUDIO_DURATION = 5
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
        cur_dir = os.path.join(BASE_DIR, 'yt_dataset', label)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        if start_time is None:
            min_duration = 1.5
            max_duration = DATASET_AUDIO_DURATION
            threshold = 67
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
            cough_intevals = librosa.effects.split(signal, top_db=threshold)
            for interval in cough_intevals:
                int_st = interval[0]
                int_en = interval[1]
                step = sample_rate*max_duration
                for i in range(int_st, int_en, step):
                    if int_en-i >= sample_rate*min_duration:
                        fixed_signal = signal[i:min(i+step, int_en)]
                        if int_en-i < step:
                            fixed_signal = librosa.util.fix_length(fixed_signal, step)
                        sf.write(os.path.join(cur_dir, str(curr_file_number)+'.wav'), fixed_signal, SAMPLE_RATE)
                        curr_file_number += 1

        else:
            signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, offset=start_time, duration=duration)
            int_st = np.int64(0)
            int_en = np.int64(duration*sample_rate)
            min_duration = 1.5
            max_duration = DATASET_AUDIO_DURATION
            step = sample_rate*max_duration
            for i in range(int_st, int_en, step):
                if int_en-i >= sample_rate*min_duration:
                    fixed_signal = signal[i:min(i+step, int_en)]
                    if int_en-i < step:
                        fixed_signal = librosa.util.fix_length(fixed_signal, step)
                    sf.write(os.path.join(cur_dir, str(curr_file_number)+'.wav'), fixed_signal, SAMPLE_RATE)
                    curr_file_number += 1

    except Exception as e:
        print(e)


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
                try:
                    start_time = float(entry['gsx$starttime']['$t'])
                    duration = float(entry['gsx$duration']['$t'])
                except Exception as e:
                    start_time = None
                    duration = None
                
                print(labels[i], entry['gsx$link']['$t'], start_time, duration)
                download_youtube_audio(label=labels[i], link=entry['gsx$link']['$t'], start_time=start_time, duration=duration)

    with open(os.path.join(BASE_DIR, 'link_filename_map.json'), 'w') as fp:
        json.dump(link_filename_map, fp)