# -*- coding: utf-8 -*-

import json
import os
import math
import librosa

DATASET_PATH = "dataset/train"
JSON_PATH = "processed_data.json"
SAMPLE_RATE = 22050


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from dataset and saves them into a json file along with labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfccs": [],
        "spectral_centeroids": [],
        "spectral_rolloffs": [],
        "spectral_bandwidth_2": [],
        "spectral_bandwidth_3": [],
        "spectral_bandwidth_4": [],
        "zero_crossing_rates": [],
        "chroma_features": [],
    }

    # loop through all sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            log_file.write("\nProcessing: {}\n".format(semantic_label))

            # process all audio files in sub-dir
            for f in filenames:

                # audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                log_file.write("\nSignal shape: {}, Sample rate: {}".format(signal.shape, sample_rate))
                
                track_duration = librosa.get_duration(y=signal, sr=sample_rate)
                samples_per_track = sample_rate * track_duration
                log_file.write("\nTrack Duration: {}s, Samples per track: {}".format(track_duration, samples_per_track))
                
                samples_per_segment = int(samples_per_track / num_segments)
                num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
                log_file.write("\nSamples per segment: {}, No. of mfcc vectors per segment: {}".format(samples_per_segment, num_mfcc_vectors_per_segment))

                # process all segments of file
                for d in range(num_segments):
                    log_file.write("\n{}, segment:{}".format(file_path, d+1))

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment




                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    log_file.write("\nShape of mfcc: {}".format(mfcc.shape))
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfccs"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        
                        
                    ##################################
                    # ADD IF CHECKS ON EVERY FEATURE #  
                    ##################################
                    
                    # extract spectral centeroid
                    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0]
                    log_file.write("\nShape of spectral centeroids: {}".format(spectral_centroids.shape))
                    data["spectral_centeroids"].append(spectral_centroids.tolist())
                    
                    
                    # extract spectral rolloff
                    spectral_rolloffs = librosa.feature.spectral_rolloff(signal+0.01, sr=sample_rate)[0]
                    log_file.write("\nShape of spectral rolloffs: {}".format(spectral_rolloffs.shape))
                    data["spectral_rolloffs"].append(spectral_rolloffs.tolist())
                    
                    # extract spectral bandwidth
                    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate)[0]
                    log_file.write("\nShape of spectral bandwidth 2: {}".format(spectral_bandwidth_2.shape))
                    data["spectral_bandwidth_2"].append(spectral_bandwidth_2.tolist())
                    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=3)[0]
                    log_file.write("\nShape of spectral bandwidth 3: {}".format(spectral_bandwidth_3.shape))
                    data["spectral_bandwidth_3"].append(spectral_bandwidth_3.tolist())
                    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=4)[0]
                    log_file.write("\nShape of spectral bandwidth 4: {}".format(spectral_bandwidth_4.shape))
                    data["spectral_bandwidth_4"].append(spectral_bandwidth_4.tolist())
                    
                    # extract zero-crossing rate                    
                    zero_crossing_rates = librosa.feature.zero_crossing_rate(signal, pad=False)[0]
                    log_file.write("\nShape of zero crossing rates: {}".format(zero_crossing_rates.shape))
                    data["zero_crossing_rates"].append(zero_crossing_rates.tolist())
                    
                    # extract croma features
                    chroma_features = librosa.feature.chroma_stft(signal, sr=sample_rate, hop_length=hop_length)
                    chroma_features = chroma_features.T
                    log_file.write("\nShape of chroma features: {}".format(chroma_features.shape))
                    data["chroma_features"].append(chroma_features.tolist())
                    
                     
                     
                     
                
                log_file.write("\n")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    log_file = open("log.txt", 'w')
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1)
    log_file.close()