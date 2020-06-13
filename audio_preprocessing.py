# -*- coding: utf-8 -*-

import json
import csv
import os
import math
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_DATASET_PATHS = [os.path.join(BASE_DIR, "detection_dataset")]
DETECTION_JSON_PATH = "detection_data.json"
DETECTION_CSV_PATH = "detection_data.csv"
CLASSIFICATION_DATASET_PATHS = [os.path.join(BASE_DIR, "yt_dataset"), os.path.join(BASE_DIR, "clean_classification_dataset")]
CLASSIFICATION_JSON_PATH = "classification_data.json"
CLASSIFICATION_CSV_PATH = "classification_data.csv"
SAMPLE_RATE = 22050


def save_features_in_JSON(dataset_paths, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from dataset and saves them into a json file along with labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save features
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
        "spectral_centroids": [],
        "spectral_rolloffs": [],
        "spectral_bandwidth_2": [],
        "spectral_bandwidth_3": [],
        "spectral_bandwidth_4": [],
        "zero_crossing_rates": [],
        "chroma_features": [],
    }

    # loop through all the dataset paths
    for dataset_path in dataset_paths:
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
                        spectral_centroids = librosa.feature.spectral_centroid(signal[start:finish], sr=sample_rate)[0]
                        log_file.write("\nShape of spectral centroids: {}".format(spectral_centroids.shape))
                        data["spectral_centroids"].append(spectral_centroids.tolist())
                        
                        # extract spectral rolloff
                        spectral_rolloffs = librosa.feature.spectral_rolloff(signal[start:finish]+0.01, sr=sample_rate)[0]
                        log_file.write("\nShape of spectral rolloffs: {}".format(spectral_rolloffs.shape))
                        data["spectral_rolloffs"].append(spectral_rolloffs.tolist())
                        
                        # extract spectral bandwidth
                        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal[start:finish]+0.01, sr=sample_rate)[0]
                        log_file.write("\nShape of spectral bandwidth 2: {}".format(spectral_bandwidth_2.shape))
                        data["spectral_bandwidth_2"].append(spectral_bandwidth_2.tolist())
                        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal[start:finish]+0.01, sr=sample_rate, p=3)[0]
                        log_file.write("\nShape of spectral bandwidth 3: {}".format(spectral_bandwidth_3.shape))
                        data["spectral_bandwidth_3"].append(spectral_bandwidth_3.tolist())
                        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal[start:finish]+0.01, sr=sample_rate, p=4)[0]
                        log_file.write("\nShape of spectral bandwidth 4: {}".format(spectral_bandwidth_4.shape))
                        data["spectral_bandwidth_4"].append(spectral_bandwidth_4.tolist())
                        
                        # extract zero-crossing rate                    
                        zero_crossing_rates = librosa.feature.zero_crossing_rate(signal[start:finish], pad=False)[0]
                        log_file.write("\nShape of zero crossing rates: {}".format(zero_crossing_rates.shape))
                        data["zero_crossing_rates"].append(zero_crossing_rates.tolist())
                        
                        # extract croma features
                        chroma_features = librosa.feature.chroma_stft(signal[start:finish], sr=sample_rate, hop_length=hop_length)
                        chroma_features = chroma_features.T
                        log_file.write("\nShape of chroma features: {}".format(chroma_features.shape))
                        data["chroma_features"].append(chroma_features.tolist())
                        
                    log_file.write("\n")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def get_features_csv_row(signal, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512):

    csv_row = []

    # extract mfcc
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    csv_row += sum(mfcc.tolist(), [])
        
    # extract spectral centeroid
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0]
    csv_row += spectral_centroids.tolist()
    
    # extract spectral rolloff
    spectral_rolloffs = librosa.feature.spectral_rolloff(signal+0.01, sr=sample_rate)[0]
    csv_row += spectral_rolloffs.tolist()
    
    # extract spectral bandwidth
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate)[0]
    csv_row += spectral_bandwidth_2.tolist()
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=3)[0]
    csv_row += spectral_bandwidth_3.tolist()
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=4)[0]
    csv_row += spectral_bandwidth_4.tolist()
    
    # extract zero-crossing rate                    
    zero_crossing_rates = librosa.feature.zero_crossing_rate(signal, pad=False)[0]
    csv_row += zero_crossing_rates.tolist()
    
    # extract croma features
    chroma_features = librosa.feature.chroma_stft(signal, sr=sample_rate, hop_length=hop_length)
    chroma_features = chroma_features.T
    csv_row += sum(chroma_features.tolist(), [])

    return csv_row
        
def save_features_in_CSV(dataset_paths, csv_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from dataset and saves them into a csv file along with labels.

        :param dataset_path (str): Path to dataset
        :param csv_path (str): Path to csv file used to save features
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    with open(csv_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        # loop through all the dataset paths
        for dataset_path in dataset_paths:
            # loop through all sub-folder
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
                # ensure we're processing a sub-folder level
                if dirpath is not dataset_path:
                    # save label  mapping
                    semantic_label = dirpath.split("/")[-1]
                    mapping_file.write("\n{} : {}".format(i-1, semantic_label))
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

                            csv_row = get_features_csv_row(signal[start:finish], sample_rate, num_mfcc, n_fft, hop_length)
                            # label
                            csv_row += [i-1]
                            wr.writerow(csv_row)
                            
                        log_file.write("\n")
        
if __name__ == "__main__":


    mapping_file = open(os.path.join(BASE_DIR, "mapping.txt"), 'w')
    print("Creating detection dataset.")
    mapping_file.write("\nDetection:")
    # log_file = open(os.path.join(BASE_DIR, "detection_log.txt"), 'w')
    # save_features_in_JSON(DETECTION_DATASET_PATHS, DETECTION_JSON_PATH, num_segments=1)
    # log_file.close()
    log_file = open(os.path.join(BASE_DIR, "detection_log.txt"), 'w')
    save_features_in_CSV(DETECTION_DATASET_PATHS, DETECTION_CSV_PATH, num_segments=1)
    log_file.close()
    print("Detection dataset created successuflly.")

    print("Creating classification dataset.")
    mapping_file.write("\nClassification:")
    # log_file = open(os.path.join(BASE_DIR, "classification_log.txt"), 'w')
    # save_features_in_JSON(CLASSIFICATION_DATASET_PATHS, CLASSIFICATION_JSON_PATH, num_segments=1)
    # log_file.close()
    log_file = open(os.path.join(BASE_DIR, "classification_log.txt"), 'w')
    save_features_in_CSV(CLASSIFICATION_DATASET_PATHS, CLASSIFICATION_CSV_PATH, num_segments=1)
    log_file.close()
    print("Classification dataset created successuflly.")
    mapping_file.close()