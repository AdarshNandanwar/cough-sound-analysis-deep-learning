import soundfile as sf
import numpy as np
import librosa
import os
import shutil

SAMPLE_RATE = 22050
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRTY_DATASET_PATHS = [os.path.join(BASE_DIR, "detection_dataset"), os.path.join(BASE_DIR, "classification_dataset")]
DATASET_AUDIO_DURATION = 5
curr_file_number = 0

def clean_datasets(dirty_dataset_paths, min_duration=1.5, max_duration=5.0):
    """Creates a clean dataset from the existing datasets by cropping and extending the audio files to the max_duration
        """

    global curr_file_number

    # loop through all the dataset paths
    for dataset_path in dirty_dataset_paths:
        dataset_name = dataset_path.split("/")[-1]
        clean_dataset_path = os.path.join(BASE_DIR, 'clean_'+dataset_name)
        print("Creating {}...".format(clean_dataset_path))
        if os.path.exists(clean_dataset_path):
            shutil.rmtree(clean_dataset_path)
        # loop through all sub-folder
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            # ensure we're processing a sub-folder level
            if dirpath is not dataset_path:
                # save label  mapping
                semantic_label = dirpath.split("/")[-1]
                clean_dataset_dir_path = os.path.join(clean_dataset_path, semantic_label)
                if not os.path.exists(clean_dataset_dir_path):
                    os.makedirs(clean_dataset_dir_path)
                # process all audio files in sub-dir
                for f in filenames:
                    # audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                    threshold = 67
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
                                if not os.path.exists(clean_dataset_dir_path):
                                    os.makedirs(clean_dataset_dir_path)
                                sf.write(os.path.join(clean_dataset_dir_path, str(curr_file_number)+'.wav'), fixed_signal, SAMPLE_RATE)
                                curr_file_number += 1
        print("Done!")
                    
if __name__ == "__main__":

    clean_datasets(DIRTY_DATASET_PATHS, max_duration=DATASET_AUDIO_DURATION)