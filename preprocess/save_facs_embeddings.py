import numpy as np
import os
import json
import re
import h5py
import librosa
from audio_preprocessing import melspectrogram
import math
import string
import re
import sys
import torch
import time
import torch
import tqdm
import io
import pickle
import multiprocessing as mp


def save_facs_files(facs_embedding, name, dst_path, subset):
    try:
        if os.path.exists(os.path.join(dst_path, subset + '.facs_embedding.h5')):
            with h5py.File(os.path.join(dst_path, subset + '.facs_embedding.h5'), "a") as f:
                f.create_dataset(name, data=facs_embedding, dtype=facs_embedding.dtype)
        else:
            with h5py.File(os.path.join(dst_path, subset + '.facs_embedding.h5'), "w") as f:
                f.create_dataset(name, data=facs_embedding, dtype=facs_embedding.dtype)

        return True
    except:
        print("Embedding extraction failed: ", name)
        return False


files_path = sys.argv[1]  # source path (should contain ['train.files', 'train.text', 'test.files', ...] 
facs_dir = sys.argv[2]  # directory containing ['train', 'test', ...]
vidid_to_name_dict_dir = sys.argv[3]  # directory containing ['train.vidid_to_name.pkl', 'test.vidid_to_name.pkl', ...]
dst_path = sys.argv[4]  # destination path

facs_file_ext = '.npy'

if not os.path.exists(dst_path):
    print("Directory "+dst_path + " created!")
    os.mkdir(dst_path)

for subset in ['train', 'dev', 'test']:
    print(f"Processing {subset}")
    with open(os.path.join(vidid_to_name_dict_dir, f"{subset}.vidid_to_name.pkl"), 'rb') as f:
        vidid_to_name_dict = pickle.load(f)

    facs_subset_dir = os.path.join(facs_dir, subset)
    vidids_fp = os.path.join(files_path, f"{subset}.files")

    processed_vidids = set()
    with io.open(vidids_fp, mode='r', encoding='utf-8') as files_file:
        for vidid in tqdm.tqdm(files_file):
            vidid = vidid.strip()

            if vidid in processed_vidids:
                continue
            
            if vidid not in vidid_to_name_dict:
                print(f"Vidid {vidid} not found in {subset}.vidid_to_name.pkl")
                continue

            vid_dirname, filename = vidid_to_name_dict[vidid]
            facs_fp = os.path.join(facs_subset_dir, vid_dirname, filename + facs_file_ext)

            facs_embedding = np.load(facs_fp).astype(np.float16)

            if save_facs_files(facs_embedding, vidid, dst_path, subset):
                processed_vidids.add(vidid)
            else:
                print("Failed to save: ", vidid)
        
    print(f"Saved {len(processed_vidids)} out of {len(vidid_to_name_dict)} files for {subset} in {dst_path}")