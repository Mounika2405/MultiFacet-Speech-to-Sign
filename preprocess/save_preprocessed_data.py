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
import pickle
import multiprocessing as mp

EXTRACT_TEXT_EMBEDDINGS = False

if EXTRACT_TEXT_EMBEDDINGS:
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

def save_embeddings(text, name):
    indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.zeros(len(indexed_tokens))
    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        raise NotImplementedError()

'''
This will read the Openpose keypoints, audio and text for every video and save it in the destination folder separately
'''

def get_array_with_counter(kp):
    '''
        input shape = x,255 for 3d coordinates
        add counter value at last of every frame keypoints (x,256) and flatten
    '''
    kp = np.array(kp)
    kpts = np.zeros((kp.shape[0], kp.shape[1] + 1))
    full_len = kpts.shape[0]

    for i in range(kp.shape[0]):
    	kpts[i][-1] = i/full_len
    # print(kpts[:][1:].shape, kp.shape)
    for ind,i in enumerate(kp):
    	kpts[ind][:-1] = i
    kpts = kpts.reshape((1, - 1))
    return kpts

def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    
def save_files(dst_path, subset, kpts, specgram, text, name):
    
    ## save skels
    if os.path.exists(os.path.join(dst_path, subset + '.norm_skels')):
        with open(os.path.join(dst_path, subset + '.norm_skels'), "a") as f:
            np.savetxt(f, kpts)
    else:
        np.savetxt(os.path.join(dst_path, subset + ".norm_skels"), kpts)

    # if os.path.exists(os.path.join(dst_path, subset + '.audio.h5')):
    #     with h5py.File(os.path.join(dst_path, subset + '.audio.h5'), "a") as f:
    #         f.create_dataset(name, data=specgram, dtype=specgram.dtype)
    # else:
    #     with h5py.File(os.path.join(dst_path, subset + '.audio.h5'), "w") as f:
    #         f.create_dataset(name, data=specgram, dtype=specgram.dtype)

    # ## save file names
    # if os.path.exists(os.path.join(dst_path, subset + '.files')):
    #     with open(os.path.join(dst_path, subset + '.files'), "a") as f:
    #         f.write('\n' + name)
    # else:
    #     with open(os.path.join(dst_path, subset + '.files'), "w") as f:
    #         f.write(name)

    # ## save text translation
    # if os.path.exists(os.path.join(dst_path, subset + '.text')):
    #     with open(os.path.join(dst_path, subset + '.text'), "a") as f:
    #         f.write('\n' + text)
    # else:
    #     with open(os.path.join(dst_path, subset + '.text'), "w") as f:
    #         f.write(text)
            
def save_tars(dst_path, subset, kpts, specgram, text, name):
    
    data_item = {}
    data_item['src'] = specgram
    data_item['text'] = text
    data_item['trg'] = kpts
    
    save_path = os.path.join(dst_path, f'{subset}_data_text.pth.tar')
    print('save_path', save_path)

    if os.path.exists(save_path) and os.stat(save_path).st_size > 0:
        print(f"Attempting to load file: {save_path}")
        try:
            data = torch.load(save_path)
            print(f"Successfully loaded file: {save_path}")
        except Exception as e:
            print(f"Error loading file: {save_path}. Error: {e}")
            data = {}
    else:
        print(f"File does not exist or is empty: {save_path}")
        data = {}

    data[name] = data_item

    try:
        print(f"Attempting to save file: {save_path}")
        torch.save(data, save_path)
        print(f"Successfully saved file: {save_path}")
    except Exception as e:
        print(f"Error saving file: {save_path}. Error: {e}")

def normalize_3D_skeleton_landmarks(landmarks_3D):
    
    # Indices for left and right shoulder in the keypoints array
    idx_left_shoulder, idx_right_shoulder = 1,0 #12, 11
    
    # Calculate midpoint between left and right shoulder
    mid_point = landmarks_3D[:, [idx_left_shoulder, idx_right_shoulder], :].mean(axis=1, keepdims=True)
    
    # Calculate the distance between the left and right shoulder, which is used for normalization
    shoulder_distance = np.linalg.norm(landmarks_3D[:, idx_left_shoulder, :] - landmarks_3D[:, idx_right_shoulder, :], ord=2, axis=1)
    # print('shoulder_distance', shoulder_distance)
    
    # Remove frames from beginning where shoulder-length is zero i.e. person not detected
    MIN_VALID_SHOULDER_LEN = 1e-2
    valid_indices_bool_flt  = np.cumsum(shoulder_distance > MIN_VALID_SHOULDER_LEN) > 0
    
    landmarks_3D = landmarks_3D[valid_indices_bool_flt]
    mid_point = mid_point[valid_indices_bool_flt].mean(axis=0, keepdims=True)
    mean_shoulder_length = np.mean(shoulder_distance[shoulder_distance > MIN_VALID_SHOULDER_LEN])
    
    # Normalize the 3D keypoints based on the midpoint and the mean shoulder distance
    normalized_3D_landmarks = (landmarks_3D - mid_point) / mean_shoulder_length #shoulder_distance[:, None, None]
    
    return normalized_3D_landmarks.reshape(normalized_3D_landmarks.shape[0], -1)


path = sys.argv[1]  # source path (should contain ['train_cut', 'test_cut', 'dev_cut'] with each having ['audio', 'text', 'OP'])
dst_path = sys.argv[2]  # destination path

if not os.path.exists(dst_path):
    print("Directory "+dst_path + " created!")
    os.mkdir(dst_path)
    
src_vids = []
len_subsets = []

for subset in ['dev', 'test', 'train']:
    subset_vids = os.listdir(os.path.join(path, f'{subset}'))
    len_subsets.append(len(subset_vids))
    src_vids.extend([f'{subset}/{vid}' for vid in subset_vids if os.path.isdir(os.path.join(path, f'{subset}', vid))])
        

for i, vid in enumerate(tqdm.tqdm(sorted(src_vids))):   
# def process_video(vid): 
    subset = vid.split('/')[0]
    try:
        variations = [n.split('.')[0] for n in sorted(os.listdir(os.path.join(path, vid, 'text')))]
    except Exception as e:
        print(e)
    print('Processing vid...', vid)
    for var in variations:
        
        audio_path = os.path.join(path, vid, 'audio', var + '.wav')
        text_path = os.path.join(path, vid, 'text', var + '.txt')
        # pose_path = os.path.join(path, vid, 'OP',  f'keypoints-{var}-filter.h5')
        pose_path = os.path.join(path, vid, 'OP', var + '.pkl')
        # print(audio_path, text_path, pose_path)
        if not os.path.exists(pose_path):
            continue
        
        try:
            with open(text_path, 'r') as f:
                text = f.readlines()[0]
            
            text= re.sub(r'[^\w\s]', '', text)
            text = text.lower()
            
            if(text[0]==' '):
                text=text[1:]
            
            if(text[-1]==' '):
                text=text[:-1]

            if text[-1]!='.':
                text += ' .'

            
            waveform, sr = librosa.load(audio_path, sr = 16000)
            waveform, index = librosa.effects.trim(waveform)  ## to remove starting and trailing silences
            mel = melspectrogram(waveform) # numpy array of shape (80, T)

            pose_path = pose_path.encode('unicode_escape').decode().replace('\\u','#U')
        # try:
            # kpts = h5py.File(pose_path, 'r')[var]
            with open(pose_path, 'rb') as f:
                kpts = pickle.load(f)
        except Exception as e:
            # print("missing/faulty keypoints ", pose_path)
            print('Exception', e)
            continue 
        

        kpts = normalize_3D_skeleton_landmarks(kpts.reshape(-1, 85,3))
        kpts = get_array_with_counter(kpts)
        
        name = vid+'_'+var
        
        # print('Saving...')
        # save_tars(dst_path, subset, kpts, mel, text, name)
        save_files(dst_path, subset, kpts, mel, text, name)
        # save_embeddings(text, name)