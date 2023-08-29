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

EXTRACT_TEXT_EMBEDDINGS = True
MAX_TEXT_LENGTH = 40

if EXTRACT_TEXT_EMBEDDINGS:
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

def save_embeddings(text, name, dst_path, subset):
    text_tokens = tokenizer.tokenize(text)
    num_pad_tokens = max(0, MAX_TEXT_LENGTH - len(text_tokens) - 1)
    text_tokens = ['[CLS]'] + text_tokens[:MAX_TEXT_LENGTH-1] + (['[PAD]'] * num_pad_tokens)
    text_mask = [1] * (len(text_tokens) - num_pad_tokens) + ([0] * num_pad_tokens)

    indexed_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        encoded_layers = bert_model(tokens_tensor)
        text_embedding = encoded_layers.last_hidden_state.squeeze().cpu().numpy().astype(np.float16)
        
    try:
        if os.path.exists(os.path.join(dst_path, subset + '.text_embedding.h5')):
            with h5py.File(os.path.join(dst_path, subset + '.text_embedding.h5'), "a") as f:
                f.create_dataset(name, data=text_embedding, dtype=text_embedding.dtype)
            
            with h5py.File(os.path.join(dst_path, subset + '.text_embedding_mask.h5'), "a") as f:
                f.create_dataset(name, data=text_mask, dtype=bool)
        else:
            with h5py.File(os.path.join(dst_path, subset + '.text_embedding.h5'), "w") as f:
                f.create_dataset(name, data=text_embedding, dtype=text_embedding.dtype)

            with h5py.File(os.path.join(dst_path, subset + '.text_embedding_mask.h5'), "w") as f:
                f.create_dataset(name, data=text_mask, dtype=bool)
                

        return True
    except Exception as e:
        print("Embedding extraction failed: ", name)
        print(e)
        return False

path = sys.argv[1]  # source path (should contain ['train.files', 'train.text', 'test.files', ...] 
dst_path = sys.argv[2]  # destination path

if not os.path.exists(dst_path):
    print("Directory "+dst_path + " created!")
    os.mkdir(dst_path)


for subset in ['train', 'dev', 'test']:
    text_fp = os.path.join(path, f"{subset}.text")
    vidids_fp = os.path.join(path, f"{subset}.files")
    
    out_fp = os.path.join(dst_path, f"{subset}.text_embedding.h5")
    if os.path.exists(out_fp):
        with h5py.File(os.path.join(dst_path, subset + '.text_embedding.h5'), "r") as f:
            processed_vidids = set([f'{subset}/{k}' for k in f[subset].keys()])
    else:
        processed_vidids = set()

    print(f'Existing in {subset}:', len(processed_vidids))

    with io.open(text_fp, mode='r', encoding='utf-8') as nonreg_trg_file, \
        io.open(vidids_fp, mode='r', encoding='utf-8') as files_file:
        for text, vidid in tqdm.tqdm(zip(nonreg_trg_file, files_file)):
            text, vidid = text.strip(), vidid.strip()

            if vidid in processed_vidids:
                continue

            text= re.sub(r'[^\w\s]', '', text)
            text = text.lower()

            if len(text) < 1:
                print("Skipping: ", vidid, text)
                continue

            if(text[0]==' '):
                text=text[1:]
            
            if(text[-1]==' '):
                text=text[:-1]

            if text[-1]!='.':
                text += ' .'

            if save_embeddings(text, vidid, dst_path, subset):
                processed_vidids.add(vidid)

    print(f'Total in {subset}', len(processed_vidids))