import librosa
import torch
import numpy as np
from scipy.io.wavfile import write
from model import Tacotron2, load_model
from hparams import create_hparams
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
import h5py
import os
from torch.autograd import Variable

def load_mel(path, hparams, stft):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError(f"{sampling_rate} SR doesn't match target {stft.sampling_rate} SR")
    audio_norm = torch.autograd.Variable(torch.from_numpy(audio).unsqueeze(0), requires_grad=False)
    mel = stft.mel_spectrogram(audio_norm).cuda()
    
    # clear GPU memory after use
    audio_norm = None
    torch.cuda.empty_cache()
    return mel

def prepare_data(hparams, stft):
    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
    dataloader = TextMelLoader('mellotron_filelist_test.txt', hparams)
    idx = 0
    text_encoded, mel, pitch_contour, sids, paths = [], [], [], [], []
    for audio_path, text, speaker_id in dataloader.audiopaths_and_text:
        text_encoded.append(torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda())
        mel.append(load_mel(audio_path, hparams, stft))
        pitch_contour.append(dataloader[idx][3][None].cuda())
        sids.append(speaker_id)
        idx += 1
        paths.append(audio_path)

    return text_encoded, mel, pitch_contour, sids, paths

def save(text_embeddings, gst_embeddings):
    def path_to_key(path):
        subset_dir = os.path.basename(os.path.dirname(path)) # should be train, dev or test
        assert subset_dir in ['train', 'dev', 'test']
        filename = os.path.splitext(os.path.basename(path))[0]
        key = os.path.join(subset_dir, filename)
        # key = subset_dir + "_" + filename
        return key
    
    text_outfp = 'mellotron_text_test.h5'
    mode = 'a' if os.path.exists(text_outfp) else 'w'

    with h5py.File(text_outfp, mode) as hf:
        for path, embed in text_embeddings.items():
            embed = embed.squeeze()
            key = path_to_key(path)
            hf.create_dataset(name=key, data=embed, dtype=embed.dtype)
        
    gst_outfp = 'mellotron_gst_test.h5'
    mode = 'a' if os.path.exists(gst_outfp) else 'w'
    with h5py.File(gst_outfp, mode) as hf:
        for path, embed in gst_embeddings.items():
            embed = embed.squeeze().transpose(0,1)
            key = path_to_key(path)
            hf.create_dataset(name=key, data=embed, dtype=embed.dtype)

def extract_prosody(hparams, stft, mellotron, text_encoded, mel, pitch_contour, sids, paths):
    frequency_scaling = 0.4 # check
    text_embeddings = {}
    gst_embeddings = {}
    for te, m, pc, sid, path in zip(text_encoded, mel, pitch_contour, sids, paths):
        sid = torch.LongTensor([int(sid)]).cuda()
        with torch.no_grad():
            text_embed, gst_embed = mellotron.inference((te, m, sid, pc * frequency_scaling))
        text_embeddings[path] = text_embed.cpu().numpy().astype(np.float16)
        gst_embeddings[path] = gst_embed.cpu().numpy().astype(np.float16)
        
    save(text_embeddings, gst_embeddings)
    # return text_embed, gst_embed

hparams = create_hparams()
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)
mellotron = load_model(hparams).cuda().eval()
mellotron.load_state_dict(torch.load("mellotron_libritts.pt")['state_dict'])

text_encoded, mel, pitch_contour, sids, paths = prepare_data(hparams, stft)
extract_prosody(hparams, stft, mellotron, text_encoded, mel, pitch_contour, sids, paths)
