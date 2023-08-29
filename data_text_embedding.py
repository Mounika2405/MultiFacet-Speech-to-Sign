# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
from typing import Optional
import io
import numpy as np

try:
    from torchtext import data
    from torchtext.data import Dataset, Iterator, Field
except:
    from torchtext.legacy import data
    from torchtext.legacy.data import Dataset, Iterator, Field
    
import torch

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD, FACS_TARGET_PAD, KEYPOINT_SCALE
from vocabulary import build_vocab, Vocabulary

import librosa
import h5py
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
wordnet_lemmatizer = WordNetLemmatizer()

# Load the Regression Data
# Data format should be parallel .txt files for src, trg and files
# Each line of the .txt file represents a new sequence, in the same order in each file
# src file should contain a new source input on each line
# trg file should contain skeleton data, with each line a new sequence, each frame following on from the previous
# Joint values were divided by 4 to move to the scale of -1 to 1
# Each joint value should be separated by a space; " "
# Each frame is partioned using the known trg_size length, which includes all joints (In 2D or 3D) and the counter
# Files file should contain the name of each sequence on a new line


def load_data(cfg: dict, mode='train') -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """

    data_cfg = cfg["data"]
    is_train = mode=='train'
    
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    text_src_lang = data_cfg["text_src"]
    text_src_mask_lang = data_cfg["text_src_mask"]
    trg_lang = data_cfg["trg"]
    nonreg_trg_lang = data_cfg["nonreg_trg"]
    files_lang = data_cfg.get("files", "files")
    facs_lang = data_cfg.get("facs", "facs")
    
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "char" #"word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]

    nonreg_as_text_src = data_cfg.get("nonreg_as_text_src", False)
    
    # Target size is plus one due to the counter required for the model
    src_size = cfg["model"]["src_size"]
    text_src_size = cfg["model"]["text_src_size"] if "text_src_size" in cfg["model"] else 0
    trg_size = cfg["model"]["trg_size"] + 1
    facs_trg_size = cfg["model"]["facs_decoder"].get("trg_size", 0) if "facs_decoder" in cfg["model"] else 0
    num_keypoints = cfg["data"].get("num_keypoints", 85) # Defaults to all keypoints
    
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    num_sec=data_cfg.get("num_sec", 6) 

    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    # Source field is a tokenised version of the source words
    src_field = data.Field(sequential=True,
                           use_vocab=False,
                           dtype=torch.float32,
                           batch_first=True,
                           include_lengths=True,
                           pad_token=torch.ones((src_size,))*TARGET_PAD,)

    # Text Source field is embeddings of the source text words
    if not nonreg_as_text_src:
        text_src_field = data.Field(sequential=True,
                            use_vocab=False,
                            dtype=torch.float32,
                            batch_first=True,
                            include_lengths=False,
                            pad_token=torch.ones((src_size,))*TARGET_PAD,)
    else:
        text_src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)
    
    # Text Source mask field is a mask for the text source field
    text_src_mask_field = data.Field(sequential=True,
                           use_vocab=False,
                           dtype=torch.uint8,  # torch.bool not supported in torchtext
                           batch_first=True,
                           include_lengths=False,
                           pad_token=torch.ones((src_size,))*TARGET_PAD,)
                           

    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,))*TARGET_PAD)
    
    ## For text translation                           
    nonreg_trg_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)
    
    ## For facs translation                           
    facs_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=True,
                               pad_token=torch.ones((facs_trg_size,))*FACS_TARGET_PAD)
    
    # Create the Training Data, using the SignProdDataset
    print('Created train data')
    train_data = SignProdDataset(path=train_path,
                                    exts=("." + src_lang, "." + text_src_lang, "." + text_src_mask_lang, 
                                          "." + trg_lang, "." + nonreg_trg_lang, 
                                          "." + files_lang, "." + facs_lang),
                                    fields=(src_field, text_src_field, text_src_mask_field, 
                                            reg_trg_field, nonreg_trg_field, files_field, facs_trg_field),
                                    trg_size=trg_size,
                                    src_size = src_size,
                                    text_src_size = text_src_size,
                                    facs_trg_size=facs_trg_size,
                                    skip_frames=skip_frames, is_train = is_train,
                                    num_sec = num_sec, num_keypoints=num_keypoints, 
                                    nonreg_as_text_src=nonreg_as_text_src)
                           

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    nonreg_trg_min_freq = data_cfg.get("nonreg_trg_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = [None]*src_size
    
    # Create a target vocab just as big as the required target vector size -
    trg_vocab = [None]*trg_size

    # Create a facs vocab just as big as the required facs vector size -
    facs_trg_vocab = [None]*facs_trg_size
    
    nonreg_trg_vocab = build_vocab(field="nonreg_trg", min_freq=nonreg_trg_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file) 

    if nonreg_as_text_src:
        text_src_vocab = build_vocab(field="text_src", min_freq=1,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=None) 
    else:
        text_src_vocab = None

    print(('Created vocab'))

    # Create the Validation Data
    dev_data = SignProdDataset(path=dev_path,
                               exts=("." + src_lang, "." + text_src_lang, "." + text_src_mask_lang, 
                                     "." + trg_lang, "." + nonreg_trg_lang, 
                                     "." + files_lang, "." + facs_lang),
                                fields=(src_field, text_src_field, text_src_mask_field, 
                                        reg_trg_field, nonreg_trg_field, files_field, facs_trg_field),
                                trg_size=trg_size,
                                src_size = src_size,
                                text_src_size = text_src_size,
                                facs_trg_size=facs_trg_size,
                                skip_frames=skip_frames, is_train = is_train,
                                num_sec=num_sec, num_keypoints=num_keypoints, 
                                nonreg_as_text_src=nonreg_as_text_src)
    

    # Create the Testing Data
    test_data = SignProdDataset(path=test_path,
                               exts=("." + src_lang, "." + text_src_lang, "." + text_src_mask_lang, 
                                     "." + trg_lang, "." + nonreg_trg_lang, 
                                     "." + files_lang, "." + facs_lang),
                                fields=(src_field, text_src_field, text_src_mask_field, 
                                        reg_trg_field, nonreg_trg_field, files_field, facs_trg_field),
                                trg_size=trg_size,
                                src_size = src_size,
                                text_src_size = text_src_size,
                                facs_trg_size=facs_trg_size,
                                skip_frames=skip_frames, is_train = is_train,
                                num_sec=num_sec, num_keypoints=num_keypoints, 
                                nonreg_as_text_src=nonreg_as_text_src)
    
    src_field.vocab = src_vocab
    nonreg_trg_field.vocab = nonreg_trg_vocab
    facs_trg_field.vocab = facs_trg_vocab
    text_src_field.vocab = text_src_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab, nonreg_trg_vocab, facs_trg_vocab, text_src_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter

# Main Dataset Class
class SignProdDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, exts, fields, trg_size, src_size, text_src_size, facs_trg_size, num_sec, skip_frames=1,
                  is_train=True, num_keypoints=85, nonreg_as_text_src=False, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('text_src', fields[1]), ('text_src_mask', fields[2]), 
                      ('trg', fields[3]),('nonreg_trg', fields[4]), 
                      ('file_paths', fields[5]), ('facs_trg', fields[6])]

        src_path, text_src_path, text_src_mask_path, trg_path, nonreg_trg_path, file_path, facs_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []

        src_fps, trg_fps = 100, 25
        split = src_path.split('/')[-1].split('.')[0]

        if not is_train and split != 'test':
            super(SignProdDataset, self).__init__(examples, fields, **kwargs)

        LOAD_TEXT_SRC = True if (text_src_size > 0 and not nonreg_as_text_src) else False
        LOAD_FACS = True if facs_trg_size > 0 else False
        LOAD_NONREG_TRG_AS_TEXT_SRC = nonreg_as_text_src

        with h5py.File(src_path, 'r') as src_file, \
            h5py.File(text_src_path, 'r') as text_src_file, \
                h5py.File(text_src_mask_path, 'r') as text_src_mask_file, \
                    h5py.File(trg_path, mode='r') as trg_file, \
                        io.open(nonreg_trg_path, mode='r', encoding='utf-8') as nonreg_trg_file, \
                            io.open(file_path, mode='r', encoding='utf-8') as files_file, \
                                h5py.File(facs_path, mode='r') as facs_file:
            
            for nonreg_line, files_line in zip(nonreg_trg_file, files_file):

                # Strip away the "\n" at the end of the line
                # trg_line, nonreg_line, files_line = trg_line.strip(), nonreg_line.strip(), files_line.strip()
                nonreg_line, files_line = nonreg_line.strip(), files_line.strip()

                line_subset, line_vidid_part = files_line.split('/')

                if line_subset not in ['train', 'dev', 'test']:
                    print("Incorrect subset, skipping line", files_line)
                    continue

                if line_vidid_part not in src_file[line_subset] or \
                    line_vidid_part not in trg_file[line_subset] or \
                        line_vidid_part not in facs_file[line_subset]:
                    print("Incorrect vidid, skipping line", files_line)
                    continue

                src_data = src_file[line_subset][line_vidid_part][()].astype(np.float32)
                if src_data.shape[0] == src_size:
                    src_data = src_data.T
                src_frames = src_data.tolist()

                trg_data = trg_file[line_subset][line_vidid_part][()].astype(np.float32)
                trg_data += 1e-8 # To avoid log(0) errors
                # TODO: Remove this hardcoded stuff since this is only for removing face keypoints for an experiment.
                valid_keypts_range = num_keypoints  # 85 - for body+face, 48 - for body only
                body_wcounter_indices = list(range(valid_keypts_range*3)) + [-1]
                trg_frames = trg_data[range(0, trg_data.shape[0], skip_frames)]

                # Divide the trg frames (keypoints) using a constant for DTW
                trg_frames[:, :-1] /= KEYPOINT_SCALE
                trg_frames = trg_frames[:, body_wcounter_indices].tolist()

                # if src_data.shape[0] > src_fps *  num_sec and np.array(trg_frames).shape[0]>trg_fps*num_sec:
                #     continue
                # if (len(src_frames) > src_fps*num_sec) and (len(trg_frames) > trg_fps*num_sec):
                #     # print("Too long src and trg, skipping line", files_line)
                #     continue

                # lemmantize the text translations
                src_wrds = nonreg_line.split(" ")
                lemma=[]
                for w in src_wrds:
                    if(wordnet_lemmatizer.lemmatize(w)=='wa'):
                        lemma.append(w)
                    elif(wordnet_lemmatizer.lemmatize(w)=='ha'):
                        lemma.append(w)
                    else:
                        lemma.append(wordnet_lemmatizer.lemmatize(w))
                
                nonreg_line = " ".join(lemma) 

                if LOAD_TEXT_SRC:
                    if line_vidid_part not in text_src_file[line_subset] or \
                        line_vidid_part not in text_src_mask_file[line_subset]:
                        print("Incorrect vidid, skipping line", files_line)
                        continue

                    text_src_data = text_src_file[line_subset][line_vidid_part][()].astype(np.float32)
                    text_src_frames = text_src_data.tolist()

                    text_src_mask_data = text_src_mask_file[line_subset][line_vidid_part][()].astype(np.uint8)
                    text_src_mask_frames = text_src_mask_data.tolist()
                elif LOAD_NONREG_TRG_AS_TEXT_SRC:
                    text_src_frames = nonreg_line
                    text_src_mask_frames = [[False]]
                else:
                    text_src_frames, text_src_mask_frames = [[0]], [[False]]

                if LOAD_FACS:
                    facs_data = facs_file[line_subset][line_vidid_part][()].astype(np.float32)
                    facs_frames = facs_data[range(0, len(facs_data), skip_frames)].tolist()
                else:
                    facs_data, facs_frames = None, [[0]]

                
                examples.append(data.Example.fromlist(
                    [src_frames[:src_fps*num_sec], text_src_frames, text_src_mask_frames, 
                     trg_frames[:trg_fps*num_sec], nonreg_line, files_line, facs_frames[:trg_fps*num_sec]], 
                     fields))
                
        print("Num of {} examples is {}".format(split, len(examples)))

        super(SignProdDataset, self).__init__(examples, fields, **kwargs)
