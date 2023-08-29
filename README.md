# MultiFacet-Speech-to-Sign

Implementation of the paper "MultiFacet: A Multi-Tasking Framework for Speech-to-Sign Language Generation" accepted at GENEA, ICMI 2023.

## Overview

This repository contains the implementation of an architecture proposed in the paper "MultiFacet: A Multi-Tasking Framework for Speech-to-Sign Language Generation." The architecture leverages prosodic information from speech audio and semantic context from text to generate sign pose sequences. The approach adopts a multi-tasking strategy that includes an additional task of predicting Facial Action Units (FAUs). FAUs capture intricate facial muscle movements crucial for conveying specific facial expressions during sign language generation. The models are trained on an existing Indian Sign language dataset comprising sign language videos with audio and text translations. Model evaluation involves Dynamic Time Warping (DTW) and Probability of Correct Keypoints (PCK) scores. The results show that combining prosody and text as input, along with incorporating facial action unit prediction as an additional task, outperforms previous models in terms of DTW and PCK scores.


## Dataset Download and Preprocessing

To obtain and preprocess the raw data, follow these steps:
1. Download the video ids from [this link](https://drive.google.com/drive/folders/1pwh9khgS-77tLXwNYXzxyKMBwrhcISwV?usp=sharing).
2. Execute `dowload_video_ydl.py` to download the videos along with subtitles.
3. Use `cut_segments.py` to extract short videos from long videos.
4. Utilize `mediapipe_extract_85.py` to extract keypoints for hands and face from segmented videos using Mediapipe.
5. Normalize keypoints.
6. The extracted GST embeddings can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mounika_k_research_iiit_ac_in/EjF2JMtFsE1Pm7aEO8WOyNMBdhoYsmdEEkm-jMI_GR8hNg?e=A9If6B). For extracting Tacotron 2 GST embeddings, follow the instructions provided in the [repository](https://github.com/NVIDIA/mellotron/tree/master) and resample videos accordingly.
7. The preprocessed FAUs can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mounika_k_research_iiit_ac_in/Eq_b-6AHA-tEg4zG5vQ146EBWJZWDwuyb4aGtAsvAS7Oow?e=hOgy5g). For FAUs extraction, refer to the [repository](https://github.com/CVI-SZU/ME-GraphAU#:~:text=action%20unit%20categories.-,Learning%20Multi%2Ddimensional%20Edge%20Feature%2Dbased%20AU%20Relation%20Graph,for%20Facial%20Action%20Unit%20Recognition&text=The%20main%20novelty%20of%20the,are%20illustrated%20in%20this%20figure). After extraction of FAUs, follow the below steps.
a. Thresholding,
b. Linear interpolation,
c. Hanning smoothing
8. The text embeddings were extracted using BERT. To download the extracted BERT embeddings for the ISL dataset, visit [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mounika_k_research_iiit_ac_in/EsEt2p7BIoxApqnkFGFhpycB9LOZ2lzUWNOM5yFD8kaz5w?e=JWMEMf)

## Download Pretrained Models
Pretrained models can be found [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/mounika_k_research_iiit_ac_in/EmhaLJ8HLilNmFqZIt8vx0oB9sfvNZ2PesuYn3BJ8G6wsQ?e=N7OGYB).



## Training 

To train the model, use the following command:

```bash 
python __main__.py train Configs/config.yml
```

## Evaluation

```bash 
python __main__.py test Configs/config.yml
```

## License and Citation
The software is licensed under the MIT License. Please include the following citation if you have used this code:

## Acknowledgements
This code is a modified version of this [repo](https://github.com/kapoorparul/Towards-Automatic-Speech-to-SL).
