import pickle

import sys
import math
import numpy as np
import cv2
import os
import _pickle as cPickle
import gzip
import subprocess
import torch

import librosa
import soundfile as sf

import subprocess

try:
    from dtw import dtw
    from constants import PAD_TOKEN, KEYPOINT_SCALE
except:
    PAD_TOKEN = '<pad>'
    dtw = lambda *args, **kwargs: None


# last_bone_length = dict()
# MAX_BONE_LEN = 200 # Based on image size in pixels
# MIN_BONE_LEN = 10
# MAX_BONE_LEN_RATIO = 10

def writeAudio(vid_loc, audio_loc):
    waveform, sr = librosa.load(audio_loc, sr=16000)
    waveform, index = librosa.effects.trim(waveform) 

    # librosa.output.write_wav("tmp.wav", waveform, sr)
    if os.path.exists("/tmp/tmp.wav"):
        os.remove("/tmp/tmp.wav")
    sf.write("/tmp/tmp.wav", waveform, sr)
    new_vid_loc = vid_loc.split(".avi")[0] + "_audio.mp4"
    
    # cmd = "ffmpeg -loglevel panic -i " + vid_loc + " -i /tmp/tmp.wav"  # temporary .wav audio location
    # cmd += " -c:v copy -c:a aac -strict experimental " + new_vid_loc
    # os.system(cmd)
    cmd = ["ffmpeg", "-loglevel", "panic", "-i", vid_loc, "-i", "/tmp/tmp.wav", "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", new_vid_loc]
    subprocess.run(cmd)
    os.remove("/tmp/tmp.wav")

    
    return new_vid_loc

# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None,
               audio_path=None):
    
    # print('joints.shape',joints.shape)
    # joints = normalize_skeleton_landmarks(joints)

    # Create video template
    FPS = (25 // skip_frames)
    video_file = file_path + video_name.split("/")[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    num_frames = 0

    timesteps = joints.shape[0]
    num_keypts = (joints.shape[0] *  (joints.shape[1]-1)) // (timesteps * 3)
    
    if references is not None:
        pred_joints_2d = joints[:, :-1].reshape((timesteps, num_keypts, 3))[..., :2]
        pred_shoulder_len = np.mean(np.linalg.norm(pred_joints_2d[:, 0] - pred_joints_2d[:, 1]))
        
        ref_joints_2d = references[:, :-1].reshape((timesteps, num_keypts, 3))[..., :2]
        ref_shoulder_len =  np.mean(np.linalg.norm(ref_joints_2d[:, 0] - ref_joints_2d[:, 1]))
        
        ref_shoulder_mean = ref_joints_2d[:, [0, 1], :].mean(axis=1, keepdims=True).mean(axis=0, keepdims=False)
        shoulder_ratio = pred_shoulder_len / ref_shoulder_len
    else:
        ref_shoulder_mean = None
        shoulder_ratio = None
        
    # print(ref_shoulder_mean, shoulder_ratio)

    for (j, frame_joints) in enumerate(joints):

        # Reached padding
        if PAD_TOKEN in frame_joints:
            continue

        # Initialise frame of white
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Cut off the percent_tok, multiply by 3 to restore joint size
        frame_joints = frame_joints[:-1] * KEYPOINT_SCALE
        # TODO - Remove the *3 if the joints weren't divided by 3 in data creation
        # frame_joints = frame_joints[:-1] * 3

        # Reduce the frame joints down to 2D for visualisation - Frame joints 2d shape is (75,2)
        frame_joints_2d = np.reshape(frame_joints, (-1, 3))[:, :2]
        # Draw the frame given 2D joints
        draw_frame_2D(frame, frame_joints_2d)

        cv2.putText(frame, "Predicted Sign Pose", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            ref_joints = ref_joints[:-1] * KEYPOINT_SCALE 
            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
            # ref_joints = ref_joints[:-1] * 3

            # Reduce the frame joints down to 2D- Frame joints 2d shape is (75,2)
            ref_joints_2d = np.reshape(ref_joints, (-1, 3))[:, :2]
            
            # Zero-mean and rescale this using ratio of shoulder lengths
            ref_joints_2d = (ref_joints_2d - ref_shoulder_mean) * shoulder_ratio

            # Draw these joints on the frame
            draw_frame_2D(ref_frame, ref_joints_2d)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)

            sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
            cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
        # Write the video frame
        video.write(frame)
        num_frames += 1
    # Release the video
    if audio_path is not None:
        writeAudio(video_file, audio_path)
    # Release the video
    video.release()

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # Shoulder and Arms
        (0,1,0),
        (0,2,1),
        (1,3,1),

        # Waist
        (0,4,2),
        (1,5,2),
        (4,5,2),

        # Left Hand
        (2, 6, 20),
        (6, 7, 0),
        (7, 8, 1),
        (8, 9, 2),
        (9, 10, 3),
        (6, 11, 4),
        (11, 12, 5),
        (12, 13, 6),
        (13, 14, 7),
        (11, 15, 8),
        (15, 16, 9),
        (16, 17, 10),
        (17, 18, 11),
        (15, 19, 12),
        (19, 20, 13),
        (20, 21, 14),
        (21, 22, 15),
        (19, 23, 16),
        (23, 24, 17),
        (24, 25, 18),
        (25, 26, 19),

        # Right Hand
        (3, 27, 20),
        (27, 28, 0),
        (28, 29, 1),
        (29, 30, 2),
        (30, 31, 3),
        (27, 32, 4),
        (32, 33, 5),
        (33, 34, 6),
        (34, 35, 7),
        (32, 36, 8),
        (36, 37, 9),
        (37, 38, 10),
        (38, 39, 11),
        (36, 40, 12),
        (40, 41, 13),
        (41, 42, 14),
        (42, 43, 15),
        (40, 44, 16),
        (44, 45, 17),
        (45, 46, 18),
        (46, 47, 19),

        # Face - Nose
        (60, 49, 0),
        (49, 77, 0),

        # Face - Nose bridge
        (49, 65, 1),

        # Face - Right eyebrow
        (56, 57, 2),
        (57, 58, 2),

        # Face - Left eyebrow
        (73, 74, 2),
        (74, 75, 3),

        # Face - Right eye
        (53, 67, 3),
        (67, 64, 3),
        # (64, 83, 3),
        # (83, 53, 3),
        (64, 53, 3),

        # Face - Left eye
        (80, 81, 3),
        (81, 71, 3),
        # (71, 84, 3),
        # (84, 80, 3),
        (71, 80, 3),

        # Face - mouth
        (59, 55, 4),
        (55, 54, 4),
        (54, 48, 4),
        (48, 72, 4),
        (72, 76, 4),
        (76, 52, 4),
        (52, 70, 4),
        (70, 59, 4),

        # Face - lips
        (59, 61, 5),
        (61, 62, 5),
        (62, 50, 5),
        (50, 78, 5),
        (78, 76, 5),
        (76, 51, 5),
        (51, 69, 5),
        (69, 59, 5),

        # Face - jaw
        (63, 68, 6),
        (68, 66, 6),
        (66, 82, 6),
        (82, 79, 6),
    )

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints, width=1):
    # Line to be between the stacked
    draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)
    # Give an offset to center the skeleton around
    offset = [350, 250]

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]
    num_keypts = joints.shape[0]

    # Increase the size and position of the joints
    joints = joints * 6 * 12 * 2 #10 * 12 * 2
    joints = joints + np.ones_like(joints) * offset

    # Ignore indices 
    ignore_idx = [17, 18, 19, 20, 21, 
                  22, 25, 26, 27, 28, 
                  29, 30, 31, 32]

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):

        (s, e, b) = skeleton[j]
        # if s in ignore_idx or e in ignore_idx:
        #     continue

        if s >= num_keypts or e >= num_keypts:
            continue

        c = get_bone_colour(skeleton,j)

        # length = np.linalg.norm(joints[s] - joints[e])
        # if (s, e) not in last_bone_length:
        #     last_bone_length[(s, e)] = max(length, MIN_BONE_LEN)
        # else:
        #     len_ratio = length / last_bone_length[(s, e)]
        #     if len_ratio > MAX_BONE_LEN_RATIO or length > MAX_BONE_LEN:
        #         continue
        #     last_bone_length[(s, e)] = max(length, MIN_BONE_LEN)

        draw_line(frame, [joints[s][0], joints[s][1]],
                  [joints[e][0], joints[e][1]], c=c, t=1, width=width)
        
    # Plot eyes
    eye_indices = [83, 84]
    for i in eye_indices:
        if i >= num_keypts:
            continue
        cv2.circle(frame, (int(joints[i][0]), int(joints[i][1])), 2, (0, 0, 255), -1)

# get bone colour given index
def get_bone_colour(skeleton,j):
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2:  # left arm
        c = (0, 102, 204)
    elif bone == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 4:  # right arm
        c = (0, 153, 0)
    # elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
    #     c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)

    return c

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    _ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cut the hypothesis down to the max count value
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()

    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame

def normalize_skeleton_landmarks(keypoints):
    left_shoulder, right_shoulder = 1, 0 #12, 11
    mid = keypoints[:, [left_shoulder, right_shoulder], :].mean(axis=1, keepdims=True)

    shoulder_length = np.linalg.norm(keypoints[:, left_shoulder, :] - keypoints[:, right_shoulder, :], ord=2, axis=1)
    normalized_keypts = (keypoints - mid)/shoulder_length[:, None, None]
    return normalized_keypts

