import cv2
import os
import numpy as np
import mediapipe as mp
from glob import glob
import pickle
import argparse
from tqdm.auto import tqdm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_holistic_keypoints(frames, holistic=mp_holistic.Holistic(static_image_mode=False, model_complexity=2, refine_face_landmarks=True)):
    keypoints = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        pose_landmarks = results.pose_landmarks
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        face_landmarks = results.face_landmarks
        # print(left_hand_landmarks, right_hand_landmarks)

        pose_kps = np.zeros((6, 3))
        left_hand_kps = np.zeros((21, 3))
        right_hand_kps = np.zeros((21, 3))
        face_kps = np.zeros((37, 3))

        if pose_landmarks:
            pose_kps = np.array(
                [
                    [pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y, pose_landmarks.landmark[11].z],
                    [pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y, pose_landmarks.landmark[12].z],
                    [pose_landmarks.landmark[13].x, pose_landmarks.landmark[13].y, pose_landmarks.landmark[13].z],
                    [pose_landmarks.landmark[14].x, pose_landmarks.landmark[14].y, pose_landmarks.landmark[14].z],
                    [pose_landmarks.landmark[23].x, pose_landmarks.landmark[23].y, pose_landmarks.landmark[23].z],
                    [pose_landmarks.landmark[24].x, pose_landmarks.landmark[24].y, pose_landmarks.landmark[24].z]
                ]
            )

        if left_hand_landmarks:
            left_hand_kps = np.array(
                [[p.x, p.y, p.z] for p in left_hand_landmarks.landmark]
            )

        if right_hand_landmarks:
            right_hand_kps = np.array(
                [[p.x, p.y, p.z] for p in right_hand_landmarks.landmark]
            )
        face_indices = [0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397, 468, 473]
        if face_landmarks:
            face_kps = np.array(
                [
                    [face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z]
                    for i in face_indices  # changed to range(68) to get keypoints related to eyes, eyebrows, lips, and face outline
                ]
            )
        
        keypoints.append(np.concatenate([pose_kps.flatten(), left_hand_kps.flatten(), right_hand_kps.flatten(), face_kps.flatten()]))

    return np.array(keypoints)

def load_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, img = vidcap.read()
        if not success:
            break
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (640, 480))
        frames.append(img)

    vidcap.release()
    # cv2.destroyAllWindows()
    return np.asarray(frames)


    
        
def gen_keypoints_for_video(video_path, save_path):
    print('Processing', video_path)
    if not os.path.isfile(video_path):
        print("SKIPPING MISSING FILE:", video_path)
        return
    frames = load_frames_from_video(video_path)
    kps = get_holistic_keypoints(frames)
    # print(frames.shape, kps.shape)
    #  s, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', help='Number of jobs to run in parallel', default=5, type=int)
    parser.add_argument('-vid', '--vid_dir', help='Directory containing video segments', required=True)
    parser.add_argument('-dest', '--dest_dir', help='Destination directory for keypoints', required=True)
    args = parser.parse_args()

    videos = [name for name in os.listdir(args.vid_dir) if os.path.isdir(os.path.join(args.vid_dir, name))]
    
    for video in videos:
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_save_dir = os.path.join(args.dest_dir, video_name, "OP")
        os.makedirs(video_save_dir, exist_ok=True)

        video_src_dir = os.path.join(args.vid_dir, video_name)
        
        segments = glob(os.path.join(video_src_dir, "*.mp4"))
        # print(video_src_dir, video_save_dir, segments)
        for segment in tqdm(segments):
            # try:
            # print(segment)
            segment_name = os.path.basename(segment)  # Extracts the filename from the path
            segment_name_without_extension = os.path.splitext(segment_name)[0]  # Removes the extension
            segment_name = segment_name_without_extension.split('_')[-1] 
            segment_path = os.path.join(args.dest_dir, video_name, "OP", segment_name)
            # print(segment_path)
            gen_keypoints_for_video(segment, segment_path)

