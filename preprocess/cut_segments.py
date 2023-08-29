import os
import subprocess as sp
import glob
import webvtt
import argparse
from tqdm import tqdm

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

''' To break longer videos into short segments using vtt file downloaded along with videos'''

def cut_segments(file, args):
    # for file in files:

    in_vid_file = file 
    time_chunks = []
    print('cutting')
    video = os.path.basename(in_vid_file)[:-4].replace(' ','_') ## remove spaces from video name

    dst_dir = args.dest_dir
    vid_dir = args.vid_dir

    if not os.path.exists(os.path.join(dst_dir, video)):
        os.makedirs(os.path.join(dst_dir, video))
        os.makedirs(os.path.join(dst_dir, video, "audio"))
        os.makedirs(os.path.join(dst_dir, video, "text"))
    else:
        print("already exists ", video)
        #continue

    if not os.path.exists(os.path.join(vid_dir, video)):
        os.makedirs(os.path.join(vid_dir, video))

    prev_start = '00:00:01.000'
    subs = os.path.join(args.source_dir, os.path.basename(in_vid_file)[:-4] + '.en.vtt')

    '''
    If subtitles are from t1-t2 and t3-t4 then t5-t6
    then vid1  = t1,t3 (initially t1 = 00:00:01.000)
    next vid2   = t3,t5 and so on
    .
    .
    '''
    print('reading subs')
    for i, caption in enumerate(webvtt.read(subs)):
        if i==0:
            prev_caption = caption.text
            continue
        end_time = caption.start
        time_chunks.append((prev_start, end_time, prev_caption))

        prev_start = caption.start
        prev_caption = caption.text
    
    print('done reading')

    for i, t in enumerate(time_chunks):
        beg_t = t[0]
        end_t = t[1]
        text = t[2]

        out_vid_file = os.path.join(vid_dir, video, str(i).zfill(4)+'.mp4')
        out_aud_file = os.path.join(dst_dir, video, "audio", str(i).zfill(4)+'.wav')
        out_text_file = os.path.join(dst_dir, video, "text", str(i).zfill(4)+'.txt')

        text = text.encode('ascii', 'ignore').decode('ascii')

        cmd = ["ffmpeg",  "-i", in_vid_file, "-ss", beg_t, "-to", end_t, "-y", "-avoid_negative_ts", "1",  "-vcodec", "h264", "-acodec" ,"copy", out_vid_file]
        sp.run(cmd, stderr=sp.STDOUT) ## to get video for this segment

        cmd = ["ffmpeg", "-i",  out_vid_file, "-ab", "160k", "-ac", "2", "-ar", "16000", "-vn", out_aud_file]
        sp.run(cmd, stderr=sp.STDOUT)  ## to get audio for this segment


        with open(out_text_file, "w") as text_file:
            # print('text', text) 
            text_file.write(text)  ## save text

if __name__ == '__main__':


        parser = argparse.ArgumentParser()
        parser.add_argument('-j', '--jobs', help='Number of jobs to run in parallel', default=40, type=int)
        parser.add_argument('-src', '--source_dir', help='Directory containing mp4 and vtt files', required=True)
        parser.add_argument('-dest', '--dest_dir', help="Folder where final segmented audio and text files are stored", required=True)
        parser.add_argument('-vid', '--vid_dir', help='Cut Videos', required=True)
        args = parser.parse_args()

        # data = preprocess(args)
        files = glob.glob(os.path.join(args.source_dir, '*.mp4'))

        print('files', files)

        p = ThreadPoolExecutor(args.jobs)

        threads = [p.submit(cut_segments, file, args) for file in files]
        _ = [r.result() for r in tqdm(as_completed(threads), total=len(threads))]