import concurrent.futures
import yt_dlp as youtube_dl
import os
import argparse

def download_video_and_subtitles(params):
    video_id, output_path = params

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'writesubtitles': True,
        'subtitlesformat': 'vtt',
        'noplaylist': True,
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'http://www.youtube.com/watch?v={video_id}'])


def read_video_ids_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos and subtitles.")
    parser.add_argument("--video_ids_file", help="Path to file containing YouTube video IDs")
    parser.add_argument("--output_path", help="Path to output directory")
    args = parser.parse_args()

    video_ids = read_video_ids_from_file(args.video_ids_file)
    
    args_list = [(video_id, args.output_path) for video_id in video_ids]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(download_video_and_subtitles, args_list)

if __name__ == "__main__":
    main()
