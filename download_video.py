import subprocess
import multiprocessing
import argparse

def download_video(video_id, output_dir):
    # set the command to download the video and its corresponding subtitle and save them to the specified output directory
    command = [
        "youtube-dl",
        "--verbose",
        "--write-sub",
        "--sub-lang", "en",
        "--convert-subs", "vtt",
        "--extract-audio",
        "--audio-format", "mp3",
        "--output", f"{output_dir}/%(title)s.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}"
    ]

    # execute the command using subprocess
    subprocess.run(command)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-idf", "--input_file", help="path to file containing video ids")
    parser.add_argument("-od", "--output_dir", help="path to output directory")

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    # read the video IDs from the input file
    with open(input_file, "r") as f:
        video_ids = f.read().splitlines()

    # create a multiprocessing pool with the number of processes you want to use
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # download the videos and subtitles using multiprocessing
    results = []
    for video_id in video_ids:
        results.append(pool.apply_async(download_video, args=(video_id, output_dir)))

    # wait for all processes to finish
    pool.close()
    pool.join()

    # get the results from the multiprocessing pool
    for result in results:
        result.get()
