import os
import sys
import pickle
import tqdm
import multiprocessing as mp
from yt_dlp import YoutubeDL

ydl_params = dict(quiet=True, no_warnings=True, skip_download=True, verbose=False)

def vidid_to_name(vidid):
    with YoutubeDL(params=ydl_params) as ydl:
        info_dict = ydl.extract_info(f"https://youtube.com/watch?v={vidid}", download=False)
        video_title = info_dict.get('title', None)
        return video_title

def format_name(vidname):
    # Returns a formatted name for the video for matching with aus_dict keys
    result = vidname.replace(' ', '_')
    result = result.replace('/', '_')

    # Replace special characters
    result = result.replace('‘', '')
    result = result.replace('’', '')
    result = result.replace('₹', '')
    result = result.replace('“', '')
    result = result.replace('”', '')
    return result

def get_aus_dict(aus_dir):
    # Returns a dict mapping formatted name to actual directory name
    aus_dict = {}
    dirnames = os.listdir(aus_dir)
    for dirname in dirnames:
        formatted_dirname = dirname.replace('#U2018', '')
        formatted_dirname = formatted_dirname.replace('#U2019','')
        formatted_dirname = formatted_dirname.replace('#U20b9','')
        formatted_dirname = formatted_dirname.replace('#U201c','')
        formatted_dirname = formatted_dirname.replace('#U201d','')

        if '#U' in formatted_dirname:
            print(dirname)

        # Add to dict
        aus_dict[formatted_dirname] = dirname
    return aus_dict

def get_formatted_name_and_part(line):
    try:
        line_subset, line_vidid_part = line.split('/')
        part = line_vidid_part.split('_')[-1]
        vidid = line_vidid_part.replace(f'_{part}', '')
        name = vidid_to_name(vidid)
        formatted_name = format_name(name)
        return (line, formatted_name, part)
    except:
        return (line, None, None)


files_dir = sys.argv[1]
aus_dir = sys.argv[2]
save_dir = sys.argv[3]

subsets = ['dev', 'test', 'train']

for subset in subsets:
    print(f'Processing {subset} files')
    aus_dict = get_aus_dict(os.path.join(aus_dir, subset))
    vidid_to_name_dict = {}
    with open(os.path.join(files_dir, f'{subset}.files'), 'r') as f:
        lines = f.read().splitlines()
    
    pool = mp.Pool(10)
    with tqdm.tqdm(total = len(lines)) as pbar:
        for (line, formatted_name, part) in pool.imap_unordered(get_formatted_name_and_part, lines):
            if formatted_name is None:
                print(f'Not found: {line}')
                continue
            if formatted_name not in aus_dict:
                print(f'Not found in aus_dict: {formatted_name}')
                continue
            vidid_to_name_dict[line] = (aus_dict[formatted_name], part)
            pbar.update()

    out_fp = os.path.join(save_dir, f'{subset}.vidid_to_name.pkl')
    with open(out_fp, 'wb') as f:
        pickle.dump(vidid_to_name_dict, f)
        print('Saved {} out of {} lines for {} in {}'.format(
            len(vidid_to_name_dict), len(lines), subset, out_fp))

        




