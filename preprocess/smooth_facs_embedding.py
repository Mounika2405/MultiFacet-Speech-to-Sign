import os
import sys
import h5py
import numpy as np
import pandas as pd


def smooth(x,window_len=11,window='hanning', masked=True):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    if masked:
        y=np.ma.convolve(w/w.sum(),s,mode='valid')
    else:
        y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def smooth_facs(facs, window_len=11, window='hamming'):
    si,ei = window_len//2, -window_len//2 + 1
    smoothed_facs = np.zeros_like(facs)

    for i in range(facs.shape[1]):
        facs_component = facs[:, i]
        corrected_facs = facs_component.copy()
        corrected_facs[corrected_facs == 0] = np.nan
        corrected_facs = pd.Series(corrected_facs).interpolate(method='linear').values
        corrected_facs = np.nan_to_num(corrected_facs, nan=0)

        corrected_facs_smooth = smooth(corrected_facs, window_len=window_len, 
                                    window=window, masked=False)[si:ei]
        
        smoothed_facs[:, i] = corrected_facs_smooth
    
    return smoothed_facs

subsets = ['train', 'dev', 'test']
window_len = 11
assert(window_len % 2 == 1)
window = "hamming"

files_dir = sys.argv[1]

for subset in subsets:
    facs_embed_fp = f'{files_dir}/{subset}.facs_embedding.h5'
    facs_outfp = f'{files_dir}/{subset}.smoothed_facs_embedding.h5'
    if not os.path.exists(facs_embed_fp):
        continue
    with h5py.File(facs_embed_fp) as f, \
        h5py.File(facs_outfp, 'w') as g:
        for filename in f[subset].keys():
            try:
                if subset not in g.keys():
                    g.create_group(subset) 
                if filename in g[subset]:
                    continue
                facs = f[subset][filename][()]
                smoothed_facs = smooth_facs(facs, window_len=11, window='hamming')
                smoothed_facs = smoothed_facs.astype(np.float16)
                g.create_dataset(f'{subset}/{filename}', 
                                data=smoothed_facs, dtype=smoothed_facs.dtype)
            except Exception as e:
                print(filename, e)
                continue
        print(subset, len(f[subset].keys()), len(g[subset].keys()))
