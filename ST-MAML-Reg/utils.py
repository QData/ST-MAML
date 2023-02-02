
import numpy as np

import glob
import os

def get_saved_file(path):
    filelist = glob.glob(path+'/*.pt')
    if len(filelist) == 0:
        return None
    else:
        pass
        
    index = np.array([int(os.path.splitext(os.path.basename(i).rsplit('_')[-1])[0]) for i in filelist])
    max_index = np.max(index)
    epoch = max_index
    
    return epoch