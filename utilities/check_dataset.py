import torch

from torchvision.io import read_video
from torchvision.transforms import v2

from pathlib import Path

from tqdm import tqdm


# if __name__ == "__main__":

#     data_dir = Path("/scratch/olg7848/MIMIC_ECHO")

#     filepaths = list(data_dir.rglob("*.avi"))

#     for filepath in tqdm(filepaths):
#         vframes, _, _ = read_video(str(filepath.resolve()), pts_unit='sec', output_format='TCHW')
#         num_frames = vframes.shape[0]
#         if num_frames == 0:
#             print(str(filepath.resolve()), num_frames)

import multiprocessing

def process_video(filepath):
    vframes, _, _ = read_video(str(filepath.resolve()), pts_unit='sec', output_format='TCHW')
    num_frames = vframes.shape[0]
    if num_frames == 0:
        print(str(filepath.resolve()), num_frames)
    elif num_frames == 1:
        print(str(filepath.resolve()), num_frames, 1)


def main():
    data_dir = Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p15/")
    filepaths = list(data_dir.rglob("*.avi"))
    
    # Determine the number of processes to use; you might need to adjust this based on your system
    num_processes = multiprocessing.cpu_count() 
    
    # Create a pool of workers to execute processes
    with multiprocessing.Pool(8) as pool:
        # Map process_video to each filepath
        list(tqdm(pool.imap(process_video, filepaths), total=len(filepaths)))

if __name__ == "__main__":
    main()


