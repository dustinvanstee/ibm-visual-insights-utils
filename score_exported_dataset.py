import numpy as np 
import cv2
import os.path
import sys
import glob
import pu.paiv_utils as paiv
import hashlib
import time
import json
from collections import defaultdict
from collections import deque

#### Presets ############

TRAINED_MODEL_EP ='https://129.40.2.225/powerai-vision/api/dlapis/12a4a62a-7b67-488a-9c97-35b63768d4d7'
DATASET_DIR = "/data/work/osa/2018-10-PSEG/paiv_code/dv_test_epri"

# the ip for pseg - 129.40.2.225




def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


#######
# class pseg():
#     def __init__(self,orig_video,paiv_video):
#         self.orig_video = orig_video
#         self.paiv_video = paiv_video
#         self.is_file(self.orig_video)
#         self.is_file(self.paiv_video)
#         self.video_shape = None
#
#     def is_file(self,file) :
#         a=1



# Helpers ...
def plot_image(img, pause=0):
    cv2.imshow('frame', img)
    cv2.waitKey(pause) # if pause = 0, then hit escape ... else pause in ms
    cv2.destroyAllWindows()

def main():

    # 1. build function to read in dataset!
    # def fetch_scores(paiv_url, mode="video", num_threads=2, frame_limit=50, image_dir="na", video_fn="na", json_fn="fetch_scores.json"):

   paiv.fetch_scores(paiv_url=TRAINED_MODEL_EP, mode="image", image_dir=DATASET_DIR, paiv_results_file="fetch_scores.json")
    paiv_dict = paiv.validate_model(paiv_results_file="fetch_scores.json",  image_dir=DATASET_DIR)

    # 2.  foreach file, hit api and score keep resutl

    # 3.  build truth table as a first cut...
    #   add in MAP, IOU etc etc ....
    #


if __name__== "__main__":
  main()

# Todo : add threading for quicker video building
# Todo : add custom logic for tracking ball touches with denoising / smoothing
# Todo : add custom logic for displaying number of players at any given time with denoising / smoothing
