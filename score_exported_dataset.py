import numpy as np 
import cv2
import os.path
import sys
import glob
import paiv_utils as paiv
import hashlib
import time
import json
from collections import defaultdict
from collections import deque
import argparse as ap

# example Invocations
# python score_exported_dataset.py --validate_mode=classification --model_url=https://129.40.2.225/powerai-vision/api/dlapis/8f80467f-470c-47f3-bf3c-ab7e0880a66b --data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_97_classification_augmented_dataset-test



def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


# Helpers ...
def plot_image(img, pause=0):
    cv2.imshow('frame', img)
    cv2.waitKey(pause) # if pause = 0, then hit escape ... else pause in ms
    cv2.destroyAllWindows()

class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Tool to validate a PowerAI Vision Model with an exported golden dataset'
                                           'Example Invocations'
                                            '  python score_exported_dataset.py --validate_mode=classification --model_url=https://129.40.2.225/powerai-vision/api/dlapis/8f80467f-470c-47f3-bf3c-ab7e0880a66b --data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_97_classification_augmented_dataset-test',
                                formatter_class=CustomFormatter)

    parser.add_argument(
        '--validate_mode', action='store', nargs='?', type=str.lower, required=True,
        choices=['object', 'classification'],
        help='S|--mode=[classification|object] '
             'Default: %(default)s')

    parser.add_argument(
        '--model_url', action='store', nargs='?',
        required=True,
        help='S|--model_url=<deployed model endpoint>')

    parser.add_argument(
        '--data_directory', action='store', nargs='?',
        required=True,
        help='S|--data_directory=<location of exported PAIV dataset>')

    #parser.add_argument(
    #    '--batch_size', type=int, default=64,
    #    help='S|Batch size. Default: %(default)s')
#
    #parser.add_argument(
    #    '--neuron_size', type=int, default=100,
    #    help='S|PDE network size. '
    #         'Default: %(default)s')
#
    #parser.add_argument(
    #    '--time_steps', type=int, default=20,
    #    help='S|Algorithm time steps.'
    #         'Default: %(default)s')
#
    #parser.add_argument(
    #    '--maxsteps', type=int, default=4000,
    #    help='S|Number of steps to run preferably divisible by 100. '
    #         'Default: %(default)s')
#
#
#
    #parser.add_argument(
    #    '--dtype', action='store', nargs='?', type=str.lower, const='float32',
    #    choices=['float32', 'float64'], default='float32',
    #    help='S|Default type to use. On GPU float32 should be faster.\n'
    #         'If TF  < 1.4.0 then float32 is used.\n'
    #         'Default: %(default)s')
#
    #parser.add_argument(
    #    '--valid_feed', action='store', nargs='?', type=int,
    #    const=256,  # 256 if valid_feed is specified but value not provided
    #    help='S|Run validation via feed_dict. Decouples validation but\n'
    #         'runs a bit slower. Set this flag otherwise not decoupled.\n'
    #         'Optionally specify validation size: 256 by default.')

    args = parser.parse_args()

    return args


def main():

    # 1. build function to read in dataset!
    # def fetch_scores(paiv_url, mode="video", num_threads=2, frame_limit=50, image_dir="na", video_fn="na", json_fn="fetch_scores.json"):
    args = _parser()

    #### Presets ############

    #--validate_mode=classification
    #--model_url=https://129.40.2.225/powerai-vision/api/dlapis/8f80467f-470c-47f3-bf3c-ab7e0880a66b
    #--data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_97_classification_augmented_dataset-test
    #paiv.fetch_scores(paiv_url=TRAINED_MODEL_EP, validate_mode=args.validate_mode, media_mode="image", image_dir=DATASET_DIR, paiv_results_file="fetch_scores.json")
    #paiv_dict = paiv.validate_model(paiv_results_file="fetch_scores.json",  image_dir=DATASET_DIR, validate_mode=args.validate_mode)

    #--validate_mode=object
    #--model_url=https://129.40.2.225/powerai-vision/api/dlapis/12a4a62a-7b67-488a-9c97-35b63768d4d7
    #--data_directory=/data/work/osa/2018-10-PSEG/datasets_local/dv_test_epri
    #paiv.fetch_scores(paiv_url=TRAINED_MODEL_EP, validate_mode=args.validate_mode, media_mode="image", image_dir=DATASET_DIR, paiv_results_file="fetch_scores.json")
    #paiv_dict = paiv.validate_model(paiv_results_file="fetch_scores.json",  image_dir=DATASET_DIR, validate_mode=args.validate_mode)#

    paiv.fetch_scores(paiv_url=args.model_url, validate_mode=args.validate_mode, media_mode="image", image_dir=args.data_directory , paiv_results_file="fetch_scores.json")
    paiv_dict = paiv.validate_model(paiv_results_file="fetch_scores.json",  image_dir=args.data_directory, validate_mode=args.validate_mode)#


    # 2.  foreach file, hit api and score keep resutl

    # 3.  build truth table as a first cut...
    #   add in MAP, IOU etc etc ....
    #


if __name__== "__main__":
  main()

# Todo : add threading for quicker video building
# Todo : add custom logic for tracking ball touches with denoising / smoothing
# Todo : add custom logic for displaying number of players at any given time with denoising / smoothing
