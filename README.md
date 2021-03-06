# powerai-vision-utils

## Sample code and utilities library for working with powerai vision

---
**Prerequisites**

You will need a python 3.X environment (recommend Anaconda) for this program.  You will need to install the following libraries
* opencv
* requests
* urllib3
* numpy

---

**Example 1 : Create a customized video with bounding box annotations**

Program : **annotate_video.py**

Parameters 
* --input_video [Path to File]  (tested with mp4 and mov so far)
* --model_url [PowerAI Vision Model deployment endpoint]
* --output_directory [directory where you want the movie to be saved]
* --sample_rate [integer : controls sample rate of video to downsample in cases of large videos]


Example incantation

python annotate_video.py --input_video  /tmp/myvideo.mp4 --model_url https://xxx.xxx.xxx.xxx/powerai-vision/api/dlapis/bda90858-45e4-4ca6-8161-7d63436bb6c6 --output_directory /tmp/output --sample_rate 20

---
