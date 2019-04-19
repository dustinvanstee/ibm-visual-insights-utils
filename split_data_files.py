import numpy as np 
import glob
import shutil
import os

print("="*90)
print("Warning : Works for classification only for now! ")
print("="*90)

INDIR = "/data/work/osa/2018-10-PSEG/datasets/ct_97_classification_augmented_dataset"
OUTDIR_TEST = INDIR + "-test"
OUTDIR_TRAIN = INDIR + "-train"
if(os.path.exists(OUTDIR_TEST)) :
    shutil.rmtree(OUTDIR_TEST)
if(os.path.exists(OUTDIR_TRAIN)) :
    shutil.rmtree(OUTDIR_TRAIN)

os.mkdir(OUTDIR_TEST)
os.mkdir(OUTDIR_TRAIN)

TEST_SAMPLE_PCT = 0.20

files = glob.glob(INDIR + "/*.JPG")
files = files + glob.glob(INDIR + "/*.jpg")
files = files + glob.glob(INDIR + "/*.PNG")
files = files + glob.glob(INDIR + "/*.png")
print(len(files))

file_range = list(range(len(files)))

file_test_idx = sorted(np.random.choice(file_range, int(TEST_SAMPLE_PCT*len(files)), replace=False))

j = 0
for i in range(len(files)) :
    # This file goes into test bucket
    print("i {} j {} file_test_idx[j] {} ".format(i,j,file_test_idx[j]))
    if(i == file_test_idx[j]) :
        #copy file to test bucket
        shutil.copy(files[i], OUTDIR_TEST + "/")
        j += 1
    else :
        # copy flie to train bucket
        shutil.copy(files[i], OUTDIR_TRAIN + "/")

shutil.copy(INDIR + "/prop.json", OUTDIR_TRAIN + "/" )
shutil.copy(INDIR + "/prop.json", OUTDIR_TEST + "/")
