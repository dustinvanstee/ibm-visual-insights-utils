# paiv_utils.py

# Set of utility functions and classes to help me with PAIV projects
import cv2
import json
import requests
import urllib3
import numpy as np
import colorsys
import random
import sys
from queue import Queue
from threading import Thread
import glob
import xml.etree.ElementTree
import os
import hashlib
import time
from sklearn.metrics import confusion_matrix
import sklearn_utils as su
from collections import defaultdict
import pdb
# functions that start with _ imply that these are private functions

def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


def _convert_xml(filein) :
    e = xml.etree.ElementTree.parse(filein).getroot()
    #class_list = []
    #box_list = []
    rv_list = []
    for obj in e.findall('object') :
        objclass = obj.find('name').text
        objclass = objclass.replace(' ','_')
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        if(xmin < 0.0 or ymin < 0.0 or xmax < 0 or ymax <0 ) :
            nprint("Error, file {} somehow has a negative pixel location recorded.  Omitting that box ....".format(filein))
        else :
            box_dict = {"label" : objclass, "xmin" : xmin, "ymin" : ymin, "xmax" : xmax, "xmin" : xmin}
            #class_list.append(objclass)
            #box_list.append([xmin,ymin,xmax,ymax])
            rv_list.append(box_dict)
    # zip the class and box dimensions !
    #rv = list(zip(class_list,box_list))
    #image_name =  get_image_from_xml(filein)
    #pdb.set_trace()
    #write_coco_labels(experiment_dir,image_name,0.0,box_list,class_list,cn_dict,labels_file)

    return rv_list

def get_image_fn(paiv_file_name_base) :
    image_file = paiv_file_name_base + ".JPG"
    if(not(os.path.isfile(image_file))):
        image_file = paiv_file_name_base + ".jpg"
    if(not(os.path.isfile(image_file))):
        image_file = paiv_file_name_base + ".png"
    if(not(os.path.isfile(image_file))):
        image_file = None

    return image_file

# Hash function to uniquely identify an image by just pixel values
def get_np_hash(npary) :
    return hashlib.md5(npary.data).hexdigest()

#  Function to Read in an AI Vision data directory and return a python object (dict)
#  to be used for all kinds of experiments.
#  supports both object, classification modes (validate_mode setting)
#  returns a dictionary of dictionaries
#    top level dictionary keys are hash of np array
#      second level dictionary is the metadata associated with the image
def _load_paiv_dataset(data_dir, validate_mode) :
    paiv_dataset = {}
    if(os.path.exists(data_dir)) :
        os.chdir(data_dir)

        if(validate_mode == "object") :
            xml_file_list = glob.glob("*.xml")
            for xf in xml_file_list :
                (xf_file_base,junk) = xf.split(".")
                image_fn = get_image_fn(xf_file_base)
                np_hash = get_np_hash(cv2.imread(image_fn))
                paiv_dataset[np_hash] = {'id' : xf_file_base , 'boxes' : _convert_xml(xf)}
        else : # classification
            # read in prop.json file and stuff into hash !
            json_str = open(data_dir + "/prop.json").read()
            json_parsed = json.loads(json_str)
            file_class_list = json.loads(json_parsed['file_prop_info'])

            for i in file_class_list :
                xf_file_base = i['_id']
                # print(xf_file_base)

                image_fn = get_image_fn(xf_file_base)
                if(image_fn != None) :
                    np_hash = get_np_hash(cv2.imread(image_fn))
                    paiv_dataset[np_hash] = {'id' : xf_file_base, 'class' : i['category_name']}  # save the class for the image!


    else :
        nprint("Data Directory {} does not exist".format(data_dir))

    return paiv_dataset


def validate_model(paiv_results_file,  image_dir,  validate_mode) :
    '''
    Validate model : 
        prerequisites : 
        1.  first need to run paiv.fetch_scores to build a paiv_results_file.  This file contains
        all the scored results from hitting a PAIV model with images/video from a directory
        2.  need to specify the image directory where the source images exist
        
        3.  validate_mode is based on either doing object detection or classification validation ['object'|'classification']


    '''
    #test exists
    nprint("Loading Dataset to get ground truth labels")
    ground_truth = _load_paiv_dataset(image_dir, validate_mode)

    model_predictions_json = open(paiv_results_file).read()
    model_predictions = json.loads(model_predictions_json)

    #  Truth table
    # use sklearn metrics confusion_matrix.  need to build a long list of y_true, y_pred
    tt_label = {}
    ytrue_cum = []
    ypred_cum = []
    for mykey in model_predictions.keys() :
        # Each prediction could have a list of boxes ..
        print("model prediction : {}".format(model_predictions[mykey]))
        print("ground_truth     : {}".format(ground_truth[mykey]))
        if(validate_mode == 'object') :
            (ytrue, ypred) = return_ytrue_ypre_objdet(ground_truth[mykey]['boxes'], model_predictions[mykey]['classified'])
        elif(validate_mode == 'classification') :
            (ytrue, ypred) = return_ytrue_ypre_classification(ground_truth[mykey], model_predictions[mykey]['classified'])
        else :
            nprint("Error : invalid validate_mode passed")
        ytrue_cum = ytrue_cum + ytrue
        ypred_cum = ypred_cum + ypred
    # Function to automatically fetch data from API in threaded mode...
    # modes = video / image_dir
    classes = list(set(ytrue_cum+ypred_cum))
    cm = confusion_matrix(ytrue_cum, ypred_cum, labels=classes)
    su.plot_confusion_matrix(cm, classes, title='confusion matrix',normalize=False)
    print(cm)
    diag_sum = 0
    total_sum = np.sum(cm)
    for i in range(len(classes)) :
        tp = float(cm[i,i])
        diag_sum += tp
        tpfp = float(np.sum(cm[:,i]))
        tpfn = float(np.sum(cm[i,:]))
        fn = tpfn - tp
        fp = tpfp - tp
        if(tpfp != 0) :
            precision = tp / tpfp
        else :
            precision = 0.0

        if(tpfn != 0) :
            recall    = tp / tpfn
        else :
            recall = 0.0

        #nprint("class  = {} TP = {}  TP+FP ={}  TP+FN = {} ".format(classes[i], cm[i,i],np.sum(cm[:,i]),np.sum(cm[i,:])))
        nprint("class = {} : tp = {} : fp = {} : fn = {} : Precision = {:0.2f}  Recall = {:0.2f}".format(classes[i],tp,fp, fn,precision,recall))
    nprint("Overall Accuracy = {:0.2f}".format(diag_sum/total_sum))

def return_ytrue_ypre_classification(ground_truth, model_predictions) :
    # Examples of whats passed
    #model prediction : {'classified': {'No Nest': '0.95683'}, 'result': 'success', 'imageMd5': '0acfd6f5b3368d380a67ca9c8d309acd', 'imageUrl': 'http://powerai-vision-portal:9080/powerai-vision-api/uploads/temp/8f80467f-470c-47f3-bf3c-ab7e0880a66b/18c20462-c693-4735-875a-64a30d9974d4.jpg', 'webAPIId': '8f80467f-470c-47f3-bf3c-ab7e0880a66b'}
    #ground_truth     : {'class': 'No Nest', 'id': 'ffbc9c99-201a-4b1a-bce1-a1a1c3ecdc92'}

    # To be compatible with object detection, need to return a list of one item ...
    ytrue = []
    ytrue.append((ground_truth['class']))
    ypred = []
    ypred.append(list(model_predictions.keys())[0])
    return (ytrue, ypred)


def return_ytrue_ypre_objdet(ground_truth, model_predictions) :
    # 1. build a sorted list of labels
    a = ground_truth
    ytrue_labels = [a["label"] for a in ground_truth]
    ypred_labels = [a["label"] for a in model_predictions]

    ytrue_labels = sorted(ytrue_labels)
    ypred_labels = sorted(ypred_labels)
    # 2. smart zipper labels

    ip = 0
    it = 0
    ytrue = []
    ypred = []
    # yaesh
    while( not( it >= len(ytrue_labels) and ip >= len(ypred_labels))) :

        ytl =  "zzzzzzzzz_null" if it >= len(ytrue_labels) else  ytrue_labels[it]
        ypl =  "zzzzzzzzz_null" if ip >= len(ypred_labels) else  ypred_labels[ip]

        if(ytl == ypl) :
            ytrue.append(ytl)
            ypred.append(ypl)
            it += 1
            ip += 1
        elif(ytl != ypl and ytl <= ypl) :
            ytrue.append(ytl)
            ypred.append("null")
            it += 1
        elif(ytl != ypl and ypl < ytl) :
            ytrue.append("null")
            ypred.append(ypl)
            ip += 1

    print("ytrue = {}".format(ytrue))
    print("ypred = {}".format(ypred))
    return (ytrue, ypred)




def fetch_scores(paiv_url, validate_mode="classification", media_mode="video", num_threads=2, frame_limit=50, sample_rate=10, image_dir="na", video_fn="na", paiv_results_file="fetch_scores.json"):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # This consumer function yanks Frames off the queue and stores result in json list ...
    def consume_frames(q,result_dict,thread_id):
        fetch_fn = "paiv_{}.jpg".format(thread_id)
        while (q.qsize() > 0):

            (frame_key, frame_id, frame_np) = q.get()
            if(q.qsize() % 1 == 0) :
                print("Thr {} : Size of queue = {}".format(thread_id, q.qsize()))
                #print("Thr {} : Hash key = {}".format(thread_id, frame_key))
                #print("Thr {} : Frame id = {}".format(thread_id, frame_id))

            json_rv = get_json_from_paiv(paiv_url, frame_np, fetch_fn, thread_id )
            result_dict[frame_key] = json_rv
            print("Thr {} : Task complete".format(thread_id))
            q.task_done()

    q = Queue(maxsize=0)
    result_json_hash = {}
    cap = None

    if(media_mode == "video") :

        frame_limit = int(frame_limit)
        cap  = cv2.VideoCapture(video_fn)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) # fps = video.get(cv2.CAP_PROP_FPS)
        secs = total_frames / fps
        nprint("Total number of frames  = {} (frames)".format(total_frames))
        nprint("Frame rate              = {} (fps)".format(fps))
        nprint("Total seconds for video = {} (s)".format(secs))

        if(frame_limit > total_frames) :
            frame_limit = int(total_frames)
        framecnt = 1 # equals one b/c I read a frame ...
        nprint("Total Frames to annotate= {}".format(int(frame_limit/sample_rate)))
        result_json_hash = [None] * (int(frame_limit/sample_rate))


        # load num_threads images into queue with framecnt as index
        # Load Frames into Queue here.. then release the hounds
        # This is a serialized producer ....
        nprint("Serially loading queue for multi-threaded api access.  Num frames = {}".format(frame_limit))
        annotatecnt = 0
        while(framecnt < frame_limit):
            # Load
            for i in range(frame_limit) :
                if(framecnt % 500 == 0 ):
                    nprint("Loaded {} frames".format(framecnt))
                ret, frame = cap.read()

                # Only load queue if matching the frame stride ...
                if(framecnt%sample_rate == 0) :
                    q.put((annotatecnt,framecnt,frame))
                    annotatecnt+=1
                framecnt += 1

    elif(media_mode == "image") :
        # load in numpy array into Q !!
        if(image_dir == "na") :
            nprint("ERROR : need to specify image_dir=<image directory> in function call")
            return "error"
        nprint("Loading Dataset to prepare for inferencing ")
        paiv_data = _load_paiv_dataset(image_dir, validate_mode)
        #result_json_hash = [None] * len(paiv_data)

        #load the images here!
        idx = 0
        for image_hash_key in paiv_data.keys() :
            image_id = paiv_data[image_hash_key]['id']
            image_file = get_image_fn(image_id)

            npary = cv2.imread(image_file)
            if(npary.any() == None) :
                nprint("Error loading {}.  Unsupported file extension\nexiting ....".format(image_file))
                return 1;

            mykey =get_np_hash(npary)

            q.put((mykey,npary))
            #print(idx)
            idx += 1


    # Setup Consumers.  They will fetch frame json info from api, and stick it in results list
    nprint("Consuming all frames in queue.  Numthreads = {}".format(num_threads))

    threads = [None] * num_threads
    for i in range(len(threads)):
        nprint("spawning thread {}".format(i))
        threads[i] = Thread(target=consume_frames, args=(q, result_json_hash, i))
        threads[i].start()

    # Block until all consumers are finished ....
    nprint("Waiting for all consumers to complete ....")
    for i in range(len(threads)):
        threads[i].join()
    #nprint("Total number of frames processed : {} ".format(fram))

    if(media_mode == "video") :
        cap.release()

    nprint("Writing json data to {}".format(paiv_results_file))
    f = open(paiv_results_file, 'w')
    f.write(json.dumps(result_json_hash))
    f.close()

############################################################################################################
# PAIV API Funcs
############################################################################################################


def get_json_from_paiv(endpoint, img, temporary_fn , thr_id=0):
    json_rv = None

    tstamp = "thr_id:{}-{}".format(thr_id,temporary_fn )
    if(endpoint != None ) :
        headers = {
            'authorization': "Basic ZXNlbnRpcmVcYdddddddddddddd",
            'cache-control': "no-cache",
        }
        # Not very performant, but the API is forcing me to do this!
        cv2.imwrite(temporary_fn, img)
        files_like1 = open(temporary_fn,'rb')
        files1={'files': files_like1}

        #nprint("endpoint {}".format(endpoint))
        #nprint("files1 {}".format(files1))
        nprint("{} : sending post".format(tstamp))

        retry = True
        retry_count = 0

        while(retry == True and retry_count < 3) :
            try :
                resp = requests.post(url=endpoint, verify=False, files=files1, headers=headers, timeout=5)  # files=files
                nprint("{} : rcv post".format(tstamp))
                json_rv = json.loads(resp.text)
                if(resp.status_code == 200) :
                    retry = False
                else :
                    nprint("{} Bad Status code , retry ...".format(tstamp))
                    retry_count +=1
            except(requests.exceptions.Timeout):
                retry_count +=1
                nprint("{} Timeout, retry (inc count) {} ...".format(tstamp, retry_count))

        if(json_rv == None) :
            nprint("{} API failure : did not retrieve data".format(tstamp))
            json_rv = {'empty_url' : 'fetch failed'}

        #print(json.loads(resp.text))
    else :
        json_rv = {'empty_url' : ''}

    #nprint("Returning data : {}".format(json_rv))

    return json_rv


############################################################################################################
# Video Funcs
############################################################################################################
def split_video(input_video, output_directory, max_frames=4, force_refresh=True, sample_rate=1) :

    cap = cv2.VideoCapture(input_video)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) # fps = video.get(cv2.CAP_PROP_FPS)
    secs = total_frames / fps
    print("Total number of frames  = {} (frames)".format(total_frames))
    print("Frame rate              = {} (fps)".format(fps))
    print("Total seconds for video = {} (s)".format(secs))
 
    loopcnt = 0
    frames_written = 0
    os.system("mkdir -p {}".format(output_directory))
    while(loopcnt < total_frames):
        ret, frame = cap.read()
        # Frame skipping .....
        if(loopcnt % sample_rate == 0 and frames_written < max_frames) :
            output_fn = output_directory + '/' + input_video.split('/')[-1] 
            output_fn = output_fn.replace(".mov", ".png")
            output_fn = output_fn.replace(".mp4", ".png")
            output_fn = output_fn.replace(".png", "_{}_.png".format(loopcnt))


            nprint(output_fn)
            # Hand labeled set
            #pdb.set_trace()
            cv2.imwrite( output_fn, frame )
            frames_written +=1
        loopcnt +=1

    nprint("Complete.  Wrote {} frames to {}".format(frames_written,output_directory))


def edit_video(input_video, model_url,output_directory, output_fn, max_frames=50, force_refresh=True, sample_rate=1):
    paiv_colors = generate_colors()

    if(not(os.path.isfile(input_video))) :
        nprint("Error : Input File {} does not exist.  Check path".format(input_video))
        return 1;
    print(input_video)


    loopcnt = 1 # loopcnt set to one since we read the first frame

    counter_dict = defaultdict(int)

    metric_dict = defaultdict(int)
    color_dict = defaultdict()

    # Make
    if(not(os.path.exists(output_directory))) :
        #shutil.rmtree(output_directory)
        os.mkdir(output_directory)


    cache_file = output_directory + "/cache.json"
    # Step1 : Determine if i need to fetch data or if cache file exists ...
    # if it doesnt exist, build it ....

    if(not os.path.isfile(cache_file) or force_refresh==True) :
        nprint("Fetching scores from PAIV url = {}, output dir = {} ".format(model_url, output_directory))
        fetch_scores(model_url, 'object', media_mode='video',num_threads=6,frame_limit=max_frames, sample_rate=sample_rate,image_dir="na", video_fn=input_video, paiv_results_file=cache_file)
        #def fetch_scores(paiv_url, validate_mode="classification", media_mode="video", num_threads=2, frame_limit=50, image_dir="na", video_fn="na", paiv_results_file="fetch_scores.json"):

    box_cache_json =open(cache_file).read()
    box_cache_list = json.loads(box_cache_json)

    nprint("Read in {} json records".format(len(box_cache_list)))

    # Second Pass over video
    nprint("Annotating {} and saving in {}".format(input_video, output_directory))
    cap  = cv2.VideoCapture(input_video)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) # fps = video.get(cv2.CAP_PROP_FPS)
    secs = total_frames / fps
    nprint("Total number of frames  = {} (frames)".format(total_frames))
    nprint("Frame rate              = {} (fps)".format(fps))
    nprint("Total seconds for video = {} (s)".format(secs))

    if(max_frames > total_frames) :
        max_frames = total_frames
        print("Processing number of frames  = {} (frames)".format(max_frames))
    ret, frame = cap.read()

    output  = cv2.VideoWriter(output_directory + "/" + output_fn, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1],frame.shape[0]), True)

    # Used to properly index into json list
    sample_rate_idx = 0
    while(loopcnt < max_frames ):
        ret, frame = cap.read()


        # Frame striding .....
        if(loopcnt % sample_rate == 0) :
            # plot_image( frame )

            # If use_cache is true and I have my cache.json file, then just use previous labels !!
            json_rv = box_cache_list[sample_rate_idx]
            boxes = get_boxes_from_json(json_rv)
            metric_dict = update_metrics(boxes, 1, metric_dict)

            for box in boxes :
                color_dict[box.label] = paiv_colors[int(hashlib.md5(box.label.encode('utf-8')).hexdigest(), 16 ) % 6]
                color = paiv_colors[int(hashlib.md5(box.label.encode('utf-8')).hexdigest(), 16 ) % 6]
                frame = draw_annotated_box(frame, box, color)
                #framep = paiv.draw_annotated_box(framep, box, color)

            frame = draw_counter_box(frame,'Running Counts', metric_dict, color_dict)
            output.write(frame)
            sample_rate_idx += 1

        loopcnt += 1

        if(loopcnt % sample_rate == 0 ) :
            nprint("Complete {} frames".format(loopcnt))

    cap.release()
    output.release()
    nprint("Program Complete : Wrote new movie : {}/{}".format(output_directory,output_fn))


# Custom Logic for this soccer video
# Keep track of raw box counts
# Also track
# - current opppent players (smoothed using ewma)
# - current csa players (smoothed using ewma)
# - ball touches with timer

def update_metrics(boxes, frame_count, metric_dict):
    #Embed some better logic here in the future

    for box in boxes :
        # Running counts
        metric_dict[box.label]+=1

    return metric_dict


def generate_colors(class_names=['a','b','c','d','a1','b1','c1','d1','a2','b2','c2','d2']):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors



def get_boxes_from_json(json_in) :
    '''
    Function to parse boxes in json form and read into a standard box class so that I can manipulate boxes ...
    :param json_in: 
    :return: list of boxes
    '''
    #print(json_in)
    rv_box_list = []
    if(json_in != None) :
        try :
            box_list = json_in['classified']
            for box in box_list :
                tmpbox = Box(box['label'],box['xmin'],box['ymin'],box['xmax'],box['ymax'],box['confidence'])
                rv_box_list.append(tmpbox)

        except KeyError :
            nprint("No Json available")
    else :
        nprint("No Json available")

    return rv_box_list

# This Function draws a nice looking bounding box ...
def draw_annotated_dot(img, box, color_bgr) :
    cv2.circle(img, box.center(), 6,  color_bgr, thickness=-1, lineType=8, shift=0)
    return img

def draw_annotated_box(img, box, color_bgr, mode="all") :
    cv2.rectangle(img, box.ulc(), box.lrc(), color_bgr, 1 )

    cv2.rectangle(img, box.ulc(yoff=-20), box.urc(), color_bgr, -1 ,)

    cv2.circle(img, box.center(), 6,  color_bgr, thickness=-1, lineType=8, shift=0)

    # Add a better looking font
    # cv2 use bgr, pil uses rgb!
    modified_img = img
    ft = cv2.FONT_HERSHEY_COMPLEX_SMALL
    COLOR_BLACK=(0,0,0)   # Make the text of the labels and the title white

    sz = 0.35
    # Draw Header ...
    txt_y_off = 30
    cv2.putText(img, box.label, box.ulc(yoff=-10,xoff=4), ft, sz, COLOR_BLACK, 1, cv2.LINE_AA)

    return modified_img





# This Function will parse a counter dictionary and draw a nice box in upper left hand corner
# mode = ["counts" | "screen_time"]
def draw_counter_box(img, counter_title, counter_dict, color_dict, mode="counts" ) :
    # This is the location on the screen where the ad times will go - if you want to move it to the right increase the AD_START_X
    num_counters = len(counter_dict)
    # Start at (25,25) for ulc, and scale accordingly for counters ....
    box_length = 260
    overlay_box = Box('none', 25,25,25+box_length, 100+num_counters*25,1.0)

    AD_BOX_COLOR=(180,160,160)  # Make the ad timing box grey
    COLOR_WHITE=(255,255,255)   # Make the text of the labels and the title white

    # Make an overlay with the image shaded the way we want it...
    overlay = img.copy()

    # Shade Counter Box
    cv2.rectangle(overlay, overlay_box.ulc(sf=1.0), overlay_box.lrc(sf=1.0), AD_BOX_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    ft = cv2.FONT_HERSHEY_SIMPLEX
    sz = 0.7
    # Draw Header ...
    txt_y_off = 30
    cv2.putText(img, counter_title, overlay_box.ulc(sf=1.0,xoff=10,yoff=txt_y_off), ft, sz, COLOR_WHITE,2,cv2.LINE_AA)

    #Draw Counters
    i=1
    sz = 0.6
    for (k,v ) in sorted(counter_dict.items()):
        col = color_dict[k] if k in color_dict else (255,255,255)
        txt = "{} : {}".format(k, counter_dict[k])
        cv2.putText(img, txt, overlay_box.ulc(sf=1.0,xoff=10,yoff=txt_y_off+25*i), ft, sz, color_dict[k],2,cv2.LINE_AA)
        i += 1

    return img


# This Function will parse a counter dictionary and draw a nice box in upper left hand corner
def draw_text_box(img, box_title, box_text_list ) :
    # This is the location on the screen where the ad times will go - if you want to move it to the right increase the AD_START_X
    
    # Start at (25,25) for ulc, and scale accordingly for counters ....
    box_length = 400
    # 25 pixels per text row ..
    overlay_box = Box('none', 25,25,25+box_length, 100+len(box_text_list)*25,1.0)

    AD_BOX_COLOR=(180,160,160)  # Make the ad timing box grey
    COLOR_WHITE=(255,255,255)   # Make the text of the labels and the title white

    # Make an overlay with the image shaded the way we want it...
    overlay = img.copy()

    # Shade Counter Box
    cv2.rectangle(overlay, overlay_box.ulc(sf=1.0), overlay_box.lrc(sf=1.0), AD_BOX_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    ft = cv2.FONT_HERSHEY_SIMPLEX
    sz = 0.6
    # Draw Header ...
    txt_y_off = 30
    cv2.putText(img, box_title, overlay_box.ulc(sf=1.0,xoff=10,yoff=txt_y_off), ft, sz, COLOR_WHITE,2,cv2.LINE_AA)

    #Draw Counters
    i=1
    sz = 0.6
    
    col = (255,255,255)
    for i in range(len(box_text_list)) :
        row = i + 1
        box_text = box_text_list[i]
        cv2.putText(img, box_text, overlay_box.ulc(sf=1.0,xoff=10,yoff=txt_y_off+25*row), ft, sz, col ,2,cv2.LINE_AA)
        

    return img

# This Function will parse a counter dictionary and draw a nice box in upper left hand corner
def add_image_thumbnail(img, img_thumbnail, overlay_dimensions_xy=(400,200), ) :
    
    COLOR_WHITE=(255,255,255)   # Make the text of the labels and the title white
    COLOR_BLACK=(0,0,0)
    
    
    #img_thumbnail = "dvlk//infer/unknown/vader_slice10-1-of-10.png"
    # Make an overlay with the image shaded the way we want it...
    overlay =cv2.imread(img_thumbnail) # cv2.IMREAD_UNCHANGED
    (y,x,c) = img.shape
    overlay1 = cv2.resize(overlay,overlay_dimensions_xy)
    (yo,xo,co) =overlay1.shape
    

    # Use this Box for positioning logic ...    
    #xmin = 25
    #ymin = y - overlay_dimensions_xy[1]-25
    #pos_box = Box('none', xmin,ymin,xmin+overlay_dimensions_xy[0], ymin+overlay_dimensions_xy[1],1.0)
    xpad = 25
    ypad = 25
    #                                     top, bottom, left, right
    overlay2= cv2.copyMakeBorder(overlay1,y-yo-ypad,ypad,xpad,x-xo-xpad,cv2.BORDER_CONSTANT,value=COLOR_BLACK)

    # Make a mask the same size as the overlay box .. 100x100
    nprint("o shape {},o1 shape {},o2 shape {}, img shape {}".format(overlay.shape,overlay1.shape,overlay2.shape,img.shape))

    #mask = np.full(overlay_dimensions, 0, dtype=np.uint8)

    # Shade Counter Box
    #cv2.addWeighted(overlay2, 0.7, img, 0.3, 0, img)
    # zero out part of image where png will reside ..
    # ones_mask = np.full((img.shape[0], img.shape[1]), 1, dtype=np.uint8)
    img[y-yo-ypad:y+ypad,xpad:xo+xpad,:] = 0 
    img = cv2.bitwise_or(img, overlay2)

    return img

class Box():
    '''
    data structure to hold box data
    '''
    def __init__(self,label,xmin,ymin,xmax,ymax,confidence):
        self.label = label
        self.xmin = xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax = ymax
        self.confidence = confidence

    def center(self) :
        return (int((self.xmin+self.xmax)/2.0),int((self.ymin+self.ymax)/2.0))

    def ul(self):
        return (self.xmin,self.ymin)

    def lr(self):
        return (self.xmax,self.ymax)

    def ur(self):
        return (self.xmax,self.ymin)

    def ll(self):
        return (self.xmin,self.ymax)

    def ulc(self, sf=0.5, xoff=0,yoff=0):
        #scaled upper left box
        cent = self.center()
        ul =  self.ul()
        xulc = int(sf*(ul[0]-cent[0])+cent[0]) + xoff
        yulc = int(sf*(ul[1]-cent[1])+cent[1]) + yoff
        return(xulc,yulc)

    def lrc(self, sf=0.5, xoff=0,yoff=0):
        #scaled upper left box
        cent = self.center()
        lr =  self.lr()
        xlrc = int(sf*(lr[0]-cent[0])+cent[0]) + xoff
        ylrc = int(sf*(lr[1]-cent[1])+cent[1]) + yoff
        return(xlrc,ylrc)

    def urc(self, sf=0.5, xoff=0,yoff=0):
        #scaled upper left box
        cent = self.center()
        ur =  self.ur()
        xurc = int(sf*(ur[0]-cent[0])+cent[0]) + xoff
        yurc = int(sf*(ur[1]-cent[1])+cent[1]) + yoff
        return(xurc,yurc)


    def scale(self,wratio, hratio,offset_px):
        '''
        This function returns 480,640 boxes back to the original image scale
        :param ratio: 
        :param offset_px: 
        :return: a new box, with scale [xy] min/maxes
        '''
        import copy
        newbox = copy.copy(self)
        newbox.xmin = int(newbox.xmin * wratio )
        newbox.xmax = int(newbox.xmax * wratio )
        newbox.ymin = int(newbox.ymin * hratio + offset_px)
        newbox.ymax = int(newbox.ymax * hratio + offset_px)
        return newbox






        # SLOW SLOW SLOW PILLOW IMPLEMENTATION
        #if(mode == "use_pillow") :
        #    cv2_im_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #    pil_im = Image.fromarray(cv2_im_rgb)
        #    draw = ImageDraw.Draw(pil_im)
        #    # use a truetype font
        #    font = ImageFont.truetype("verdanab.ttf", 10)
        #
        #    # Draw the text
        #    color_rgbf = (color_bgr[2],color_bgr[1],color_bgr[0],0)
        #    color_rgbf_black = (0,0,0,0)
        #
        #    draw.text(box.ulc(yoff=-17,xoff=4), box.label, font=font, fill=color_rgbf_black)
        #    #cv2.putText(img,box.label, box.ulc(), font, 0.5, color,1,cv2.LINE_AA)
        #
        #    # Get back the image to OpenCV
        #    modified_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        #elif(mode == "all") :