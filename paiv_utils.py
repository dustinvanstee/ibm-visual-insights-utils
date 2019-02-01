# paiv_utils.py

# Set of utility functions and classes to help me with PAIV projects
import cv2
import json
import requests
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import colorsys
import random
import sys
from queue import Queue
from threading import Thread
import time


# Function to automatically fetch data from API in threaded mode...
def fetch_scores(video_fn, json_fn, paiv_url, mode="video", num_threads=2, frame_limit=50):
    frame_limit = int(frame_limit)

    # This consumer function yanks Frames off the queue and stores result in json list ...
    def consume_frames(q,result_list,thread_id):
        fetch_fn = "paiv_{}.jpg".format(thread_id)
        while (q.qsize() > 0):
            print("Thr {} : Size of queue = {}".format(thread_id, q.qsize()))
            (frame_id, frame_np) = q.get()
            print("Thr {} : Frame id = {}".format(thread_id, frame_id))

            json_rv = get_json_from_paiv(paiv_url, frame_np, fetch_fn )
            result_list[frame_id] = json_rv
            q.task_done()


    cap  = cv2.VideoCapture(video_fn)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) # fps = video.get(cv2.CAP_PROP_FPS)
    secs = total_frames / fps
    print("Total number of frames  = {} (frames)".format(total_frames))
    print("Frame rate              = {} (fps)".format(fps))
    print("Total seconds for video = {} (s)".format(secs))


    if(frame_limit > total_frames) :
        frame_limit = total_frames
    framecnt = 0 # equals one b/c I read a frame ...
    result_json_list = [None] * int(frame_limit)

    q = Queue(maxsize=0)

    # load num_threads images into queue with framecnt as index


    # Load Frames into Queue here.. then release the hounds
    # This is a serialized producer ....
    while(framecnt < frame_limit):
        # Load
        for i in range(frame_limit) :
            ret, frame = cap.read()
            q.put((framecnt,frame))
            framecnt += 1

    # Setup Consumers.  They will fetch frame json info from api, and stick it in results list
    threads = [None] * num_threads
    for i in range(len(threads)):
        threads[i] = Thread(target=consume_frames, args=(q, result_json_list, i))
        threads[i].start()

    # Block until all consumers are finished ....
    nprint("Waiting for all consumers to complete ....")
    for i in range(len(threads)):
        threads[i].join()
    #nprint("Total number of frames processed : {} ".format(fram))

    cap.release()

    nprint("Writing json data to {}".format(json_fn))
    f = open(json_fn, 'w')
    f.write(json.dumps(result_json_list))
    f.close()



def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))

def generate_colors(class_names=['a','b','c','d','a1','b1','c1','d1','a2','b2','c2','d2']):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def get_json_from_paiv(endpoint, img, temporary_fn ):
    json_rv = None
    if(endpoint != None ) :
        headers = {
            'authorization': "Basic ZXNlbnRpcmVcYdddddddddddddd",
            'cache-control': "no-cache",
        }
        # This code will stay here for reference.  Use this style if loading an image from disk

        cv2.imwrite(temporary_fn, img)
        files_like1 = open(temporary_fn,'rb')
        files1={'files': files_like1}

        # This code attempts to avoids writing the numpy array to disk, and just converts it to a byteIO stream...
        # This code attempts to avoids writing the numpy array to disk, and just converts it to a byteIO stream...

        # file_like = io.BufferedReader(io.BytesIO(img.tobytes()))
        # file_like2 = io.BufferedReader(dvIO(img.tobytes()))
        # files={'files': file_like2 }

        resp = requests.post(url=endpoint, verify=False, files=files1, headers=headers)  # files=files
        json_rv = json.loads(resp.text)

        #print(json.loads(resp.text))
    else :
        json_rv = {'empty_url' : ''}


    return json_rv

def get_boxes_from_json(json_in) :
    '''
    Function to parse boxes in json form and read into a standard box class so that I can manipulate boxes ...
    :param json_in: 
    :return: list of boxes
    '''
    #print(json_in)
    rv_box_list = []

    try :
        box_list = json_in['classified']
        for box in box_list :
            tmpbox = Box(box['label'],box['xmin'],box['ymin'],box['xmax'],box['ymax'],box['confidence'])
            rv_box_list.append(tmpbox)

    except KeyError :
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
def draw_counter_box(img, counter_title, counter_dict, color_dict ) :
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