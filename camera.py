import cv2
import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        self.parser.add_argument('--camera', type=int, default=0)

        self.parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        self.parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

        self.parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        self.parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
        self.parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
        self.args = self.parser.parse_args()

        logger.debug('initialization %s : %s' % (self.args.model, get_graph_path(self.args.model)))
        self.w, self.h = model_wh(self.args.resize)
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(self.args.model), target_size=(self.w, self.h))
        else:
            self.e = TfPoseEstimator(get_graph_path(self.args.model), target_size=(432, 368))
        logger.debug('cam read+')
        filename = 'static/video.avi'
        frames_per_second = 6.0
        res = '720p'
        VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        #'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'MP4V'),
        }
        STD_DIMENSIONS =  {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
        }
        
   
        width, height = STD_DIMENSIONS["480p"]
        if res in STD_DIMENSIONS:
            width,height = STD_DIMENSIONS[res]

        self.video = cv2.VideoCapture(0)
        self.video.set(3, width)
        self.video.set(4, height)
     
        self.out = cv2.VideoWriter(filename, VIDEO_TYPE['avi'], 25, (width, height))

    
    def __del__(self):
        self.video.release()
        self.out.release()
    
    def get_frame(self):
        logger.debug('+image processing+')
        ret_val, image = self.video.read()
        positions = ['sitting', 'upright', 'walking']

        logger.debug('+postprocessing+')
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=self.args.resize_out_ratio)
        img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        logger.debug('+classification+')
        # Getting only the skeletal structure (with white background) of the actual image
        image = np.zeros(image.shape,dtype=np.uint8)
        image.fill(255) 
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # Classification
        pose_class = label_img.classify(image)
        
        logger.debug('+displaying+')
        cv2.putText(img,
                    "Current predicted pose is : %s" %(pose_class),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        
        self.out.write(img)
        ret, jpeg = cv2.imencode('.jpg',img )
        return jpeg.tobytes()
