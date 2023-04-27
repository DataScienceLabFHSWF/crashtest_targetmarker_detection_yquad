"""
This class is used to deal with the targetmarkers. It is designed to handle its own status through methods like update(), confirm() etc
so that it can handle itself instead of always having to keep track on it in the main program. It also offers the classificiation by a Bayesian Neural 
Network and adds filter functionality as well as the center point detection for DOT and MXT markers that was talked about in the bachelor thesis.

v3: removed check for reconfirmation (last_confirm_delta_thresh) in tracker/frame updates due to few false positives in YQUAD testdata (only for YQUAD project)

@autor Daniel Gierse
"""

import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore')

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import helpertools as htools
import itertools
from skimage import measure
import logging



logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

"""
targetmarker_class_id:
 0: YQUAD

 bbox format:   tf: [ymin,xmin,ymax,xmax] relative (0-1)
                cv2:[xmin,ymin,width,height] absolute
"""
class Targetmarker:
    newest_id = itertools.count()
    
    mxt_list = []
    marker_list = []
    yquad_median_area = 0
    min_area_threshold = 0
    max_area_threshold = 100000
    min_area_threshold_factor = 0.6
    max_area_threshold_factor = 1.41
    is_min_area_threshold_set = False
    is_max_area_threshold_set = False
    
    @classmethod
    def on_marker_append(cls,marker_instance):
        cls.marker_list.append(marker_instance)
        # # print("on marker append -  len(marker_list): {}".format(len(cls.marker_list)))
        
    @classmethod
    def on_frame_end(cls):
        cls.yquad_median_area = np.median([marker.bboxes_cv2[-1][2]*marker.bboxes_cv2[-1][3] for marker in cls.marker_list if marker.is_active])

        if math.isnan(cls.yquad_median_area):
            cls.yquad_median_area = 0
        else:
            cls.is_min_area_threshold_set = True
            cls.is_max_area_threshold_set = True

        cls.min_area_threshold = int(cls.yquad_median_area*cls.min_area_threshold_factor)
        cls.max_area_threshold = int(cls.yquad_median_area*cls.max_area_threshold_factor)
        
    @classmethod
    def reset(cls):
        cls.newest_id = itertools.count()
        cls.marker_list.clear()
        cls.min_area_threshold = 0
        cls.max_area_threshold = 100000
        cls.is_min_area_threshold_set = False
        cls.is_max_area_threshold_set = False
        cls.yquad_median_area = 0
        
    
    def __init__(self,frame,frame_id,bbox,target_region_cv2,bbox_format="tf",targetmarker_class_id=1,confirm_delta_thresh=15,video_id=1,initial_detection_score=0,bbox_resize_factor=1):
        
        Targetmarker.on_marker_append(self)#.marker_list.append(self)
        new_id = next(Targetmarker.newest_id)
        self.marker_id = new_id
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.bboxes_cv2 = []
        self.centerpoints_abs = [] #[x,y]
        self.centerpoints_rel = [] 
        self.bnn_predictions = []
        self.frame_ids = []
        self.confirmations = 0
        self.confirmations_thresh = confirm_delta_thresh
        self.target_region_bboxcv2 = target_region_cv2

        self.bbox_resize_factor = bbox_resize_factor
        
        self.start_frame_id = frame_id
        self.video_id = video_id
        
        if bbox_format == "tf":
            self.bboxes_cv2.append(htools.convertCoordsTFtoCV2abs(bbox,self.frame_width,self.frame_height))
        else: # cv2abs
            self.bboxes_cv2.append(bbox)
        
        self.tm_format = targetmarker_class_id
        
        self.last_confirm_delta = 0
        self.last_confirm_delta_thresh = confirm_delta_thresh
        self.confirmed_for_frame = True
        self.is_active = True

        self.has_tracker_set = False
        
        self.detection_scores = [initial_detection_score]  
              
        self.targetmarker_class_id = targetmarker_class_id

        self.__create_tracker(frame)
        self.__update_centerpoint()
        self.__update_targetmarker_label()
      
    #
    #
    # updates the # printlabel variable which is used for the on-frame drawings:
    #
    # 1: YQUAD
    def __update_targetmarker_label(self):
        self.printlabel = "yquad"

    #
    # Creates new cv2 tracker object and changes class parameters to represent the current status    
    #       
    def __create_tracker(self,frame):
        if self.check_if_in_region(self.bboxes_cv2[-1]):
            tmp_tracker = cv2.TrackerCSRT_create()  # für openCV > 4.5.5

            succ = tmp_tracker.init(frame,tuple(self.bboxes_cv2[-1]))
            if succ:
                self.tracker = tmp_tracker
                self.has_tracker_set = True
                self.is_active = True
            else:
                self.has_tracker_set = False
                self.is_active = False
        else:
            self.has_tracker_set = False
            self.is_active = False
            
    #
    # Checks whether or not the marker lies inside target region        
    def check_if_in_region(self,tm_bboxcv2):
        bbox_in_target_region = False
        if tm_bboxcv2[0] > self.target_region_bboxcv2[0] and tm_bboxcv2[1] > self.target_region_bboxcv2[1] and (tm_bboxcv2[0] + tm_bboxcv2[2]) < (self.target_region_bboxcv2[0] + self.target_region_bboxcv2[2]) and (tm_bboxcv2[1] + tm_bboxcv2[3]) < (self.target_region_bboxcv2[1] + self.target_region_bboxcv2[3]):
            bbox_in_target_region = True
        
        return bbox_in_target_region
    
    #
    # called on the begin of each frame
    # manages the "update cycle" of the marker instance by updating the tracker, checking for the region as well as calling the 
    # prediction and center point methods.
    # On fail the marker gets deactived and/or deleted
    def update(self,frame_normalized,frame_id):
        if self.is_active:
            succ, bbox = self.tracker.update(frame_normalized)
            bbox = np.array(bbox).clip(min=0).astype(np.int)   # tracker sometimes returns negative results (-0.0x) which lead to errors in marker_region assignment
            bbox = tuple([bbox[e] for e in range(len(bbox))])
            
            #
            # check whether or not detected tracker was updated succesfully and then check if marker lies inside target region. 
            if succ:
                succ = self.check_if_in_region(bbox)
           
            #    
            # depending on tracker update and position of marker. On update appends latest bounding box to the BB list for this instance.
            # Then extracts the marker region from the image, resizes it to the shape of the Bayesian Neural Network and calls self.__update_model_pred_mxtdot()
            # in order to update the markers prediction whenever it is not classified as DOT marker. Then it calls the functions to update the center points.

            if succ:
                self.bboxes_cv2.append(bbox)
                marker_region = frame_normalized[int(bbox[1]):int((bbox[1]+bbox[3])), int(bbox[0]):int((bbox[0]+bbox[2])), :]
                if marker_region.size != 0:
                    if self.is_active:
                        self.frame_ids.append(frame_id)
                        self.__update_centerpoint()
                #
                #
                # deactivates and deletes marker whenever area == 0 (maybe bug)
                else:
                    self.is_active = False
                    del self.bboxes_cv2[-len(self.bboxes_cv2):]
            else:
                self.is_active = False
            
                
    def change_bounding_box(self,frame,bboxcv2,score,frame_id,label_id):
        # Updates the bounding box whenever the detection score of the Faster-RCNN is greater than the markers current one (stored in detection_scores).
        # Substitutes latest cv2 BB with the updated one from the RCNN network (tracker update happens "earlier" in the frame than the Faster-RCNN inference)
        # Deletes current tracker and creates new one with the updated coordinates
        # Resets the marker type, as a better fit of the BB should lead to a better prediction of the BNN (and resets decision_counter to allow new prediction process)
        # Reset center point variance since the position inside the BB most likely changed
        # Set ID to 5 ("RECHECKING") in order to visualize the change

        if ((self.__class__.is_min_area_threshold_set and bboxcv2[2]*bboxcv2[3] > self.__class__.min_area_threshold) 
            and ((self.__class__.is_max_area_threshold_set and bboxcv2[2]*bboxcv2[3] < self.__class__.max_area_threshold))):

            if self.has_tracker_set:
                del self.tracker
        
            self.bboxes_cv2[-1] = bboxcv2
            self.detection_scores.append(score)
            
            self.__create_tracker(frame)
            
            self.targetmarker_class_id = label_id

    
    def __find_centerpoints_intersection(self,relative=False):
        if relative:
            return [(self.bboxes_cv2[-1][0]+int(self.bboxes_cv2[-1][2]/2))/self.frame_width,(self.bboxes_cv2[-1][1]+int(self.bboxes_cv2[-1][3]/2))/self.frame_height]
        else:
            return [self.bboxes_cv2[-1][0]+int(self.bboxes_cv2[-1][2]/2),self.bboxes_cv2[-1][1]+int(self.bboxes_cv2[-1][3]/2)]
    
    def __update_centerpoint(self):
        self.centerpoints_abs.append(self.__find_centerpoints_intersection())
        self.centerpoints_rel.append(self.__find_centerpoints_intersection(relative=True))
    
    def check_confirmation(self):
        if self.confirmed_for_frame:
            self.last_confirm_delta = 0
            self.confirmations += 1
        else:
            self.last_confirm_delta += 1
            if self.last_confirm_delta > self.last_confirm_delta_thresh:
                self.is_active = False
                # delete unconfirmed entries if there are too many (tracker likely wrong cause of occlusion or stuff)
                if self.confirmations < self.confirmations_thresh:
                    del self.bboxes_cv2[-(self.last_confirm_delta + 1):] # +1 cause first_frame (pickup frame) always validated, but wrong in this case 
                    del self.centerpoints_abs[-(self.last_confirm_delta + 1):]

        self.confirmed_for_frame = False
    
    def confirm(self):
        if self.is_active:
            self.confirmed_for_frame = True

    
    def save_to_dataframe(self,destination_folder,video_title):
        xs,ys = map(list,zip(*self.centerpoints_abs))
        xs_rel,ys_rel = map(list,zip(*self.centerpoints_rel))
        column_names = ["frame_id","x_abs","y_abs","x_rel","y_rel"]

        df = pd.DataFrame(list(zip(self.frame_ids,xs,ys,xs_rel,ys_rel)),columns=column_names)
        df.to_csv(os.path.join(join(destination_folder,"csvs"),"{}_marker_{}.csv".format(video_title,self.marker_id)))
            
