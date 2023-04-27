"""
process_video
----------------
This script represents the main program loop for the targetmarker detection.
It implements the logic for handling different videos as well as dealing with duplicate detections, the creation of new targetmarkers
and finally the drawing functionalities. 
The script allows for the automatic detection of targetmarkes from type YQUAD.

@author: Daniel Gierse
"""

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import torch
import configparser

from targetmarker import Targetmarker
import helpertools as htools

crop_regions = [[0,500,0,440],[400,900,0,440],[800,1300,0,440],[1200,1700,0,440],[1420,1920,0,440],
           [0,500,340,780],[400,900,340,780],[800,1300,340,780],[1200,1700,340,780],[1420,1920,340,780],
           [0,500,640,1080],[400,900,640,1080],[800,1300,640,1080],[1200,1700,640,1080],[1420,1920,640,1080]]

# # xmin,ymin,width,height
# def trans_to_image_coords(region_bbox,marker_bbox):
#     xnew = region_bbox[0] + marker_bbox[0]
#     ynew = region_bbox[1] + marker_bbox[1]
    
#     return [xnew,ynew,marker_bbox[2],marker_bbox[3]]

# xmin,xmax,ymin,max to xmin,ymin,width,height
def crop_region_to_cv2(crop_region):
    return [crop_region[0],crop_region[2],crop_region[1]-crop_region[0],crop_region[3]-crop_region[2]]

def create_markers(frame,frame_id,video_id,all_regions_markers_list,active_targetmarker_list,res_factor,duplicate_iou_threshold,target_region,tm_confirm_delta_thresh,all_targetmarker_list):
    succ = False
    for (bboxcv2,label_id,detection_score) in all_regions_markers_list:
        is_duplicate = False
        succ = False
      
        for tm in active_targetmarker_list:
            is_duplicate = False                                                                
            if htools.calc_iou_cv2(bboxcv2,tm.bboxes_cv2[-1],duplicate_iou_threshold)[0]:
                if detection_score > tm.detection_scores[-1]: 
                    tm.change_bounding_box(frame,bboxcv2,detection_score,frame_id,label_id)
                    
                tm.confirm()
                
                is_duplicate = True
                break
            
        if is_duplicate:
            continue

        succ = True
        
        temp_marker = Targetmarker(frame, frame_id, bboxcv2, target_region, "cv2", label_id, 
                                   confirm_delta_thresh=tm_confirm_delta_thresh,
                                   video_id=video_id,initial_detection_score=detection_score,
                                   bbox_resize_factor=res_factor)
        
        active_targetmarker_list.append(temp_marker)
        all_targetmarker_list.append(temp_marker)
        
    return succ,active_targetmarker_list,all_targetmarker_list # # # templates_list
        

def get_new_detections_regions(frame_id,target_result_bboxes,detection_scores,label_id,frame_width,frame_height,all_regions_markers_list,crop_region_bbox,duplicate_iou_threshold):
    succ = False
    
    if len(target_result_bboxes) == 0:
        return succ,all_regions_markers_list 

    new_found_markers_bboxes = []
    
    region_bbox = crop_region_to_cv2(crop_region_bbox)
    
    for detection_id,bbox in enumerate(target_result_bboxes):
        is_duplicate = False
        
        detection_score = detection_scores[detection_id]
        bboxcv2 = htools.convert_coords_pytorch_to_cv2(bbox)

        if ((Targetmarker.is_min_area_threshold_set and (bboxcv2[2]*bboxcv2[3] < Targetmarker.min_area_threshold)) 
            and (Targetmarker.is_max_area_threshold_set and (bboxcv2[2]*bboxcv2[3] > Targetmarker.max_area_threshold))):
            continue
        
        #
        # iterates across all active targetmarkers (determined by the tracker.update() function) and checks whether or not a newly detected
        # marker has IoU with already known marker. If thats the case -> duplicate
        #                                           If new detection score > old detection score -> change_bounding_box of marker
        #
        # confirm() marker (for frequency filtering)
        
        
        for i,(marker_bbox,label) in enumerate(new_found_markers_bboxes):
            is_duplicate = False
            if htools.calc_iou_cv2(bboxcv2,marker_bbox,duplicate_iou_threshold)[0]:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
        
        succ = True
        
        new_found_markers_bboxes.append([bboxcv2,label_id])
        all_regions_markers_list.append([htools.trans_to_image_coords(region_bbox,bboxcv2),label_id,detection_score])
    return succ, all_regions_markers_list 

def read_config(file_path:str = "video.cfg"):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    params = {}

    params["INPUT_DIR"] = os.path.join(config["DEFAULT"]["INPUT_VIDEO_DIR"])
    params["OUTPUT_DIR"] = os.path.join(config["DEFAULT"]["OUTPUT_DIR"])
    params["VIDEO_SUFFIXES"] = [c for i,c in enumerate(config["MODEL"]["VIDEO_SUFFIXES"].split("'")) if i % 2 == 1]
    
    params["MODEL_CHECKPOINT_PATH"] = os.path.join(config["MODEL"]["MODEL_CHECKPOINT_PATH"])
    params["MIN_PREDICTION_SCORE"] = float(config["MODEL"]["MIN_PREDICTION_CONFIDENCE"])

    params["TARGETMARKER_MIN_AREA_THRESHOLD_FACTOR"] = float(config["PIPELINE"]["TARGETMARKER_MIN_AREA_THRESHOLD_FACTOR"])
    params["TARGETMARKER_MAX_AREA_THRESHOLD_FACTOR"] = float(config["PIPELINE"]["TARGETMARKER_MAX_AREA_THRESHOLD_FACTOR"])

    params["CUDA_GPU_INDEX"] = int(config["CUDA"]["CUDA_GPU_INDEX"])
    
    return params

def print_infos(params: dict, device: object):
    print("input directory: {}".format(params["INPUT_DIR"]))
    print("output directory: {}".format(params["OUTPUT_DIR"]))
    print("model weights path: {}".format(params["MODEL_CHECKPOINT_PATH"]))
    print("detection threshold: {}".format(params["MIN_PREDICTION_SCORE"]))
    print("model checkpoint: {}".format(params["MODEL_CHECKPOINT_PATH"].split(os.path.sep)[-1]))
    print("CUDA gpu index: {}".format(params["CUDA_GPU_INDEX"]))
    print("video suffixes: {}".format(params['VIDEO_SUFFIXES']))
    print("device: {}".format(device))
    print()

    

# def process_videos(input,output,model,detection_threshold, model_checkpoint_name, model_architecture, cuda_gpu_index):
def process_videos(params: dict):
    htools.make_folders(params["OUTPUT_DIR"],"frames","csvs")
    print()
    
    #
    # parameter for targetmarker class
    tm_confirm_delta_thresh = 10 #20 #35 #55

    yquadlabel = 1
    num_classes = 1
    model_architecture = "yolov5"
    tracker_iou_threshold_yquad = 0.1

    base_width = 1920
    base_height = 1080

    device = torch.device(f'cuda:{params["CUDA_GPU_INDEX"]}') if torch.cuda.is_available() else torch.device('cpu')
    
    marker_detector = htools.YQUADMarkerDetectorYOLOv5(params["MODEL_CHECKPOINT_PATH"], num_classes, device, model_architecture, params["MIN_PREDICTION_SCORE"])
    
    Targetmarker.min_area_threshold_factor = params["TARGETMARKER_MIN_AREA_THRESHOLD_FACTOR"]
    Targetmarker.max_area_threshold_factor = params["TARGETMARKER_MAX_AREA_THRESHOLD_FACTOR"]
    
    vid_ops = htools.VideoOperations()

    video_files = [f for f in listdir(params["INPUT_DIR"]) if isfile(join(params["INPUT_DIR"], f)) and '.DS_Store' not in f]
    
    print_infos(params, device)
    
    print("\nfound {} files: {}\n".format(len(video_files),video_files))
    
    # loop over all found filenames
    for video_id,video_title_ext in enumerate(video_files):
        # put the relative filepath back together
        filename_in = join(params["INPUT_DIR"],video_title_ext)

        suffix = None
        
        # check whether suffix/datatype is known
        for suf in params["VIDEO_SUFFIXES"]:
            if suf in video_title_ext:
                video_title_wo_suffix = video_title_ext.rstrip(suf)
                suffix = suf
                break
    
        if suffix is None:
            print("Unknown suffix, skipping file")
            continue

        #
        #
        # load video at frame 0 and get different parameters like width and height of frame
        if vid_ops.load_video(filename_in):
            frame_id = 0
            vid_len = vid_ops.length_in_frames
            tm_confirm_delta_thresh = int(vid_len/6)

            frame_width = vid_ops.frame_width
            frame_height = vid_ops.frame_height
            
            res_factor_width = frame_width/base_width
            res_factor_height = frame_height/base_height
            res_factor = (res_factor_width,res_factor_height)
            
            target_region = [0, 0, frame_width, frame_height]
            
            print("-----------------------------------")
            print("file: {}".format(video_title_wo_suffix))  
            print("resolution: {}x{}".format(frame_width,frame_height))
            print("length: {} frames".format(vid_len))
            print()
            
            active_targetmarker_list = []
            all_targetmarker_list = []

            #
            #
            # get first frame in video
            get_frame_succ, current_frame, current_frame_normalized = vid_ops.get_frame()
            
            writer = cv2.VideoWriter(join(params["OUTPUT_DIR"], "{}_out{}".format(video_title_wo_suffix,suffix)), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
            
            #
            #
            # processing loop for each video. First the sliced target region is calculated, then scalereader is used on this sclice to 
            # find scale segments and calculate cm/pixel value. Afterwards every active targetmarker updates its tracker and get reaffirmed
            # on that list.
            # Then targetmarkers get detected by marker_detector and returned
            while get_frame_succ:
                start_time_frame = time.time()
                
                print("frame: {}".format(frame_id))
                current_frame_normalized = cv2.resize(current_frame_normalized,(base_width,base_height))
                current_frame_out = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

                tmp_targetmarker_list = []
                all_regions_markers_list = []   

                for targetmarker in active_targetmarker_list:
                    targetmarker.update(current_frame_normalized,frame_id)
                    if targetmarker.is_active:
                        tmp_targetmarker_list.append(targetmarker)
                        
                active_targetmarker_list = tmp_targetmarker_list

                for crop_region in crop_regions:
                    img_crop = current_frame_normalized[crop_region[2]:crop_region[3],crop_region[0]:crop_region[1]]

                    _, target_result_bboxes_yquad, scores_yquad, _ = marker_detector.detect(img_crop)
                    _, all_regions_markers_list = get_new_detections_regions(frame_id,target_result_bboxes_yquad,scores_yquad,yquadlabel,frame_width,frame_height,all_regions_markers_list,
                                                                             crop_region,tracker_iou_threshold_yquad)

                # sort regions by area
                all_regions_markers_list = sorted(all_regions_markers_list, key=lambda x: x[0][0]*x[0][1])
                _, active_targetmarker_list, all_targetmarker_list = create_markers(current_frame_normalized,frame_id,video_id,all_regions_markers_list,active_targetmarker_list,res_factor,tracker_iou_threshold_yquad,target_region,tm_confirm_delta_thresh,all_targetmarker_list)

                for tm in active_targetmarker_list:
                    current_frame_out = htools.draw_bounding_box(current_frame_out,tm.bboxes_cv2[-1],tm.printlabel,colorcode=1)
                    tm.check_confirmation()
                                                         
                current_frame_out = htools.draw_text(current_frame_out,5,80,400,len(active_targetmarker_list),rows=None,cols=None,section=None)
                
                Targetmarker.on_frame_end()
                print("\nprocessing time: {} s".format(time.time()-start_time_frame))
                print("------")

                writer.write(current_frame_out)
                
                if frame_id % 2 == 0:
                    cv2.imwrite(join(join(params["OUTPUT_DIR"],"frames"),"{}_{}.jpg".format(video_title_wo_suffix,frame_id)), current_frame_out)
                
                get_frame_succ, current_frame, current_frame_normalized = vid_ops.get_frame()
                
                if get_frame_succ:
                    frame_id += 1
            
            #
            # close video writer
            writer.release()

            print()
            print("saving targetmarker data...")
            #
            # save dataframes
            for tm in all_targetmarker_list:
                if len(tm.bboxes_cv2) >= 2:
                    tm.save_to_dataframe(params["OUTPUT_DIR"],video_title_wo_suffix)

            #
            # open new video writer to produce and save filtered output. Reiterates over the original video and draws saved values on frames.
            writer_clean = cv2.VideoWriter(join(params["OUTPUT_DIR"], "{}_out_filtered.{}".format(video_title_wo_suffix,suffix)), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
            
            vid_ops.restart_video()
            frame_id = 0
            get_frame_succ, current_frame, current_frame_normalized = vid_ops.get_frame()
            
            print()
            print("saving filtered video...")
            print()
            while get_frame_succ:
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                current_marker_counter = 0
                for tm in all_targetmarker_list:
                    if tm.targetmarker_class_id == 12:
                        continue
                    if len(tm.bboxes_cv2) <= 11:
                        continue
                    if frame_id >= tm.start_frame_id and (frame_id - tm.start_frame_id) < len(tm.bboxes_cv2):
                        current_frame = htools.draw_bounding_box(current_frame,tm.bboxes_cv2[frame_id - tm.start_frame_id],tm.printlabel,colorcode=tm.targetmarker_class_id)                       
                        current_frame = htools.draw_circle(current_frame,tm.centerpoints_abs[frame_id - tm.start_frame_id][0],tm.centerpoints_abs[frame_id - tm.start_frame_id][1],radius=3,colorcode=tm.targetmarker_class_id)
                        
                        current_marker_counter += 1
                
                current_frame = htools.draw_text(current_frame,5,80,400,current_marker_counter,rows=None,cols=None,section=None)
                
                if frame_id % 10 == 0:
                    cv2.imwrite(join(join(params["OUTPUT_DIR"],"filtered_images"),"{}_{}.jpg".format(video_title_wo_suffix,frame_id)), current_frame)
                    if frame_id % 50 == 0:
                        print("writing frame {}/{}".format(frame_id,vid_ops.length_in_frames))    
                
                writer_clean.write(current_frame)
                get_frame_succ, current_frame, current_frame_normalized = vid_ops.get_frame()
                if get_frame_succ:
                    frame_id += 1

            print("writing frame {}/{}".format(vid_ops.length_in_frames,vid_ops.length_in_frames))
            print()
            print("finished processing {}".format(video_title_ext))
            # close video writer
            writer_clean.release()
            
            Targetmarker.reset()

if __name__ == '__main__':
    params = read_config("video.cfg")
    process_videos(params)