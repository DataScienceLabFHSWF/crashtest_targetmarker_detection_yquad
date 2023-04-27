"""
This code allows for the usage of multiple convenience funtions like the conversion from tf to openCV image handling etc.
It also implements a videoHelper class which helps to keep track of the frames. Additionally it provides the loading and inference
functionalities for various neural networks

@author: Daniel Gierse
"""

# Disable GPU if necessary
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import isfile,isdir,join
import warnings
import logging,sys

import model_lib as models
from PIL import Image

import time

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
warnings.filterwarnings('ignore')


def convert_coords_tf_to_cv2(bboxTF,frame_width,frame_height): #absolute
    return (int(bboxTF[1]*frame_width),int(bboxTF[0]*frame_height),int((bboxTF[3]-bboxTF[1])*frame_width),int((bboxTF[2]-bboxTF[0])*frame_height))
    
def convert_coords_cv2_to_tf(bboxCV2,frame_width,frame_height): #relative
    return (bboxCV2[1]/frame_height,bboxCV2[0]/frame_width,(bboxCV2[3]+bboxCV2[1])/frame_height,(bboxCV2[2]+bboxCV2[0])/frame_width)

# pytorch: xmin, ymin, xmax, ymax  # abs
# cv2: xmin, ymin, width, height # abs
def convert_coords_pytorch_to_cv2(bbox):
    return (bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])

def convert_coords_cv2_to_pytorch(bbox):
    return (bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])

def trans_to_image_coords(region_bbox,marker_bbox):
    xnew = region_bbox[0] + marker_bbox[0]
    ynew = region_bbox[1] + marker_bbox[1]
    
    return [xnew,ynew,marker_bbox[2],marker_bbox[3]]

def __get_color(colorcode):
    if colorcode == 0: 
        color = (255,51,255) # pink
    elif colorcode == 1: 
        color = (51,204,0) # green
    elif colorcode == 2: 
        color = (0,102,204) # orange
    elif colorcode == 3: 
        color = (0,255,255) # yellow
    elif colorcode == 4:
        color = (255,0,255) # lila
    elif colorcode == 10:
        color = (0,255,255) # lila
    elif colorcode == 11:
        color = (255,255,0) # lila
    elif colorcode == 12:
        color = (122,55,255) # lila
    else:
        color = (0,255,96)

    return color
    
def draw_circle(image,x,y,radius=2,colorcode=1,thickness=-1):
    color = __get_color(colorcode)
    image = cv2.circle(image, (int(x),int(y)), radius, color, thickness) 
    return image

def draw_point(image,x,y,colorcode=1):
    color = __get_color(colorcode)

    image[y,x] = color
    return image

def save_image(image,filepath,target_region=[]):
    logging.debug("target_region: {}".format(target_region))
    if len(target_region) == 0:
        cv2.imwrite(join(filepath),image)
    else:
        img = image[target_region[1]:(target_region[1]+target_region[3]-1),target_region[0]:(target_region[0]+target_region[2]-1)]
        cv2.imwrite(join(filepath),img)
        

def draw_bounding_box(image,bbox_coordinates_cv2,label,colorcode=0): # [xmin,ymin,width,height] abs
    color = __get_color(colorcode)

    # logging.debug("draw_bounding_box: label: {}".format(label))    
    # print(f"draw_bounding_box: bbox_coordinates_cv2: {bbox_coordinates_cv2}")
    p1 = (int(bbox_coordinates_cv2[0]), int(bbox_coordinates_cv2[1]))
    p2 = (int(bbox_coordinates_cv2[0] + bbox_coordinates_cv2[2]), int(bbox_coordinates_cv2[1] + bbox_coordinates_cv2[3]))
    cv2.rectangle(image, p1, p2, color, 2, 1)
    cv2.putText(image,label,(int(p1[0]),int(p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image
    
def draw_bounding_box_score(image,bbox_coordinates_cv2,label,score=0,colorcode=0): # [xmin,ymin,width,height] abs
    color = __get_color(colorcode)  

    p1 = (int(bbox_coordinates_cv2[0]), int(bbox_coordinates_cv2[1]))
    p2 = (int(bbox_coordinates_cv2[0] + bbox_coordinates_cv2[2]), int(bbox_coordinates_cv2[1] + bbox_coordinates_cv2[3]))
    cv2.rectangle(image, p1, p2, color, 2, 1)
    cv2.putText(image,"{}: {}".format(label,score),(int(p1[0]),int(p1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return image

def draw_text(img,class_id,x,y,info_data,rows=None,cols=None,section=None):
    if rows is None and cols is None: # absolute coordinates given
        x = x
        y = y
    else: # relative coordinates given, need to multiply by factor to get absolute coordinates for drawing box
        x = x * cols
        y = y * rows

    if class_id == 4:
        cv2.putText(img, '{:2f} cm/pixel'.format(info_data), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), lineType=cv2.LINE_AA)

    elif class_id == 5:
        cv2.putText(img, '{} targetmarker currently active'.format(info_data), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), lineType=cv2.LINE_AA)

    else:
        pass

    return img
      
def calc_iou_tf(box1, box2, iou_thresh):
        ymin_inter = max(box1[0],box2[0])
        xmin_inter = max(box1[1],box2[1])
        ymax_inter = min(box1[2],box2[2])
        xmax_inter = min(box1[3],box2[3])

        box1_area = (box1[3] - box1[1] + 1) * (box1[2] - box1[0] + 1)
        box2_area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] + 1)

        inter_area = max(0,ymax_inter-ymin_inter+1) * max(0,xmax_inter-xmin_inter+1)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area/union_area
        
        if iou > iou_thresh:
            succ = True
        else:
            succ = False
        
        return succ, iou

def calc_iou_cv2(bbox1, bbox2, iou_thresh=0.4):  # expects [xmin,ymin,width,height] abs, converts to [ymin,xmin,ymax,xmax] abs
    box1 = [bbox1[1],bbox1[0],bbox1[1]+bbox1[3],bbox1[0]+bbox1[2]]
    box2 = [bbox2[1],bbox2[0],bbox2[1]+bbox2[3],bbox2[0]+bbox2[2]]
    
    return calc_iou_tf(box1,box2,iou_thresh)


# args: 1st: root folder (absolute path), rest: direct subfolders to root folder (relative path (= folder name) to root folder)
def make_folders(root_dir:str, *args):
    if root_dir is not None:
        if not os.path.exists(os.path.join(root_dir)):
            os.makedirs(os.path.join(root_dir))
            print(f"Created folder: {os.path.join(root_dir)}")
        for arg in args:
            if not os.path.exists(os.path.join(root_dir,arg)):
                os.makedirs(os.path.join(root_dir,arg))
                print(f"Created folder: {os.path.join(root_dir,arg)}")
        return True
    else:
        return False

"""
keeps track of the video, more convenient than cv2 video implementation for this purpose
"""

class VideoOperations:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.length_in_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.video_loaded = False
        self.writing_video = False
        self.cap = None
    
    def load_video(self,path_to_video):
        self.reset()
        if isfile(join(path_to_video)):
            self.cap = cv2.VideoCapture(path_to_video)
            cap_succ, image = self.cap.read()

            if cap_succ:
                self.video_loaded = True
                self.frame_width = int(self.cap.get(3))
                self.frame_height = int(self.cap.get(4))
                self.length_in_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap_succ = False
        return cap_succ
    
    # eg vid-length = 421 frames => array = [0 ... 420], highest address = 420, highest frame_id = 421
    def get_frame_at_pos(self,frame_id):
        succ = False
        frame = None
        frame_normalized = frame

        if self.video_loaded:
            if frame_id < 1:
                frame_id = 1
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) - 1) # needs to be set to 1 frame before the actual frame you wanna read (read() takes the next frame -> incrementing before evaluating frame)
            succ, frame = self.cap.read()

            if succ:
                frame_normalized = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            else:
                frame_normalized = frame
        
        return succ, frame, frame_normalized
            
    def get_frame(self):
        succ = False
        frame = None
        if self.video_loaded:
            succ, frame = self.cap.read()
            if succ:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_normalized = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            else:
                frame_normalized = frame

        return succ, frame, frame_normalized

    #@profile    
    def get_all_frames(self):
        succ = False
        frames = []
        
        if self.video_loaded:
            self.restart_video()
            read_succ, image = self.cap.read()
            succ = read_succ
            
            while read_succ:
                frames.append(image)
                read_succ, image = self.cap.read()

        return succ, frames
    
    def get_all_frames_sk(path):
        succ = True
        frames = []
        cap = video(path)
        fc = cap.frame_count()
        for i in np.arange(fc):
            frames.append(cap.get_index_frame(i))
        
        return succ, frames
    
    def get_current_frame_pos(self):
        if self.video_loaded:
            return True, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return False, -1
    
    def apply_edge_filter(self,frame):
        if frame is not None:
            return True,cv2.Canny(frame,20,50)
        return False,frame    
                     
    def restart_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    """ 
    check whether or not a scale is present in the video
    """
    def check_scale_in_video(self,scalereader):
        has_scale_in_video = False
        frames_percentages_for_target_region_selection = [10,17,22,25,27,29,40,45,50,55]
        
        target_frames = [self.get_frame_at_pos(int(self.length_in_frames*percentage/100))[1] for percentage in frames_percentages_for_target_region_selection]
        
        outer_match_counter = 0
        for frame in target_frames:
            cpp,cpp_avg,cpp_floating_avg,scale_matches,best_ssim = scalereader.read_scale(frame,False)
            
            inner_match_counter = 0
            has_scale_in_frame = False
            for match in scale_matches:
                if match[2] > 0 and match[3] > 0:
                    inner_match_counter += 1
                    
                    if inner_match_counter == 3:
                        has_scale_in_frame = True
                        break
            
            if has_scale_in_frame:
                outer_match_counter += 1
                
                if outer_match_counter == 3:
                    has_scale_in_video = True
        
        logging.debug("has_scale_in_video: {}".format(has_scale_in_video))
        
        self.restart_video()
        
        return has_scale_in_video
    
    """ 
    try to find a bounding box across the whole video that encapsulates the movement of the car
    """
    def find_target_region_bbox(self,obj_detector=None,min_target_score=0.5):
        succ = False
        target_frames = None
        target_region_bbox = [0,0,1,1]
        target_region_bbox_cv2 = [0,0,self.frame_width,self.frame_height]
        frames_for_target_region_selection = [0.1,10,30,50,90,99]

        if obj_detector is not None:  
            target_frames = [self.get_frame_at_pos(int(self.length_in_frames*percentage/100))[1] for percentage in frames_for_target_region_selection]

            """
            https://stackoverflow.com/questions/42376201/list-comprehension-for-multiple-return-function#42376244
            """
            target_succs, target_bboxes_all = zip(*[obj_detector.detect(frame,min_target_score)[0:2] for frame in target_frames])
            
            target_bboxes = []
            
            # check whether detection worked in the first and one of the last frames (to ensure correct min/max values)
            if target_succs[0] and (target_succs[len(frames_for_target_region_selection)-1] or target_succs[len(frames_for_target_region_selection)-2]):
                for i,bboxes_frame in enumerate(target_bboxes_all):
                    if target_succs[i]:
                        target_bboxes.append(bboxes_frame[0])
            
                min_y = np.amin([bbox[0] for bbox in target_bboxes])
                min_x = np.amin([bbox[1] for bbox in target_bboxes])
                max_y = np.amax([bbox[2] for bbox in target_bboxes])
                max_x = np.amax([bbox[3] for bbox in target_bboxes])
                
                target_region_bbox = [min_y,min_x,max_y,max_x]

                target_region_bbox_cv2 = convert_coords_tf_to_cv2(target_region_bbox, self.frame_width, self.frame_height)
                target_region_bbox = [min_x,min_y,max_x,max_y]
                
                succ = True

        self.restart_video()

        return succ, target_region_bbox, target_region_bbox_cv2
        
    def write_video(self,frames,output_folder,vid_name,suffix,fps,width=None,height=None):
        succ = False
        
        if len(frames) > 0 and isdir(output_folder):
            if width is None:
                width = self.frame_width
            if height is None:
                height = self.frame_height
            
            writer = cv2.VideoWriter(join(output_folder, vid_name + suffix),
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width,height))
            
            for frame in frames:
                writer.write(frame)
            writer.release()
            
            if isfile(join(output_folder, vid_name + suffix)):
                succ = True
        
        return succ


class TargetmarkerDetector():

    def __init__(self, path_to_model_checkpoint, num_classes, device, model_type = "yolov5", detection_threshold = 0.8, normalize_input = True):
        self.path_to_model_checkpoint = path_to_model_checkpoint
        self.device = device
        self.model_type = model_type
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold
        self.normalize = normalize_input

        self.models_dict = {"yolov5": models.create_ultralytics_yolov5,
                            #"yolov8": models.create_ultralytics_yolov8,
                            "faster_rcnn50": models.create_faster_rcnn_resnet50_fpn,
                            "fasterrcnn_resnet50_fpn": models.create_faster_rcnn_resnet50_fpn,
                            "fasterrcnn_resnet50_fpn_v2": models.create_faster_rcnn_resnet50_fpn_v2,
                            "fasterrcnn_resnet50_fpn_custom_anchors": models.create_faster_rcnn_resnet50_fpn_custom_anchors,
                            "fasterrcnn_mobilenet_v3_large_fpn": models.create_faster_rcnn_mobilenet_v3_large_fpn}
        
        self.setup_model()

    def setup_model(self):
        self.build_model()
        self.restore_checkpoint()
        self.model.to(self.device).eval()

    def build_model(self):
        assert self.num_classes is not None
        self.model = self.models_dict[self.model_type](self.num_classes)

    def restore_checkpoint(self):
        assert self.model is not None
        self.model.load_state_dict(models.load_custom_rpn_savestate(self.path_to_model_checkpoint,self.device))
        
    def get_tensor_transform(self, normalize:bool =True) -> T.Compose:
        transforms = []
        transforms.append(T.ToTensor())
        if normalize:
            # imagenet normalization values
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return T.Compose(transforms)

    # Prepare image
    def prepare_image_tensor(self, image: np.array) -> torch.Tensor:
        # convert to tensor, add batch dimension and transfer to self.device
        img = Image.fromarray(image)
        image_tensor = self.get_tensor_transform(self.normalize)(img.copy()).unsqueeze(0).to(self.device)
        return image_tensor

    # Perform detection
    def detect(self, image):
        with torch.no_grad():
            outputs = self.model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        return outputs


class YQUADMarkerDetectorYOLOv8(TargetmarkerDetector):
    def build_model(self):
        assert self.num_classes is not None
        self.model = self.models_dict[self.model_type](self.num_classes, checkpoint_path=self.path_to_model_checkpoint)

    def setup_model(self):
        self.build_model()
        self.xai_supported = False

    def detect(self, image):
        image = Image.fromarray(image)
        results = self.model(image)

        selected_boxes_yquad = []
        selected_scores_yquad = []
        selected_labels_yquad = []

        for res in results[0].boxes:            
            score = res.conf.cpu().numpy()[0]
            print(f"score: {score}")
            if score < self.detection_threshold:
                break
            xmin = res.xyxy.data.cpu().numpy()[0][0]
            ymin = res.xyxy.data.cpu().numpy()[0][1]
            xmax = res.xyxy.data.cpu().numpy()[0][2]
            ymax = res.xyxy.data.cpu().numpy()[0][3]

            selected_boxes_yquad.append([int(xmin),int(ymin),int(xmax),int(ymax)])
            selected_scores_yquad.append(score)
            selected_labels_yquad.append(0)

        if len(selected_boxes_yquad) != 0:
            return True, selected_boxes_yquad, selected_scores_yquad, selected_labels_yquad
        else:
            return False, np.array([]), np.array([]), np.array([])

    
class YQUADMarkerDetectorYOLOv5(TargetmarkerDetector):  
    def set_xai_target_layers(self):
        self.xai_target_layers = [self.model.model.model.model[-2]]

    def build_model(self):
        assert self.num_classes is not None
        self.model = self.models_dict[self.model_type](self.num_classes, checkpoint_path=self.path_to_model_checkpoint, device=self.device)

    def setup_model(self):
        self.build_model()
        self.model.to(self.device).eval()

    def detect(self, image):    
        results = self.model([image]).xyxy[0].cpu().numpy()
        
        selected_boxes_yquad = [[int(r[0]),int(r[1]),int(r[2]),int(r[3])] for r in results if r[4] > self.detection_threshold]
        selected_scores_yquad = [r[4] for r in results if r[4] > self.detection_threshold]
        selected_labels_yquad = [int(r[5]) for r in results if r[4] > self.detection_threshold]

        if len(selected_boxes_yquad) != 0:
            return True, selected_boxes_yquad, selected_scores_yquad, selected_labels_yquad
        else:
            return False, np.array([]), np.array([]), np.array([])
        

class YQUADMarkerDetectorFRCNN(TargetmarkerDetector):
    def setup_model(self):
        super().setup_model()
        self.build_model()
        self.restore_checkpoint() 
        self.model.to(self.device).eval()
        
    def detect(self, image):
        image = image.copy()
        image_tensor = self.prepare_image_tensor(image)

        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
        
        # load all detections to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy()

            mask = np.where(scores >= self.detection_threshold,1,0).astype(np.bool)

            selected_boxes_yquad = boxes[mask].astype(np.int16)
            selected_scores_yquad = scores[mask].astype(np.float32)
            selected_labels_yquad = labels[mask].astype(np.int8)

            return True, selected_boxes_yquad, selected_scores_yquad, selected_labels_yquad
        else:
            return False, np.array([]), np.array([]), np.array([])