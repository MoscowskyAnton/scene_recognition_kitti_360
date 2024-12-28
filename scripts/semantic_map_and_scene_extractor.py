#!/usr/bin/env python3
# coding: utf-8


'''
This script can create semantic maps and scenes from KITTI-360 dataset
'''
import xml.etree.ElementTree as ET
import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import copy
import yaml
import pandas
from scipy.optimize import linear_sum_assignment
import sys
import argparse
from datetime import datetime
from scipy.spatial.transform import Rotation
import re

# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# from https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    if area_of_union == 0:
        print(ground_truth, pred)
     
    iou = area_of_intersection / area_of_union     
    return iou


def to_cv_color(plt_color, rev = True):    
    if rev:
        return (int(256*plt_color[2]), int(256*plt_color[1]), int(256*plt_color[0]))
    else:
        return (int(256*plt_color[0]), int(256*plt_color[2]), int(256*plt_color[1]))

class SemanticMapAndScenesExtractor(object):
    
    # list of IDs of semantic classes in 2d semantic labaled images,NOTE that may have some mistakes
    semanticIDs = {
                7: 'road',
                8: 'sidewalk',
                9: 'driveway',
                11: 'building',
                12: 'wall',
                13: 'fence',
                17: 'bigPole',
                19: 'trafficLight',
                20: 'trafficSign',
                21: 'vegetation',
                22: 'ground',
                23: 'sky',
                24: 'pedestrian',
                26: 'car',
                27: 'truck',
                32: 'motorcycle',
                34: 'garage',
                35: 'gate',
                37: 'smallPole',
                38: 'lamp',
                39: 'trashbin',
                41: 'box',
                40: 'vendingmachine',
                44: 'unknownObject',
                }
    
    '''
    kitti_360_path - path to KITTI-360 folder
    save_path - path to save results, if none it will create folder in kitti-360 path named scene_recognition
    sequence - sequence number set as str like '02'
    min_frame - frame to start, if -1 starts from begining
    max_frame - frame to end (includes that), if -1 ends in end
    to_the_end - overwrited end frame of search
    min_object_area_px - if object mask area in pixels will less, that ignore it
    max_object_dist_m - if object is farer than that ignore it    
    iou_cost_th - threshold for IoU when matching rects and masks
    min_lidar_intensity - reject weak lidar point by threshold
    min_lidar_points - reject objects with less point than that number on em
    do_clip - calc CLIP features
    objects_ignore - add objects to ignore NOTE that some already are
    video_cam - saves video from cam where objects are drawn
    video_map - saves video of map with cam and seen object located
    video_mix - saves mixed video of both
    plot_invisible - draws and plots invisible objects 
    '''
    def __init__(self, kitti_360_path, save_path = None, sequence = '00', min_frame = -1, max_frame = -1, to_the_end = False, min_object_area_px = 50, max_object_dist_m = 50, iou_cost_th = 0.2, min_lidar_intensity = 0.5, min_lidar_points = 10, do_clip = False, objects_ignore = [], video_cam = True, video_map = True, video_mix = True, plot_invisible = False, **kwargs):
        
        self.kitti_360_path = kitti_360_path
        
        self.sequence = sequence
        self.min_frame = min_frame
        self.max_frame = max_frame
        self.to_the_end = to_the_end
        self.min_object_area_px = min_object_area_px
        self.max_object_dist_m = max_object_dist_m
        self.iou_cost_th = iou_cost_th
        self.do_clip = do_clip
        self.min_lidar_intensity = min_lidar_intensity
        self.min_lidar_points = min_lidar_points
        self.plot_invisible = plot_invisible
        
        self.video_cam = video_cam
        self.video_map = video_map
        self.video_mix = video_mix
        
        add_path = f"/scene_recogntion/sequence{sequence}/save_{datetime.now().strftime('%d_%m_%H_%M')}/"        
        if save_path is None:
            self.save_path = kitti_360_path + add_path
        else:
            self.save_path = save_path + add_path
        os.makedirs(self.save_path+"/scenes", exist_ok=True)
        
        self.objects_ignore = objects_ignore + ['driveway', 
                                                'fence', 
                                                'ground', 
                                                'pedestrian', 
                                                'railtrack',
                                                'road',
                                                'sidewalk',
                                                'unknownConstruction',
                                                'unknownGround',
                                                'unknownObject',
                                                'vegetation',
                                                'wall',
                                                'guardrail'
                                                ]                
        '''
        MAP = {}
        'class_label': [
             - global_id: {}
                - pose: [4x4]
                  frames: (start, end)
                  vertices: [12x3]
                  frames_filtered: []
                  mean_features: (N)
                ]
            }
        '''      
        self.MAP = {}
        self.object_colors = {}
        
        if self.do_clip:
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            
            ''' for cached please modify this:
            cached = "/home/${USER}/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/"                                    
            self.clip_processor = AutoProcessor.from_pretrained(cached)
            self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(cached)
            '''
            
            self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")

        
        
    
    def read_stuff(self):
        self.read_camera_poses()
        self.read_camera_calib()
        self.read_semantic_map()
        self.calc_frames_to_objects()
        
        self.calc_and_save_scenes(self.video_cam, self.video_map, self.video_mix)
        self.post_process_map()
        
        self.export_csv()
        
    ###
    # UTILS
    ###
    
    def load_frame(self, frame_no):
        return cv2.imread(f"{self.kitti_360_path}/data_2d_raw/2013_05_28_drive_00{self.sequence}_sync/image_00/data_rect/{str(frame_no).zfill(10)}.png")     
    
    def object_used(self, label):
        return not label in self.objects_ignore
    
    def get_total_objects(self):
        total = 0
        for label in self.MAP.values():
            total += len(label)
        return total
    
    def get_obj_by_gid(self, gid):
        for obj in self.MAP.values():
            if gid in obj:
                return obj[gid]
        raise ValueError(f'No gid {gid} in map!')
    
    def get_label(self, gid):
        for label, obj in self.MAP.items():
            if gid in obj:
                return label
        raise ValueError(f'No object {gid} in map!')
    
    def get_angles_point(self, pt):
        x = (pt[0] - self.P[0, 2]) / self.P[0, 0]
        y = (pt[1] - self.P[1, 2]) / self.P[1, 1]
        ax = np.arctan2(x, 1)
        ay = np.arctan2(y, 1)
        return float(ax), float(ay)    
    
    ###
    # READ AND PROCESS STUFF
    ###
    
    def read_camera_poses(self):
        poses_path = f'{self.kitti_360_path}/data_poses/2013_05_28_drive_00{self.sequence}_sync/cam0_to_world.txt'
    
        f_poses = open(poses_path, "r")        
        raw_cam_poses = f_poses.read().split('\n')
        
        '''
        frame_no: 
          transform: [4x4]
          objects: {obj_gid: {rect: (4,), 
                              r: float, 
                              vis: bool, 
                              features: (N),
                              cam_pose: (3)
                              }}
        '''
        self.cam_poses = {}
        
        for cam_pose in raw_cam_poses:            
            pose = cam_pose.split(" ")
            if len(pose) == 18: 
                frame_no = int(pose[0])
                if frame_no < self.min_frame and self.min_frame != -1:
                    continue
                if frame_no > self.max_frame and self.max_frame != -1:
                    break 
                self.cam_poses[frame_no] = {'transform':np.array(pose[1:17], dtype = np.float32).reshape(4,4), 'objects': {}}
        
        print("Camera poses have been read.")
        
    def read_camera_calib(self):
        cam_info_path = f'{self.kitti_360_path}/calibration/perspective.txt'
        f_cam_info = open(cam_info_path, "r")        
        raw_cam_info = f_cam_info.read().split('\n')
        for cam_info in raw_cam_info:
            cam_info_raw = cam_info.split(" ")
            if cam_info_raw[0] == 'K_00:':            
                self.K = np.array(cam_info_raw[1:10], dtype = np.float32).reshape(3,3)
            if cam_info_raw[0] == 'R_rect_00:':
                self.R = np.eye(4)
                self.R[:3,:3] = np.array(cam_info_raw[1:10], dtype = np.float32).reshape(3,3) 
            if cam_info_raw[0] == 'P_rect_00:':
                self.P = np.array(cam_info_raw[1:13], dtype = np.float32).reshape(3,4) 
            if cam_info_raw[0] == 'D_00:':
                self.D = np.array(cam_info_raw[1:6], dtype = np.float32)
        
        #test_image = self.load_frame(0) # TODO not all seqs starts with 0        
        #self.image_shape = test_image.shape
        
        self.image_shape = (376, 1408) # DANGER hardcoded
        
        fovy, fovx, _, _, _ = cv2.calibrationMatrixValues(self.P[:3,:3], self.image_shape[:2], 0, 0)
        self.fovx = np.deg2rad(fovx)
        self.fovy = np.deg2rad(fovy)
        
        print("Camera params have been read.")
        
    def read_semantic_map(self):
        print('Reading semantic map...')        
        
        full_path_3d_labels = f'{self.kitti_360_path}/data_3d_bboxes/train_full/2013_05_28_drive_00{self.sequence}_sync.xml'
        
        _3d_labels_root = ET.parse(full_path_3d_labels).getroot()    
        
        for object_tag in _3d_labels_root:
            if object_tag.tag.startswith('object'):
                obj_label = object_tag.find('label').text
                if obj_label in self.objects_ignore:
                    continue                
                if object_tag.find('dynamic').text == '1':
                    continue
                
                if not obj_label in self.MAP:
                    self.MAP[obj_label] = {}                                    
                    #self.object_colors[obj_label] = plt.get_cmap('tab10', 10)(len(self.object_colors)%10)
                    #self.object_colors[obj_label] = plt.get_cmap('Pastel1', 9)(len(self.object_colors)%9)
                    self.object_colors[obj_label] = plt.get_cmap('tab20b', 20)(len(self.object_colors)%20)
                    #self.object_colors[obj_label] = plt.get_cmap('Set3', 12)(len(self.object_colors)%12)
                
                new_obj = {}
                
                transform = np.fromstring(object_tag.find('transform').find('data').text,
                                            sep=" ",
                                            dtype=np.float32).reshape(4,4)
                new_obj['pose'] = transform
                                
                new_obj['frames'] = [int(object_tag.find('start_frame').text), int(object_tag.find('end_frame').text) ]
                
                if self.min_frame != -1 and self.min_frame > new_obj['frames'][0]:
                    new_obj['frames'][0] = self.min_frame
                if self.max_frame != -1 and self.max_frame < new_obj['frames'][1]:
                    new_obj['frames'][1] = self.max_frame
                if self.to_the_end:
                    if self.max_frame == -1:
                        new_obj['frames'][1] = max(self.cam_poses.keys())
                    else:
                        new_obj['frames'][1] = self.max_frame
                
                vert_tag = object_tag.find('vertices')                    
                                    
                new_obj['vertices'] = np.fromstring( vert_tag.find('data').text, sep=" ", dtype = np.float32).reshape(int(vert_tag.find('rows').text), int(vert_tag.find('cols').text))
                
                gid = int(object_tag.find('index').text)
                
                new_obj['frames_visible'] = []
                
                self.MAP[obj_label][gid] = new_obj
                            
        print(f"Done, labels collected: {list(self.MAP.keys())}")                       
    
    def calc_frames_to_objects(self):        
        total_objs = self.get_total_objects()
        proceed = 1     
        for label, objects in self.MAP.items():
            printProgressBar(proceed, total_objs, prefix = 'Calculating frames to objects:', suffix = 'Complete', length = 50)            
            for gid, obj in objects.items():
                proceed+=1                               
                frames = range(obj['frames'][0], obj['frames'][1]+1)                
                for frame_no in frames:                    
                    if frame_no in self.cam_poses:                    
                        # check range, check fhov
                        object_pose_in_cam_frame = (np.linalg.inv(self.cam_poses[frame_no]['transform']) @ obj['pose'] @ (0, 0, 0, 1)).T
                                                                        
                        r = object_pose_in_cam_frame[2] # signed!                                                                        
                        
                        if r < self.max_object_dist_m and r > 0:                                                        
                            
                            x_angle = np.arctan2(object_pose_in_cam_frame[0], object_pose_in_cam_frame[2])
                            y_angle = np.arctan2(object_pose_in_cam_frame[1], object_pose_in_cam_frame[2])
                                                
                            # 1.5 not 2 in devider is because center could be out of field of view
                            dev = 1.5                             
                            if abs(x_angle) <= self.fovx/dev:
                                rect = self.get_bounding_boxes_for_frame(frame_no, gid)
                                
                                if not rect is None:
                                    self.cam_poses[frame_no]['objects'][gid] = {'rect': rect, 'r': r, "cam_pose" : object_pose_in_cam_frame[:3]}                                                                    
                            
        print("Frames to objects calculated.")
        
        
    def get_bounding_boxes_for_frame(self, frame_no, obj_gid):
            
        if not frame_no in self.cam_poses:
            #print(f'Frame {frame_no} has no pose!')
            return None                        
            
        object_inst = self.get_obj_by_gid(obj_gid)
                                      
        points_tr = np.hstack( (object_inst['vertices'], np.ones((object_inst['vertices'].shape[0], 1 ))) )
        
        points_transformed = (self.P @ self.R @ np.linalg.inv(self.cam_poses[frame_no]['transform']) @ object_inst['pose'] @ points_tr.T).T                                                             
                    
        points = np.array(points_transformed / points_transformed[:,2:], dtype=int)                                            
        
        # x left
        points[(points_transformed[:,2] < 0) & (points_transformed[:,0] < 0), 0] = 0
        # x right
        points[(points_transformed[:,2] < 0) & (points_transformed[:,0] > 0), 0] = self.image_shape[1]        
        
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])                
        
        if max_x <= 0 or min_x >= self.image_shape[1]:
            return None
        if max_y <= 0 or min_y >= self.image_shape[0]:
            return None
        
        min_x = max(min_x, 0) 
        max_x = min(max_x, self.image_shape[1]-1)
        
        min_y = max(min_y, 0)
        max_y = min(max_y, self.image_shape[0]-1)                    
        
        rect = (int(min_x), int(min_y), int(max_x), int(max_y))
        if rect == (0, 0, self.image_shape[1]-1, self.image_shape[0]-1):
            return None
        
        w = max_x - min_x
        h = max_y - min_y
        if w * h <= self.min_object_area_px:
            return None        
        return rect
        
        
    def calc_and_save_scenes(self, video_cam = False, video_map = False, video_mix = True):
        proceed = 0
        full_len = len(self.cam_poses)
        
        transform_f = open(f'{self.kitti_360_path}/calibration/calib_cam_to_velo.txt', "r")
        transform_cam_velodyne = np.array(transform_f.read().split(' '), dtype=np.float64).reshape(3, 4)
        transform_cam_velodyne = np.vstack((transform_cam_velodyne, (0, 0, 0, 1)))
        
        if video_cam:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')                                
            video_path = f'{self.save_path}/video_camera.avi'
            video=cv2.VideoWriter(video_path ,fourcc, 10.0, (self.image_shape[1], self.image_shape[0]) )
    
        if video_map:
            video_map_file = None
        
        if video_mix:
            video_mix_file = None
        
        total_scenes_saved = 0
        for frame_no, frame in self.cam_poses.items():            
            proceed+=1
            printProgressBar(proceed, full_len, prefix = 'Proceeding and saving scenes:', suffix = 'Complete', length = 50)
            
            # check semantic and lidar data            
            semantic_image_path = f"{self.kitti_360_path}/data_2d_semantics/train/2013_05_28_drive_00{self.sequence}_sync/image_00/instance/{str(frame_no).zfill(10)}.png"            
            velodyne_data_path = f"{self.kitti_360_path}/data_3d_raw/2013_05_28_drive_00{self.sequence}_sync/velodyne_points/data/{str(frame_no).zfill(10)}.bin"
            
            if os.path.isfile(semantic_image_path) and os.path.isfile(velodyne_data_path):                
                ### DEAL WITH MASKS
                segmented_image = cv2.imread(semantic_image_path, -1)                
                # get all types
                unique_ids = np.unique(segmented_image)                
                # extracting objects that are good
                segmented_objects = {}
                no_inst_seg = {}
                for obj_id in unique_ids.tolist():
                                                            
                    classInstanceID = obj_id % 1000
                    semanticID = int((obj_id - classInstanceID)/1000)
                    
                    if semanticID in SemanticMapAndScenesExtractor.semanticIDs:
                        label = SemanticMapAndScenesExtractor.semanticIDs[semanticID]                                            
                        
                        if not self.object_used(label):
                            continue
                    
                        mask = cv2.inRange(segmented_image, obj_id, obj_id)
                    
                        if classInstanceID == 0:
                            no_inst_seg[label] = mask
                            continue # actually there are no instance                                                                                        
                        
                        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                        
                        fine_contours = []
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > self.min_object_area_px:
                                fine_contours.append(contour)
                        
                        if len(fine_contours):                            
                            new_mask = cv2.drawContours(np.zeros(segmented_image.shape, np.uint8), fine_contours, -1, (255), -1)                            
                            x,y,w,h = cv2.boundingRect(new_mask)                               
                            rect = (x, y, x+w, y+h)                   
                            
                            if not label in segmented_objects:
                                segmented_objects[label] = []
                                                            
                            segmented_objects[label].append({'classInstanceID': classInstanceID,
                                                             'rect': rect,
                                                             'mask': new_mask})
                            
                # now combine it with existence                
                for label, seg_objects in segmented_objects.items():
                    # get appr labels from frame objects
                    selected_objects = []
                    for gid, obj in frame['objects'].items():
                        gid_label = self.get_label(gid)
                        if gid_label == label:
                            selected_objects.append(obj)
                    
                    if len(selected_objects):                        
                        comp_mat = np.zeros( (len(selected_objects), len(seg_objects)) )                        
                        def get_score(rect1, rect2, mask): # legacy
                            return get_iou(rect1, rect2)
             
                        # fill mat
                        for i, sel_obj in enumerate(selected_objects):
                            for j, seg_obj in enumerate(seg_objects):
                                comp_mat[i, j] = get_score(sel_obj['rect'], seg_obj['rect'], seg_obj['mask'])
                                            
                        # calc optimum
                        opt_map = linear_sum_assignment(comp_mat, maximize = True)
                        for i in range(opt_map[0].shape[0]):
                            sel_obj_index = opt_map[0][i]
                            seg_obj_index = opt_map[1][i]
                            
                            # check cost btw
                            if comp_mat[sel_obj_index, seg_obj_index] > 0:
                                r = selected_objects[sel_obj_index]['rect']
                                selected_objects[sel_obj_index]['cropped_mask'] = seg_objects[seg_obj_index]['mask'][r[1]:r[3], r[0]:r[2]]
                    
                # and objects without instance too
                for label, mask in no_inst_seg.items():
                    for gid, obj in frame['objects'].items():
                        gid_label = self.get_label(gid)
                        if gid_label == label:
                            r = obj['rect']
                            cropped_mask = mask[r[1]:r[3], r[0]:r[2]]
                            
                            area = np.count_nonzero(cropped_mask)
                            if area > self.min_object_area_px:
                                obj['cropped_mask'] = cropped_mask 
            
            
                                
                ### DEAL WITH LIDAR            
                    
                pointcloud_bin_velo = np.fromfile(velodyne_data_path, dtype=np.float32).reshape(-1, 4)                                    
                points_velo = np.hstack((pointcloud_bin_velo[:,:3],np.ones((pointcloud_bin_velo.shape[0],1)))).T                
                
                # transform to cam 
                pointcloud_tr = ((np.linalg.inv(transform_cam_velodyne) @ points_velo)[:3,:]).T
                #pointcloud_tr = (transform_cam_velodyne @ points_velo)[:3,:].T            
                
                pointcloud_angles_x = np.arctan2(pointcloud_tr[:,0], pointcloud_tr[:, 2])
                pointcloud_angles_y = np.arctan2(pointcloud_tr[:,1], pointcloud_tr[:, 2])
                                                    
                # get rect angles, (xmin ymin xmax ymax)                        
                def get_rect_angles(rect):            
                    angle_x_min, angle_y_min = self.get_angles_point((rect[0], rect[1]))
                    angle_x_max, angle_y_max = self.get_angles_point((rect[2], rect[3]))                                                
                    return (angle_x_min, angle_x_max), (angle_y_min, angle_y_max)                        
                                        
                for gid, obj in frame['objects'].items():                                   
                    if self.object_used(gid) and 'cropped_mask' in obj:
                        angles_x, angles_y = get_rect_angles(obj['rect'])
                                                                
                        x, y, w, h = cv2.boundingRect(obj['cropped_mask'])
                        obj['mask_angles'] = self.get_angles_point((x+w/2 + obj['rect'][0], y+h/2 + obj['rect'][1]))
                                                                
                        good_points_ind = np.where(pointcloud_angles_x > angles_x[0], True, False) * np.where(pointcloud_angles_x < angles_x[1], True, False) * np.where(pointcloud_angles_y > angles_y[0], True, False) * np.where(pointcloud_angles_y < angles_y[1], True, False)
                        
                        rect_points = pointcloud_tr[good_points_ind]
                        
                        points_proj = self.P @ self.R @ np.hstack((rect_points, np.zeros((rect_points.shape[0],1)))).T                    
                        points_proj = ((points_proj[:2,:] / points_proj[2,:]).astype(int)).T - np.array([obj['rect'][0], obj['rect'][1]])                    
                                                                                                                    
                        if points_proj.shape[0] > 0:                        
                            ranges = []
                            weights = []
                            for i in range(points_proj.shape[0]):
                                py = points_proj[i,1]
                                px = points_proj[i,0]
                                if px < obj['cropped_mask'].shape[1] and py < obj['cropped_mask'].shape[0] and px >= 0 and py >= 0:
                                    if obj['cropped_mask'][py, px] > 0:
                                        
                                        w = pointcloud_bin_velo[good_points_ind][i, 3]
                                        if w >= self.min_lidar_intensity:                                                                        
                                            r = np.sqrt(np.sum(np.power(rect_points[i,:3], 2)))
                                            ranges.append(r)
                                            weights.append(w)                                                                
                                                                                                    
                            if len(ranges)  >= self.min_lidar_points:                                                                                    
                                obj['lidar_dist_min'] = float(np.min(ranges))
                                obj['lidar_dist_mean'] = float(np.mean(ranges))
                                obj['lidar_dist_med'] = float(np.median(ranges))
                                obj['lidar_dist_wavg'] = float(np.average(ranges, weights = weights))
                                obj['lidar_dist_q1'] = float(np.quantile(ranges, 0.25, method = "higher"))                   
                                                        
                ### DEAL WITH CLIP
                if self.do_clip:
                    image = self.load_frame(frame_no)
                    for gid, obj in frame['objects'].items():                    
                        if 'cropped_mask' in obj:
                            
                            cropped_obj = (image[obj['rect'][1]:obj['rect'][3],
                                            obj['rect'][0]:obj['rect'][2]]).copy()
                                                    
                            cropped_obj[obj['cropped_mask'] == 0] = (127,127,127)                                                                        
                            inputs = self.clip_processor(text = [""], images = [cropped_obj], return_tensors = "pt", padding=True) # TODO find way not to proceed text every time
                            outputs = self.clip_model(**inputs)
                            features = outputs.image_embeds.detach().numpy().flatten()    
                            obj['features'] = features / np.linalg.norm(features) # normalize vectors
                            self.feature_len = obj['features'].shape[0]
                                                
                ### SAVE 
                objects = {}
                for gid, obj in frame["objects"].items():                
                    label = self.get_label(gid)
                    if 'lidar_dist_min' in obj:
                        
                        self.get_obj_by_gid(gid)["frames_visible"].append(frame_no)
                    
                        objects[gid] = {"cam_pose" : obj["cam_pose"].astype(float).tolist(),
                                        "rect" : list(obj["rect"]),
                                        "label": label
                                        }                    
                        if self.do_clip:
                            objects[gid]["features"] = obj["features"].astype(float).tolist()                        
                        for key in obj.keys():
                            if "lidar_dist" in key:
                                objects[gid][key] = obj[key]                                                                    
                        objects[gid]["mask_angles"] = list(obj["mask_angles"])                        
                if len(objects) > 1:                
                    scene = {"transform": frame["transform"].astype(float).tolist(),
                                    "objects": objects,
                                    "frame_no": frame_no}
                    file_path = self.save_path + f"/scenes/frame{frame_no}.yaml"
                    with open(file_path, "w") as file:
                        yaml.dump(scene, file)                    
                        total_scenes_saved+=1                 
                
                ### VIDEOS
                if video_cam or video_mix:
                    image = self.draw_frame_objects(frame_no)
                    image = cv2.rectangle(image, (0, 0), (100, 12), (0, 0, 0), -1)                       
                    image = cv2.putText(image, f'frame{frame_no}', (0, 11), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)      
                    if video_cam:
                        video.write(image)
                
                if video_map or video_mix:
                                    
                    img = self.plot_scene_in_map_compass(frame_no)     
                    
                    if video_map and video_map_file is None:
                        fourcc_ = cv2.VideoWriter_fourcc(*'XVID')                                
                        video_path = f'{self.save_path}/video_map.avi'
                        video_map_file = cv2.VideoWriter(video_path ,fourcc_, 10.0, (img.shape[1], img.shape[0]) )
                    
                    if video_map:
                        video_map_file.write(img)
                
                if video_mix:
                    
                    scale = img.shape[0] / image.shape[0] 
                    img_re = cv2.resize(img, (int(img.shape[1] / scale), image.shape[0]) )                                     
                    
                    merged_img = np.concatenate((image, img_re), axis=1)
                    
                    if video_mix_file is None:
                        fourcc__ = cv2.VideoWriter_fourcc(*'XVID')                                
                        video_path = f'{self.save_path}/video_mix.avi'
                        video_mix_file = cv2.VideoWriter(video_path ,fourcc__, 10.0, (merged_img.shape[1], merged_img.shape[0]) )
                    
                    #cv2.imshow('Image', merged_img)
                    #cv2.waitKey(1)
                    
                    video_mix_file.write(merged_img)
                        
                
                ### DROP MASK
                for gid, obj in frame['objects'].items():                
                    obj['cropped_mask'] = None
                    
        if video_cam:
            video.release()
        if video_map:
            video_map_file.release()
        if video_mix:
            video_mix_file.release()
        
        print(f"Scenes proceeded and saved ({total_scenes_saved})!")
    

    def post_process_map(self):
        for label, objects in self.MAP.items():
            for gid, obj in objects.items():
                features = []                
                for frame_no in obj['frames_visible']:
                    scene_obj = self.cam_poses[frame_no]['objects'][gid]                    
                    if 'features' in scene_obj:
                        features.append(scene_obj['features'])
                if len(features):
                    obj['mean_features'] = np.mean(features, axis = 0)                    
        print("Map was postprocessed.")
                
                
    def export_csv(self):                
        file_path = self.save_path + f'/semantic_map.csv'        
        data = []    
        for label, objects in self.MAP.items():
            if not self.object_used(label):
                continue            
            for gid, obj in objects.items():
                if len(obj['frames_visible']) == 0:
                    continue                
                frame_str = ''
                for frame in obj['frames_visible']:
                    frame_str += f'{frame} '                    
                frame_str = frame_str[:-1]
                row = [gid, label, float(obj['pose'][0,3]), float(obj['pose'][1,3]), float(obj['pose'][2,3]), frame_str ]                
                if self.do_clip:                    
                    row += obj['mean_features'].astype(float).tolist()                                    
                data.append(row)                
        if len(data):
            df = pandas.DataFrame(data)
            lables = ['gid', 'class', 'x', 'y', 'z', 'frames']        
            if self.do_clip:
                for i in range(self.feature_len):
                    lables += [f'mf{i}']            
            df.to_csv(file_path, encoding='utf-8', header = lables)
            print(f"Map saved to {file_path}")        
    
    ###
    # PLOT (matplotlib) AND DRAW (cv2) STUFF
    ###        
    def plot_2d_map(self, plot_ids = False, plot_path = True):        
        plt.figure('2d_semantic_map')
        plt.cla()
        ax = plt.gca()
        plt.title(f'Semantic map (seq{self.sequence})')        
        for object_label, objects in self.MAP.items():            
            for gid, obj in objects.items():                                
                if len(obj['frames_visible']) == 0:
                    continue                
                plt.plot(obj['pose'][0, 3], obj['pose'][1, 3], '.', color =  self.object_colors[object_label], label = object_label)
                if plot_ids:
                    plt.text(obj['pose'][0, 3], obj['pose'][1, 3], str(gid), color =  self.object_colors[object_label])                        
        
        poses_x = []
        poses_y = []
        for frame_no, frame in self.cam_poses.items():
            poses_x.append(frame['transform'][0,3])
            poses_y.append(frame['transform'][1,3])
                    
        plt.plot(poses_x, poses_y, '--r', label = 'camera path')
        plt.text(poses_x[0], poses_y[0], 'start', color = 'r')
        plt.text(poses_x[-1], poses_y[-1], 'end', color = 'r')
                
        ax.set_aspect('equal', adjustable='box')
        plt.grid()
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))        
    
    
    def plot_scene_in_map(self, frame_no, use_dist = 'lidar_dist_min'):
        fig = plt.figure(f"frame_{frame_no}")
        
        obj_poses = []
        for gid, obj in self.cam_poses[frame_no]['objects'].items():                        
            if use_dist in obj:            
                x = obj[use_dist] * np.sin(obj['mask_angles'][0])
                y = obj[use_dist] * np.sin(obj['mask_angles'][1])
                z = obj[use_dist] * np.cos(obj['mask_angles'][0])                            
                obj_poses.append( [gid, x, y, z, 1] )            
        
        transform = np.array(self.cam_poses[frame_no]['transform'])
        robot_pose = np.array([[0, 0 , 0, 1], [self.max_object_dist_m / 5, 0, 0, 1]]).T
        robot_pose = (transform @ robot_pose).T
        
        plt.arrow(robot_pose[0, 0], robot_pose[0, 1], robot_pose[1, 0] - robot_pose[0, 0], robot_pose[1, 1] - robot_pose[0, 1], label = "cam x", color = 'red')
        
        robot_pose = np.array([[0, 0 , 0, 1], [0, 0, self.max_object_dist_m / 5, 1]]).T
        robot_pose = (transform @ robot_pose).T
        
        plt.arrow(robot_pose[0, 0], robot_pose[0, 1], robot_pose[1, 0] - robot_pose[0, 0], robot_pose[1, 1] - robot_pose[0, 1], label = "cam z", color = 'blue')
        
        obj_poses = np.array(obj_poses)                        
        
        if obj_poses.shape[0]:            
            obj_poses[:,1:] = (transform @ obj_poses[:,1:].T).T
        
        for label, objects in self.MAP.items():
            color = self.object_colors[label]
            for gid, obj in objects.items():
                if gid in self.cam_poses[frame_no]['objects']:                    
                    if use_dist in self.cam_poses[frame_no]['objects'][gid]:
                        color = self.object_colors[label]
                    else:
                        color = (0.5, 0.5, 0.5)
                        
                    if self.plot_invisible or use_dist in self.cam_poses[frame_no]['objects'][gid]:
                        plt.plot(obj['pose'][0, 3], obj['pose'][1, 3], '.', label = f'M{gid}', color = color)
                        plt.text(obj['pose'][0, 3], obj['pose'][1, 3], f'{gid}', color = color)
                    
                        for i in range(obj_poses.shape[0]):
                            if gid == obj_poses[i, 0]:
                                plt.plot([obj_poses[i, 1], obj['pose'][0, 3]], [obj_poses[i, 2], obj['pose'][1, 3]], ':', color = color)                
                        
        for i in range(obj_poses.shape[0]):
            color = self.object_colors[self.get_label(int(obj_poses[i, 0]))]
            plt.plot(obj_poses[i, 1], obj_poses[i, 2], 'o', label = f'Sl{int(obj_poses[i, 0])}',color = color)                    
            #plt.text(obj_poses[i, 1], obj_poses[i, 2], f'{int(obj_poses[i, 0])}', color = color)  
            
            #plt.plot([robot_pose[0, 0], obj_poses[i, 1]], [robot_pose[0, 1], obj_poses[i, 2]], ':', color = color)
        
        plt.grid()
        plt.title(f'Frame {frame_no}')
        plt.gca().set_aspect('equal', adjustable='box')
        m = 1
        plt.xlim(robot_pose[0, 0] - self.max_object_dist_m * m, robot_pose[0, 0] + self.max_object_dist_m * m)
        plt.ylim(robot_pose[0, 1] - self.max_object_dist_m * m, robot_pose[0, 1] + self.max_object_dist_m * m)
        
        fig.canvas.draw()
        cv_img = np.array(fig.canvas.renderer.buffer_rgba())
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
        plt.close()
        return cv_img
        
        
    def plot_scene_in_map_compass(self, frame_no, use_dist = 'lidar_dist_min'):
        fig = plt.figure(f"frame_{frame_no}")
        
        obj_poses = []
        for gid, obj in self.cam_poses[frame_no]['objects'].items():                        
            if use_dist in obj:            
                x = obj[use_dist] * np.sin(obj['mask_angles'][0])
                y = obj[use_dist] * np.sin(obj['mask_angles'][1])
                z = obj[use_dist] * np.cos(obj['mask_angles'][0])                            
                obj_poses.append( [gid, x, y, z, 1] )            
        
        transform = np.array(self.cam_poses[frame_no]['transform'])        
        
        plt.arrow(0, 0, self.max_object_dist_m / 10, 0, label = "cam x", color = 'red')                        
        plt.arrow(0, 0, 0, self.max_object_dist_m / 10, label = "cam z", color = 'blue')
        
        obj_poses = np.array(obj_poses)                                
        
        for label, objects in self.MAP.items():
            color = self.object_colors[label]
            for gid, obj in objects.items():
                if gid in self.cam_poses[frame_no]['objects']:                    
                    if use_dist in self.cam_poses[frame_no]['objects'][gid]:
                        color = self.object_colors[label]
                    else:
                        color = (0.5, 0.5, 0.5)
                        
                    if self.plot_invisible or use_dist in self.cam_poses[frame_no]['objects'][gid]:
                        
                        object_pose = np.linalg.inv(transform) @ np.array([obj['pose'][0, 3],
                                                                          obj['pose'][1, 3],
                                                                          obj['pose'][2, 3],
                                                                          1])                        
                        
                        plt.plot(object_pose[0], object_pose[2], '.', label = f'M{gid}', color = color)
                        plt.text(object_pose[0], object_pose[2], f'{gid}', color = color)
                    
                        for i in range(obj_poses.shape[0]):
                            if gid == obj_poses[i, 0]:
                                plt.plot([obj_poses[i, 1], object_pose[0]], [obj_poses[i, 3], object_pose[2]], ':', color = color)                
                        
        for i in range(obj_poses.shape[0]):
            color = self.object_colors[self.get_label(int(obj_poses[i, 0]))]
            plt.plot(obj_poses[i, 1], obj_poses[i, 3], 'o', label = f'Sl{int(obj_poses[i, 0])}',color = color)                    
                                    
        
        plt.grid()
        plt.title(f'Frame {frame_no}')
        plt.gca().set_aspect('equal', adjustable='box')
        m = 1
        
        x_shift = self.max_object_dist_m * m * np.cos(self.fovx/2)                
        plt.xlim( -x_shift-1, x_shift+1)
        plt.ylim( -1, self.max_object_dist_m * m)
        
        # compass                        
        yaw = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')[2]                
        r = self.max_object_dist_m / 20        
        plt.arrow(-x_shift + r, r, r * np.cos(yaw), r * np.sin(yaw), label = "cam x+", color = 'red')
        plt.arrow(-x_shift + r, r, r * np.cos(yaw + np.pi/2), r * np.sin(yaw + np.pi/2), label = "cam z+", color = 'blue')        
        circle = plt.Circle( (-x_shift + r, r), r, color='k', fill=False)
        plt.gca().add_patch(circle)
        
        fig.canvas.draw()
        cv_img = np.array(fig.canvas.renderer.buffer_rgba())
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
        plt.close()
        return cv_img
    
    
    def draw_frame_objects(self, frame_no, draw_masks = True, alpha = 0.4):
        
        if frame_no in self.cam_poses:                
            image = self.load_frame(frame_no)              
            
            if draw_masks:                
                overlay = image.copy()                                
                
                for gid, obj in self.cam_poses[frame_no]['objects'].items():                                                                                     
                    if 'cropped_mask' in obj:                                                                    
                        rect = obj['rect']       
                        label = self.get_label(gid)                    
                        color = to_cv_color(self.object_colors[label], rev = True)                
                        if not 'lidar_dist_min' in obj:
                            color = (127, 127, 127)                    
                                                
                        full_mask = np.zeros(image.shape[:2], np.uint8)                        
                        full_mask[rect[1]:rect[3], rect[0]:rect[2]] = obj['cropped_mask']                        
                        overlay[full_mask > 0] = color                       
                        
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)             
            for gid, obj in self.cam_poses[frame_no]['objects'].items():
                                
                if self.plot_invisible or 'lidar_dist_min' in obj:
                    
                    rect = obj['rect']                                                                                        
                    label = self.get_label(gid)                    
                    color = to_cv_color(self.object_colors[label])                
                    if not 'lidar_dist_min' in obj:
                        color = (127, 127, 127)                                        
                    image = cv2.rectangle(image, rect[:2], rect[2:], color, 2)                      
                    image = cv2.putText(image, f'{label}{gid}', (rect[0], rect[1]+20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color, 1, cv2.LINE_AA)                                               
                                                                                                                                                                                                                                             
            return image
        else:
            raise ValueError(f'No frame {frame_no} in poses!')   
                              
    
###
# INTERFACE
###
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--kitti_360_path', type=str, required = True, help = "Path to KITTI-360 folder")
    parser.add_argument('--save_path', type=str, default = None, help = "Save path")
    
    parser.add_argument('--sequence', type=str, default = "00", help = "Sequence number")
    parser.add_argument('--min_frame', type=int, default = -1, help = "Min frame")
    parser.add_argument('--max_frame', type=int, default = -1, help = "Max frame")
    
    parser.add_argument('--to_the_end', action='store_true', help = "Search to the end frame not only marked in dataset")
    
    parser.add_argument('--min_object_area_px', type=int, default = 50, help = "Min object mask area in pixels")
    parser.add_argument('--max_object_dist_m', type=float, default = 50, help = "Max dist to object in meters")
    
    parser.add_argument('--iou_cost_th', type=float, default = 0.2, help = "IoU threshold for mask and rect matching")
    parser.add_argument('--min_lidar_intensity', type=float, default = 0.2, help = "Min lidar intesity to use")
    parser.add_argument('--min_lidar_points', type=int, default = 10, help = "Min lidar number of points on object")
    
    parser.add_argument('--do_clip', action='store_true', help = "Do or not CLIP extraction")        
    parser.add_argument('--objects_ignore', nargs='+', default=[], help = "Additional objects to ignore")        
    
    parser.add_argument('--save_map_unlabeled', action='store_true', help = "Save map with all objects")
    parser.add_argument('--save_map_labeled', action='store_true', help = "Save map with all objects labeled")
    
    parser.add_argument('--video_cam', action='store_true', help = "Save video path from camera")
    parser.add_argument('--video_map', action='store_true', help = "Save video path as map")
    parser.add_argument('--video_mix', action='store_true', help = "Save mixed video path of camera and map")
    
    parser.add_argument('--plot_invisible', action='store_true', help = "Save mixed video path of camera and map")
        
    
    args = parser.parse_args()    
    dict_args = vars(args)
    print(dict_args)
    ###    
    SMSE = SemanticMapAndScenesExtractor(**dict_args)
        
    SMSE.read_stuff()    
    
    if args.save_map_labeled:
        SMSE.plot_2d_map(True)
        plt.savefig(SMSE.save_path + "/map_labeled.png", dpi=1200)
    if args.save_map_unlabeled:
        SMSE.plot_2d_map(False)
        plt.savefig(SMSE.save_path + "/map_unlebeled.png", dpi=1200)
            
    with open(SMSE.save_path + "/export_params.yaml", "w") as file:
        yaml.dump(dict_args, file)               

        
