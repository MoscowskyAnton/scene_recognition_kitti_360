#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
import argparse
import os
import yaml

def substract_angles(target, source):
    return np.arctan2(np.sin(target-source), np.cos(target-source))

class SemanticMapDevider(object):
    
    def __init__(self, ignore_labels = []):
        self.ignore_labels = ignore_labels
        
        
    def load_map_from_csv(self, path):
        self.df = pd.read_csv(path + '/semantic_map.csv', index_col=0)  
        self.path = path        
        self.MAP = self.df[~self.df['class'].isin(self.ignore_labels)][['gid', 'x', 'y', 'z']].values                
                    
    
    def devide_map(self, submap_max_size, increase_range = 50, dir_diag_size = 16):        
        n_clusters = int(np.ceil(self.MAP.shape[0] / submap_max_size ))
              
        clustering = AgglomerativeClustering(n_clusters = n_clusters)                                                    
        clustering.fit(self.MAP[:, 1:3])                
        
        self.labels = clustering.labels_        
        # recluster bigs
        for i in set(self.labels):
            if i != -1:
                cluster_ind = (self.labels == i)
                size = np.count_nonzero(cluster_ind)
                if size >= submap_max_size:
                    ec = AgglomerativeClustering(n_clusters = int(np.ceil(size / submap_max_size)))
                    ec.fit(self.MAP[cluster_ind, 1:3])
                    self.labels[cluster_ind] = np.array(ec.labels_) + np.max(self.labels) + 1
                                                                     
        # calc stuff and expand
        self.clusters = []
        for i in set(self.labels):
            if i != -1:                                                                
                cluster_ind = (self.labels == i)
                if not np.count_nonzero(cluster_ind):
                    continue
                    
                cluster = {'base_center': (np.mean(self.MAP[cluster_ind, 1]), np.mean(self.MAP[cluster_ind, 2]))}
                
                cluster_gids = self.MAP[cluster_ind, 0]                
                cluster['base_labels'] = cluster_gids
                
                dx = self.MAP[cluster_ind, 1] - cluster['base_center'][0]
                dy = self.MAP[cluster_ind, 2] - cluster['base_center'][1]
                
                all_r = np.hypot(dx, dy)
                all_angles = np.arctan2(dy, dx)
                #print(all_angles)
                
                cluster['max_rs'] = []
                angle_step = a1 = 2*np.pi/dir_diag_size
                for i in range(dir_diag_size):
                    a1 = -np.pi + angle_step * i
                    a2 = -np.pi + angle_step * (i+1)
                    
                    selected_ind = np.where(np.logical_and(all_angles >= a1, all_angles < a2))
                    if np.count_nonzero(selected_ind) == 0:
                        cluster['max_rs'].append(0)
                    else:
                        max_r = np.max(all_r[selected_ind])
                        cluster['max_rs'].append(max_r)
                    
                other_ind = ~cluster_ind
                other_dx = self.MAP[other_ind, 1] - cluster['base_center'][0]
                other_dy = self.MAP[other_ind, 2] - cluster['base_center'][1]
                other_all_angles = np.arctan2(other_dy, other_dx)
                other_r = np.hypot(other_dx, other_dy)
                
                cluster['add_labels'] = []
                for i, max_r in enumerate(cluster['max_rs']):
                    a1 = -np.pi + angle_step * i
                    a2 = -np.pi + angle_step * (i+1)
                    selected_ind = np.where(np.logical_and(other_all_angles >= a1, other_all_angles < a2))
                    
                    if np.count_nonzero(selected_ind) == 0:
                        continue
                    
                    more_selected = other_r[selected_ind] < (max_r + increase_range)
                    cluster['add_labels'] += ((self.MAP[other_ind, 0])[selected_ind])[more_selected].tolist()
                    
                cluster['add_labels'] = list(set(cluster['add_labels']))                                    
                cluster['full_labels'] = np.array((cluster['base_labels'].tolist() + cluster['add_labels'])).astype(int)
                    
                self.clusters.append(cluster)
            else:
                print("Warn! There are notclustered objects!")
        
        os.makedirs(self.path + "/submaps", exist_ok=True)
        for i, cluster in enumerate(self.clusters):
            df = self.df[self.df['gid'].isin(cluster['full_labels'])]
            df.to_csv(self.path + f'/submaps/semantic_submap_{i}.csv', index = False)
            
        
        new_size = sum([cluster['full_labels'].shape[0] for cluster in self.clusters])
        print(f"Size incrteased to {new_size} from {self.MAP.shape[0]} (+{int(100*new_size/self.MAP.shape[0] - 100)}%)")
    
    
    def get_inspected_objects(self):
        inspected_objects = {}
        for obj_i in range(self.MAP.shape[0]):
            gid = int(self.MAP[obj_i, 0])            
            inspected_objects[gid] = []
            for i, cluster in enumerate(self.clusters):
                if np.isin(gid, cluster['full_labels']):
                    inspected_objects[gid].append(i)
                    
        multicluster = {}
        for gid, clusters in inspected_objects.items():
            if len(clusters) > 1:                
                key = tuple(sorted(clusters))
                
                if not key in multicluster:
                    multicluster[key] = []
                multicluster[key].append(gid)
                    
        return multicluster            
                                
    
    def plot_divided_map(self):        
        ins_obj = self.get_inspected_objects()

        x = int(np.sqrt(len(self.clusters)))
        y = int(np.ceil(len(self.clusters) / x))        
        
        fig, axs = plt.subplots(x, y, squeeze = False, figsize=(15, 9))        
        x_ = 0
        y_ = 0
        for ic, cluster in enumerate(self.clusters):
            ax = axs[x_, y_]
            cluster_ind = np.isin(element = self.MAP[:,0], test_elements = cluster['full_labels'].tolist() )
            
            ax.plot(self.MAP[~cluster_ind, 1], self.MAP[~cluster_ind, 2], '.', color = 'grey', alpha = 1)
            ax.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.')#, label = f'base ({ic})', alpha = 1)                                                    
            
            for mc, gids in ins_obj.items():
                if ic in mc:
                    mcluster_ind = np.isin(element = self.MAP[:,0], test_elements = gids )
                    
                    key = ''.join(f'{x}' for x in mc if x != ic)
                    
                    ax.plot(self.MAP[mcluster_ind, 1], self.MAP[mcluster_ind, 2], '.', label = f'w/{key}', alpha = 1)                                                    
                        
            ax.set_aspect('equal', adjustable='box')            
            ax.set_title(f"#{ic} ({np.count_nonzero(cluster_ind)})")            
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))      
            ax.grid()
            ax.set_xticks([])
            ax.set_yticks([])
            
            y_ += 1
            if y_ == y:
                y_ = 0
                x_ += 1
                
        while x_ != x:            
            ax = axs[x_, y_]
            ax.set_axis_off()
            y_ += 1
            if y_ == y:
                y_ = 0
                x_ += 1
                if x_ == x:
                    break
                
        new_size = sum([cluster['full_labels'].shape[0] for cluster in self.clusters])
        plt.suptitle(f"Map divided with intersection (+{int(100*new_size/self.MAP.shape[0] - 100)}%)")        
        plt.savefig(self.path + f"/submaps/submaps.png", dpi = 500)                
        
        
    def plot_predivided_map(self):
        plt.figure('map divided')
        plt.title('Map predivided (initial clusters and direction diagrams)')
        
        cluster_ind = (self.labels == -1)
        plt.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.', label = f'NC {np.count_nonzero(cluster_ind)}', color = 'black')
                                    
        for i, cluster in enumerate(self.clusters):
            
            color = plt.get_cmap('tab10', 10)(i%10)
            
            cluster_ind = np.isin(element = self.MAP[:,0], test_elements = cluster['base_labels'].tolist() )
            
            plt.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.', label = f'{i}){np.count_nonzero(cluster_ind)}', color = color)
            
            plt.plot(cluster['base_center'][0], cluster['base_center'][1], 'o', color = 'black' )
            plt.plot(cluster['base_center'][0], cluster['base_center'][1], '.', color = color )
            
            step = 2*np.pi / len(cluster['max_rs'])
            for i, max_r in enumerate(cluster['max_rs']):
                angle = -np.pi + step * (i + 0.5)
                plt.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                         [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], '-', color = 'k')
                
                plt.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                         [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], ':', color = color)
                
        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))      
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(self.path + f"/submaps/predivided.png")                        
    
    
    def plot_map(self):        
        plt.figure('full_map')
        plt.title('Full Map')
        plt.plot(self.MAP[:, 1], self.MAP[:, 2], '.')
        plt.grid()       
        
        
    def plot_images_for_schema(self, increase_range):
        
        fig, axs = plt.subplots(1, 5, squeeze = True)        
        map_ax, cluster_ax, dir_diag_ax, increase_ax, final_ax = axs
        
        for ax in axs:
            ax.set_aspect('equal', adjustable='box')                        
            ax.set_xticks([])
            ax.set_yticks([])
        
        # just MAP
        #map_ax.set_title("Full Semantic Map")
        map_ax.set_title("Полная семантическая карта")                
        map_ax.plot(self.MAP[:, 1], self.MAP[:, 2], '.', color = 'grey')
        
        # clusters and 
        #cluster_ax.set_title("Initial clusterization,\n obtaining centroids")
        cluster_ax.set_title("Начальная кластеризация")
        def col(i):
            return plt.get_cmap('tab10', 10)(i%10)
        
        for i, cluster in enumerate(self.clusters):
                        
            cluster_ind = np.isin(element = self.MAP[:,0], test_elements = cluster['base_labels'].tolist() )
            
            color = col(i)
            cluster_ax.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.', color = color)
            cluster_ax.plot(cluster['base_center'][0], cluster['base_center'][1], 'o', color = 'black' )
            cluster_ax.plot(cluster['base_center'][0], cluster['base_center'][1], '.', color = color )
        
        # directional diagrams
        #dir_diag_ax.set_title("Direction diagrams calculation")
        dir_diag_ax.set_title("Построение диаграмм направленности")
        
        for i, cluster in enumerate(self.clusters):            
            cluster_ind = np.isin(element = self.MAP[:,0], test_elements = cluster['base_labels'].tolist() )            
            color = col(i)
            
            dir_diag_ax.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.', color = color, alpha = 0.25)
            
            step = 2*np.pi / len(cluster['max_rs'])
            for i, max_r in enumerate(cluster['max_rs']):
                angle = -np.pi + step * (i + 0.5)
                dir_diag_ax.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                         [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], '-', color = 'k')
                
                dir_diag_ax.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                         [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], ':', color = color)
                
        # increase!
        #increase_ax.set_title("Diagrams increase")
        increase_ax.set_title("Расширение диаграмм")
        
        increase_ax.plot(self.MAP[:, 1], self.MAP[:, 2], '.', color = 'grey', alpha = 0.25)
        
        for i, cluster in enumerate(self.clusters):            
            step = 2*np.pi / len(cluster['max_rs'])
            color = col(i)
            for j, max_r in enumerate(cluster['max_rs']):
                angle = -np.pi + step * (j + 0.5)
                increase_ax.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                         [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], ':', color = 'k')                                
                
                increase_ax.arrow(cluster['base_center'][0] + np.cos(angle) * max_r, cluster['base_center'][1] + np.sin(angle) * max_r, increase_range * np.cos(angle), increase_range * np.sin(angle), color = color, head_width = 1)
                
        # add
        CL = 1
        
        #final_ax.set_title("Adding extra objects")
        final_ax.set_title("Добавление новых объектов")
        
        cluster = self.clusters[CL]
        
        final_ax.plot(self.MAP[:, 1], self.MAP[:, 2], '.', color = 'grey', alpha = 1)
        
        cluster_ind = np.isin(element = self.MAP[:,0], test_elements = cluster['base_labels'].tolist() )
        color = col(CL)
        final_ax.plot(self.MAP[cluster_ind, 1], self.MAP[cluster_ind, 2], '.', color = color)
        
        
        for j, max_r in enumerate(cluster['max_rs']):
            angle = -np.pi + step * (j + 0.5)
            final_ax.plot([cluster['base_center'][0], cluster['base_center'][0] + np.cos(angle) * max_r],
                        [cluster['base_center'][1], cluster['base_center'][1] + np.sin(angle) * max_r], ':', color = 'k')
                        
            
            final_ax.arrow(cluster['base_center'][0] + np.cos(angle) * max_r, cluster['base_center'][1] + np.sin(angle) * max_r, increase_range * np.cos(angle), increase_range * np.sin(angle), color = color, head_width = 1)
        
        for i, cluster_ in enumerate(self.clusters):            
            if i != CL:
                for gid in cluster_['base_labels']:
                    if gid in cluster['full_labels']:
                        obj_ind = np.where(self.MAP[:, 0] == gid)
                        #print(obj_ind)
                        final_ax.plot(self.MAP[obj_ind, 1], self.MAP[obj_ind, 2], '.', color = col(i))
        
        

if __name__ == "__main__":      
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str, required = True, help = "Path to map FOLDER")
    parser.add_argument('--submap_max_size', type=int, default = 300, help = "Desireble size of submaps")
    parser.add_argument('--increase_range', type=float, default = 50, help = "Lidar range in meters")
    parser.add_argument('--dir_diag_size', type=int, default = 32, help = "Size of direction diagrams")
    parser.add_argument('--objects_ignore', nargs='+', default=[], help = "Objects to ignore")        
        
    args = parser.parse_args()    
    
    smd = SemanticMapDevider(ignore_labels = args.objects_ignore)    
    smd.load_map_from_csv(args.path)
                
    smd.devide_map(submap_max_size = args.submap_max_size, increase_range = args.increase_range, dir_diag_size = args.dir_diag_size)
    
    smd.plot_predivided_map()
    smd.plot_divided_map()
    
    dict_args = vars(args)
    
    print(f"Saving data to {args.path}/submaps ...")
    with open(args.path + "/submaps/export_params.yaml", "w") as file:
        del dict_args['path']
        yaml.dump(dict_args, file)
    print("Done!")
        
    #smd.plot_images_for_schema(args.increase_range)    
    #plt.show()
