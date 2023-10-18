# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:17:20 2022

@author: debby
"""
import math
import itertools
import os
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import open3d as o3d
import hickle as hkl
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
import keras_tuner as kt
from scipy.stats import pearsonr

######################### input part ############################################
def v_center_coordinate(idx = [0, 0, 0], box_size = 20, resolution = 1):
    '''
    Get the coordinates of a voxel center
    Parameters:
        idx - index of the queries voxel
        box_size - side length (A) of box region
        resolution - grid resolution (A)
    '''
    dtsz = math.ceil(box_size / resolution) + 1
    if (idx[0] not in range(dtsz)) or (idx[1] not in range(dtsz)) or (idx[2] not in range(dtsz)):
#        print('Illegal index of voxels!')
        return []
    else:
        coor = -box_size/2 + np.array(idx) * resolution
        return coor

def point_dist(p1, p2):
    return math.sqrt( ((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2) + ((p1[2]-p2[2])**2) )

class Voxelization(object):
    def __init__(self, fn_atomprop, ID, labels, 
                 rot_alpha_int = 0, 
                 rot_beta_int = 0, 
                 rot_gamma_int = 0):
        """
        Initialize a voxelization class.
         *** rotation matrix concerns three Euler angles alpha, beta and gamma ***
         alpha - represents a rotation around the z axis ([0, 2pi))
         beta - represents a rotation around the x axis or N axis ([0, pi])
         gamma - represents a rotation around the Z axis (new z axis)  ([0, 2pi))

        Parameters:
            fn_atomprop - path to the file storing the atomic properties of the target complex
            ID - ID for complex
            labels - dataframe storing the IDs and affinities (labels) for the complexes
            rot_alpha_int, rot_beta_int, rot_gamma_int - intervals for rotating the euler angles
        """
        self.ID = ID
        print('Constructing a voxelization object for %s.........\n' % self.ID)
        # read in data
        lb = labels.loc[labels['id'] == ID,'affinity']
        self.label = lb[lb.index[0]] # lb is a data frame, lb.index[0] finds the index of its first row
            
        if os.path.isfile(fn_atomprop):
            # read in data
            rawAP = pd.read_csv(fn_atomprop)
            self.AP = rawAP[rawAP['atmnum'] > 1] # IGNORE hydrogen atoms
            points = np.array([[x, y, z] for (x, y, z) in zip(self.AP['x'].tolist(), self.AP['y'].tolist(), self.AP['z'].tolist())])
            self.pointslst = []
            # check alpha value ---------------------------------------------------------------------
            if 0 < rot_alpha_int < 2*np.pi:
                alphalst = np.arange(0, 2*np.pi, rot_alpha_int)
            elif rot_alpha_int == 0:
                alphalst = np.array([0])
            else:
                print('Wrong alpha value for rotation!')
            # check beta value ---------------------------------------------------------------------
            if 0 < rot_beta_int < np.pi:
                betalst = np.arange(0, np.pi, rot_beta_int)
            elif rot_beta_int == 0:
                betalst = np.array([0])
            else:
                print('Wrong beta value for rotation!')
            # check gamma value ---------------------------------------------------------------------
            if 0 < rot_gamma_int < 2*np.pi:
                gammalst = np.arange(0, 2*np.pi, rot_gamma_int)
            elif rot_gamma_int == 0:
                gammalst = np.array([0])
            else:
                print('Wrong gamma value for rotation!')
            # rotate the point cloud ---------------------------------------------------------------------
            self.eulerangles = list(itertools.product(alphalst, betalst, gammalst))
            for angles in self.eulerangles:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                R = pcd.get_rotation_matrix_from_xyz(angles)
                # R = pcd.get_rotation_matrix_from_quaternion(euler_to_quaternion(angles[0], angles[1], angles[2]))
                pcd = pcd.rotate(R, center = (0, 0, 0)) # the rotation center is placed on (0, 0, 0), which is the coordinate center of ligand (heavy atoms)
                # pcd = pcd.rotate(R, center = pcd.get_center()) # .get_center() returns the center of point cloud pcd
                self.pointslst.append(np.asarray(pcd.points))

    def descriptors_pufnacy(self, box_size = 20, resolution = 1, prtcmm = 1,
                            featureset = ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
                                          'hybridization', 'heavyneighbors', 'heteroneighbors', 
                                          'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring', 
                                          'partialCH', 'moltype']):
        # extract descriptors as described in the pufnacy work
        voxels_pufnacy = []
        pointslst_filt_pufnacy = []
        tensors = []
        labels = []
        conflicts = []
        dtsz = math.ceil(box_size / resolution) + 1
        crnpoints = [list(i) for i in list(itertools.product([-box_size/2, box_size/2], [-box_size/2, box_size/2], [-box_size/2, box_size/2]))]
        for ind in range(len(self.pointslst)):
            print('No.%d conformations -------------------------------' % (ind + 1))
            labels.append(self.label)
            # 1. filter point cloud ----------------------------------------------------
            filt1 = [all(i) for i in self.pointslst[ind] <= box_size/2] 
            filt2 = [all(i) for i in self.pointslst[ind] >= -box_size/2]
            filt = [all((x,y)) for (x,y) in zip(filt1, filt2)] # truncate the interaction area (cube with side size of box_size)
            pointslst_filt_pufnacy.append(self.pointslst[ind][filt])
            APtmp = self.AP.loc[filt]
            proatmnum = APtmp['moltype'].tolist().count(0)
            ligatmnum = APtmp['moltype'].tolist().count(1)
            # 2. add corner atoms ------------------------------------------------------
            # filtpoints = [[x, y, z] for (x, y, z) in zip(APtmp['x'].tolist(), APtmp['y'].tolist(), APtmp['z'].tolist())]
            filtpoints = pointslst_filt_pufnacy[ind].tolist()
            addpoints = [i for i in crnpoints if i not in filtpoints]
            allpoints = np.array(filtpoints + addpoints)
            # 3. generate voxels ------------------------------------------------------
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(allpoints)
            cl = np.zeros_like(allpoints)
            cl[:proatmnum, 0] = 0.5 # protein atoms, red
            cl[proatmnum:(proatmnum + ligatmnum), 1] = 0.5 # ligand atoms, green
            cl[(proatmnum + ligatmnum):, ] = 1 # corner points
            # 4. display voxels ------------------------------------------------
            # cl[np.where(np.array(APtmp['negionizable'].tolist()) == 1), 1] = 0.5
            pcd.colors = o3d.utility.Vector3dVector(cl)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
            voxels_pufnacy.append(voxel_grid)
            # o3d.visualization.draw_geometries([voxel_grid])
            # 5. check conflicts in voxels --------------------------------------
            ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
            for fpt in filtpoints:
                ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
            conflict_dic = {m:n for (m, n) in ckdic.items() if n > 1}
            conflicts.append(sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
            #6. fill in voxel values --------------------------------------------------
            if prtcmm == 0:
                curdt = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset)))            
                for vind in range(len(filtpoints)):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind]))
                    curdt[vx] = curdt[vx] + APtmp.iloc[vind][featureset] # accumulate all atoms in a single voxel
                tensors.append(np.expand_dims(curdt, axis = 0))
                # .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * dtsz * dtsz * dtsz * feasize
            elif prtcmm == 1:
                curdt = np.zeros(shape = (dtsz, dtsz, dtsz, (len(featureset) - 1) * 2))
                cur_feaset = [i for i in featureset if i != 'moltype']
                for vind in range(proatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind]))
                    curdt[vx][:(len(featureset) - 1)] = curdt[vx][:(len(featureset) - 1)] + APtmp.iloc[vind][cur_feaset] # accumulate all atoms in a single voxel
                for vind in range(ligatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind + proatmnum]))
                    curdt[vx][(len(featureset) - 1):] = curdt[vx][(len(featureset) - 1):] + APtmp.iloc[vind + proatmnum][cur_feaset] # accumulate all atoms in a single voxel
                tensors.append(np.expand_dims(curdt, axis = 0))
                # .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * dtsz * dtsz * dtsz * 2(feasize-1)
            elif prtcmm == 2:
                curdt_pro = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) - 1))
                curdt_lig = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) - 1))
                cur_feaset = [i for i in featureset if i != 'moltype']
                for vind in range(proatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind]))
                    curdt_pro[vx] = curdt_pro[vx] + APtmp.iloc[vind][cur_feaset]
                for vind in range(ligatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind + proatmnum]))
                    curdt_lig[vx] = curdt_lig[vx] + APtmp.iloc[vind + proatmnum][cur_feaset]
                curdt = np.vstack([np.expand_dims(curdt_pro, axis = 0), np.expand_dims(curdt_lig, axis = 0)]) # for thermodynamics cycle
                tensors.append(np.expand_dims(curdt, axis = 0))
                # .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * 2 * dtsz * dtsz * dtsz * (feasize-1)
            else:
                curdt_pro = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) - 1))
                curdt_lig = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) - 1))
                curdt_com = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) - 1))
                cur_feaset = [i for i in featureset if i != 'moltype']
                for vind in range(proatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind]))
                    curdt_pro[vx] = curdt_pro[vx] + APtmp.iloc[vind][cur_feaset]
                    curdt_com[vx] = curdt_com[vx] + APtmp.iloc[vind][cur_feaset]
                for vind in range(ligatmnum):
                    vx = tuple(voxel_grid.get_voxel(filtpoints[vind + proatmnum]))
                    curdt_lig[vx] = curdt_lig[vx] + APtmp.iloc[vind + proatmnum][cur_feaset]
                    curdt_com[vx] = curdt_com[vx] + APtmp.iloc[vind + proatmnum][cur_feaset]
                curdt = np.vstack([np.expand_dims(curdt_com, axis = 0), np.expand_dims(curdt_pro, axis = 0), np.expand_dims(curdt_lig, axis = 0)]) # for thermodynamics cycle
                tensors.append(np.expand_dims(curdt, axis = 0))
                # .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * 3 * dtsz * dtsz * dtsz * (feasize-1)
                
        return (tensors, labels, conflicts, voxels_pufnacy, pointslst_filt_pufnacy)

    def descriptors_kdeep(self, box_size = 24, resolution = 1, prtcmm = 1,
                          featureset = ['hydrophobic', 'donor', 'acceptor', 'aromatic', 'posionizable',
                                        'negionizable', 'atmMetallic', 'exlvolume']):
        '''
        extract descriptors as described in the kdeep work
        Parameters:
            box_size - side length (A) of box region
            resolution - grid resolution (A)
            voxel_ext - considering extended voxels
            featureset - for generating descriptors
        '''    
        tensors = []
        labels = []
        conflicts = []
        dtsz = math.ceil(box_size / resolution) + 1
        add_for_iter = np.arange(-box_size/2, box_size/2 + 1, resolution)
        addpoints = [list(i) for i in list(itertools.product(add_for_iter, add_for_iter, add_for_iter))]
        for ind in range(len(self.pointslst)):
            print('No.%d conformations -------------------------------' % (ind + 1))
            labels.append(self.label)
            # 1. filter point cloud ----------------------------------------------------
            filt1 = [all(i) for i in self.pointslst[ind] <= box_size/2] 
            filt2 = [all(i) for i in self.pointslst[ind] >= -box_size/2]
            filt = [all((x,y)) for (x,y) in zip(filt1, filt2)]
            APtmp = self.AP.loc[filt]
            proatmnum = APtmp['moltype'].tolist().count(0)
            prtcmm_AP = [APtmp[:proatmnum], APtmp[proatmnum:]]
            filtpoints = self.pointslst[ind][filt].tolist()
            prtcmm_filtpoints = [filtpoints[:proatmnum], filtpoints[proatmnum:]]
            allpoints = np.array(filtpoints + addpoints)
            prtcmm_allpoints = [np.array(curfiltpoints + addpoints) for curfiltpoints in prtcmm_filtpoints]
            # 2. proteo-chemometric or not ---------------------------------------------
            if prtcmm == 0:
                # 3.1. generate voxels and get their center coordinates ----------------------
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(allpoints)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
                vx_gridinds = np.vstack([curvx.grid_index for curvx in voxel_grid.get_voxels()])
                vx_centers = np.vstack([v_center_coordinate(list(curvx.grid_index), box_size = box_size, resolution = resolution) for curvx in voxel_grid.get_voxels()])
                # 3.2. check conflicts in voxels --------------------------------------
                ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
                for fpt in filtpoints:
                    ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
                conflict_dic = {m:n for (m, n) in ckdic.items() if n > 1}
                conflicts.append(sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
                # 3.3. fill in voxel information ----------------------------------------
                curdt = np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset) + 1))         
                # compute comtributions of each atom to the voxels ---------------------
                pdist = cdist(filtpoints, vx_centers, metric = 'euclidean')  # shape of len(filtpoints) * numOfVoxels                  
                contributions = 1 - np.exp(-(np.array(APtmp['vdwrad'])[:, np.newaxis] / pdist) ** 12)
                ## contributions of atoms to voxels, shape of len(filtpoints) * numOfVoxels
                ## np.array(APtmp['vdwrad'])[:, np.newaxis] returns an array of len(filtpoints) * 1
                for vind in range(len(vx_gridinds)):
                    vx = tuple(vx_gridinds[vind])
                    newval = np.sum(APtmp[featureset + ['moltype']].to_numpy() * contributions[:, vind][:, np.newaxis], axis = 0)
                    ## (1) APtmp[featureset + ['moltype']].to_numpy(): len(filtpoints) * (feasize + 1)
                    ## (2) contributions[:, vind][:, np.newaxis]: len(filtpoints) * 1
                    ## sum((1) * (2)) -> shape of (feasize + 1,)
                    curdt[vx] = newval
                tensors.append(np.expand_dims(curdt, axis = 0))
                ## .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * dtsz * dtsz * dtsz * (feasize+1)
            elif prtcmm == 1:
                curdt = [np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset))), np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset)))]          
                tmp_conflicts = 0
                # consider proteins and ligands separately -----------------------------
                for prtcmm_ind in range(2):
                    # 4.1. generate voxels and get their center coordinates ----------------------
                    cur_points = prtcmm_filtpoints[prtcmm_ind]
                    cur_AP = prtcmm_AP[prtcmm_ind]
                    cur_allpoints = prtcmm_allpoints[prtcmm_ind]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_allpoints)
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
                    vx_gridinds = np.vstack([curvx.grid_index for curvx in voxel_grid.get_voxels()])
                    vx_centers = np.vstack([v_center_coordinate(list(curvx.grid_index), box_size = box_size, resolution = resolution) for curvx in voxel_grid.get_voxels()])
                    # 4.2. check conflicts in voxels --------------------------------------
                    ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
                    for fpt in cur_points:
                        ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
                    conflict_dic = {m:n for (m, n) in ckdic.items() if n >= 1}
                    tmp_conflicts += (sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
                    ## because protein and ligand are in different channels, so the conflicts are counted separately and then accumulated
                    # 4.3. fill in voxel information ----------------------------------------
                    pdist = cdist(cur_points, vx_centers, metric = 'euclidean')                    
                    contributions = 1 - np.exp(-(np.array(cur_AP['vdwrad'])[:, np.newaxis] / pdist) ** 12)
                    for vind in range(len(vx_gridinds)):
                        vx = tuple(vx_gridinds[vind])
                        newval = np.sum(cur_AP[featureset].to_numpy() * contributions[:, vind][:, np.newaxis], axis = 0)
                        curdt[prtcmm_ind][vx] = newval                        
                tensors.append(np.expand_dims(np.concatenate(curdt, axis = 3), axis = 0))
                ## np.concatenate(curdt, axis = 3) returns a tensor of dtsz*dtsz*dtsz*2feasize
                ## .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * dtsz * dtsz * dtsz * 2feasize
                conflicts.append(tmp_conflicts)
            elif prtcmm == 2:
                curdt = [np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset))), np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset)))]          
                tmp_conflicts = 0
                # 5.1. process protein and ligand -------------------------------------------------------
                for prtcmm_ind in range(2):
                    cur_points = prtcmm_filtpoints[prtcmm_ind]
                    cur_AP = prtcmm_AP[prtcmm_ind]
                    cur_allpoints = prtcmm_allpoints[prtcmm_ind]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_allpoints)
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
                    vx_gridinds = np.vstack([curvx.grid_index for curvx in voxel_grid.get_voxels()])
                    vx_centers = np.vstack([v_center_coordinate(list(curvx.grid_index), box_size = box_size, resolution = resolution) for curvx in voxel_grid.get_voxels()])
                    ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
                    for fpt in cur_points:
                        ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
                    conflict_dic = {m:n for (m, n) in ckdic.items() if n >= 1}
                    tmp_conflicts += (sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
                    pdist = cdist(cur_points, vx_centers, metric = 'euclidean')                    
                    contributions = 1 - np.exp(-(np.array(cur_AP['vdwrad'])[:, np.newaxis] / pdist) ** 12)
                    for vind in range(len(vx_gridinds)):
                        vx = tuple(vx_gridinds[vind])
                        newval = np.sum(cur_AP[featureset].to_numpy() * contributions[:, vind][:, np.newaxis], axis = 0)
                        curdt[prtcmm_ind][vx] = newval                        
                curdt_new = np.vstack([np.expand_dims(curdt[0], axis = 0), np.expand_dims(curdt[1], axis = 0)]) 
                tensors.append(np.expand_dims(curdt_new, axis = 0))
                conflicts.append(tmp_conflicts)
            else:
                curdt = [np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset))), np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset))), np.zeros(shape = (dtsz, dtsz, dtsz, len(featureset)))]          
                tmp_conflicts = 0
                # 5.1. process complex ------------------------------------------------------------------
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(allpoints)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
                vx_gridinds = np.vstack([curvx.grid_index for curvx in voxel_grid.get_voxels()])
                vx_centers = np.vstack([v_center_coordinate(list(curvx.grid_index), box_size = box_size, resolution = resolution) for curvx in voxel_grid.get_voxels()])
                ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
                for fpt in filtpoints:
                    ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
                conflict_dic = {m:n for (m, n) in ckdic.items() if n > 1}
                tmp_conflicts += (sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
                pdist = cdist(filtpoints, vx_centers, metric = 'euclidean')  # shape of len(filtpoints) * numOfVoxels                  
                contributions = 1 - np.exp(-(np.array(APtmp['vdwrad'])[:, np.newaxis] / pdist) ** 12)
                for vind in range(len(vx_gridinds)):
                    vx = tuple(vx_gridinds[vind])
                    newval = np.sum(APtmp[featureset].to_numpy() * contributions[:, vind][:, np.newaxis], axis = 0)
                    curdt[0][vx] = newval
                # 5.2. process protein and ligand -------------------------------------------------------
                for prtcmm_ind in range(2):
                    cur_points = prtcmm_filtpoints[prtcmm_ind]
                    cur_AP = prtcmm_AP[prtcmm_ind]
                    cur_allpoints = prtcmm_allpoints[prtcmm_ind]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_allpoints)
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
                    vx_gridinds = np.vstack([curvx.grid_index for curvx in voxel_grid.get_voxels()])
                    vx_centers = np.vstack([v_center_coordinate(list(curvx.grid_index), box_size = box_size, resolution = resolution) for curvx in voxel_grid.get_voxels()])
                    ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
                    for fpt in cur_points:
                        ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
                    conflict_dic = {m:n for (m, n) in ckdic.items() if n >= 1}
                    tmp_conflicts += (sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
                    pdist = cdist(cur_points, vx_centers, metric = 'euclidean')                    
                    contributions = 1 - np.exp(-(np.array(cur_AP['vdwrad'])[:, np.newaxis] / pdist) ** 12)
                    for vind in range(len(vx_gridinds)):
                        vx = tuple(vx_gridinds[vind])
                        newval = np.sum(cur_AP[featureset].to_numpy() * contributions[:, vind][:, np.newaxis], axis = 0)
                        curdt[prtcmm_ind+1][vx] = newval                        
                curdt_new = np.vstack([np.expand_dims(curdt[0], axis = 0), np.expand_dims(curdt[1], axis = 0), np.expand_dims(curdt[2], axis = 0)]) # for thermodynamics cycle
                tensors.append(np.expand_dims(curdt_new, axis = 0))
                conflicts.append(tmp_conflicts)
                     
        return (tensors, labels, conflicts)

    def descriptors_sfcnn(self, box_size = 20, resolution = 1, 
                          atmtps_pro = [6, 7, 8, 16], 
                          atmtps_lig = [6, 7, 8, 9, 15, 16, 17, 35, 53]):
        # extract descriptors as described in the sfcnn work
        # using a one-hot-encoding way to extract atom-type features for each voxel
        tensors = []
        labels = []
        conflicts = []
        dtsz = math.ceil(box_size / resolution) + 1
        crnpoints = [list(i) for i in list(itertools.product([-box_size/2, box_size/2], [-box_size/2, box_size/2], [-box_size/2, box_size/2]))]
        for ind in range(len(self.pointslst)):
            print('No.%d conformations -------------------------------' % (ind + 1))
            labels.append(self.label)
            # 1. filter point cloud ----------------------------------------------------
            filt1 = [all(i) for i in self.pointslst[ind] <= box_size/2] 
            filt2 = [all(i) for i in self.pointslst[ind] >= -box_size/2]
            filt = [all((x,y)) for (x,y) in zip(filt1, filt2)] # truncate the interaction area (cube with side size of box_size)
            APtmp = self.AP.loc[filt]
            # 2. add corner atoms ------------------------------------------------------
            # filtpoints = [[x, y, z] for (x, y, z) in zip(APtmp['x'].tolist(), APtmp['y'].tolist(), APtmp['z'].tolist())]
            filtpoints = self.pointslst[ind][filt].tolist()
            addpoints = [i for i in crnpoints if i not in filtpoints]
            allpoints = np.array(filtpoints + addpoints)
            # 3. generate voxels ------------------------------------------------------
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(allpoints)
            # 4. display voxels ------------------------------------------------
            # cl[np.where(np.array(APtmp['negionizable'].tolist()) == 1), 1] = 0.5
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = resolution)
            # 5. check conflicts in voxels --------------------------------------
            ckdic = {tuple(curvx.grid_index):0 for curvx in voxel_grid.get_voxels()}
            for fpt in filtpoints:
                ckdic[tuple(voxel_grid.get_voxel(fpt))] += 1
            conflict_dic = {m:n for (m, n) in ckdic.items() if n > 1}
            conflicts.append(sum(list(conflict_dic.values())) - len(list(conflict_dic.keys())))
            #6. fill in voxel values --------------------------------------------------
            curdt = np.zeros(shape = (dtsz, dtsz, dtsz, len(atmtps_pro) + len(atmtps_lig)))            
            for vind in range(len(filtpoints)):
                vx = tuple(voxel_grid.get_voxel(filtpoints[vind]))
                atp = APtmp.iloc[vind]['atmnum']
                moltp = APtmp.iloc[vind]['moltype']
                if moltp == 0 and atp in atmtps_pro:
                    channel = atmtps_pro.index(atp)
                    curdt[vx][channel] += 1 
                elif moltp == 1 and atp in atmtps_lig:
                    channel = atmtps_lig.index(atp) + len(atmtps_pro)
                    curdt[vx][channel] += 1 
                else:
                    channel = -1   
            tensors.append(np.expand_dims(curdt, axis = 0))
            # .expand_dims tries to collect all the conformers, final dim for each conformer: 1 * dtsz * dtsz * dtsz * (len(atmtps_pro) + len(atmtps_lig))

        return (tensors, labels, conflicts)

def feature_engineering(ids, fn_labels, dt_folder,
                        para = {'feat_type': 'pufnacy', 
                                'rot_ang_steps': [0, 0, 0], 
                                'box_size': 20, 'resolution': 1, 'prtcmm': 0}):
    '''
    Extrat features for BAP.
    Parameters:
        ids - a list of samples (PDB IDs) for feature extraction
        fn_labels - path to the index file (in 'indexes' folder)
        dt_folder - the folder storing the atom-properties of the target complex
        para - parameters for feature extraction
    '''
    indexdf = pd.read_csv(fn_labels)
    features = []
    labels = []
    # extract features +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for index in range(len(ids)):
        ID = ids[index]
        print(str(index) + ': ' + ID + '......................................')
        try:
            fnp = dt_folder + ID + '/' + ID + '_atm_prop.txt'
            if os.path.isfile(fnp):
                test = Voxelization(fn_atomprop = fnp, ID = ID, labels = indexdf, 
                                    rot_alpha_int = para['rot_ang_steps'][0],
                                    rot_beta_int = para['rot_ang_steps'][1], 
                                    rot_gamma_int = para['rot_ang_steps'][2])
                if para['feat_type'] == 'pufnacy':
                    feat = test.descriptors_pufnacy(box_size = para['box_size'], resolution = para['resolution'], 
                                                    prtcmm = para['prtcmm'], featureset = para['featureset']) 
                elif para['feat_type'] == 'kdeep':
                    feat = test.descriptors_kdeep(box_size = para['box_size'], resolution = para['resolution'], 
                                                  prtcmm = para['prtcmm'], featureset = para['featureset'])
                elif para['feat_type'] == 'sfcnn':
                    feat = test.descriptors_sfcnn(box_size = para['box_size'], resolution = para['resolution'], 
                                                  atmtps_pro = para['atmtps_pro'], atmtps_lig = para['atmtps_lig'])
                else:
                    print('Wrong feature type!!!')
                    return ([], [])
                features += feat[0]
                labels += feat[1]
            else:
                print('Atom-property file does not exist! Please generate that file first!')
                return []
        except:     
            print('Error')
            pass  

    final_feats = np.vstack(features)
    final_labels = np.array(labels)
    
    return (final_feats, final_labels)
######################### input part ############################################
######################### network part ############################################
def corr_true_pred(y_true, y_pred):    
    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred)
    r = r_num/r_den
    r = tf.math.maximum(tf.math.minimum(r, 1.0), -1.0)
    return r

def cus_loss(y_true, y_pred, alpha = 0.8):    
    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred)
    evl_pc = r_num/r_den
    evl_pc = tf.math.maximum(tf.math.minimum(evl_pc, 1.0), -1.0)
    evl_rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))
    loss = alpha * (1 - evl_pc) + (1 - alpha) * evl_rmse
    return loss

def validate_DL_model(mdl, test_dt_X, test_dt_y):
    pred_y = np.squeeze(mdl.predict(test_dt_X))
    # np.squeeze removes the last dimension of 1
    res = {'PC': np.corrcoef(pred_y, test_dt_y)[0, 1], 
           'rmse': mean_squared_error(pred_y, test_dt_y, squared = False)}
    return res

def validate_DL_model_ori(mdl, test_dt_X, test_dt_y, num_of_ori):
    pred_y = mdl.predict(test_dt_X)
    avg_pred_y = np.average(pred_y.reshape(-1, num_of_ori), axis = 1)
    avg_true_y = np.average(test_dt_y.reshape(-1, num_of_ori), axis = 1)
    res = {'PC': np.corrcoef(avg_pred_y, avg_true_y)[0, 1], 
           'rmse': mean_squared_error(avg_pred_y, avg_true_y, squared = False)}
    return res

class HyperMdl_CNN(kt.HyperModel):
    def __init__(self, sz = None, para = None):
        super().__init__()
        self.sz = sz
        self.para = para

    def build(self, hp):
        para = self.para['mdl_para']
        sz = self.sz
        model = Sequential()
        model.add(InputLayer(input_shape = sz, dtype = 'float32'))

        tune_lr = hp.Choice('lr', para['lr']) if type(para['lr']) == list else para['lr']
        for lyr in para['layers']:
            if 'conv' in lyr['name']:
                tune_filter = hp.Choice(lyr['name']+'filters', lyr['filter']) if type(lyr['filter']) == list else lyr['filter']
                tune_kernel_size = hp.Choice(lyr['name']+'kernel_size', lyr['kernel_size']) if type(lyr['kernel_size']) == list else lyr['kernel_size']
                addlayer = Conv3D(filters = tune_filter, kernel_size = tune_kernel_size, 
                                  strides = lyr['strides'], activation = 'relu', 
                                  padding = lyr['padding'],
                                  kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))
                model.add(addlayer)
            elif 'maxpool' in lyr['name']:

                addlayer = MaxPooling3D(pool_size = lyr['pool_size'], strides = lyr['strides'])
                model.add(addlayer)
            elif 'avgpool' in lyr['name']:
                addlayer = AveragePooling3D(pool_size = lyr['pool_size'], strides = lyr['strides'])
                model.add(addlayer)
            elif 'globalavgpool' in lyr['name']:
                addlayer = GlobalAveragePooling3D()
                model.add(addlayer)
            elif 'multidense' in lyr['name']:
                tune_num_layers = hp.Choice('nlyr', lyr['num_layers']) if type(lyr['num_layers']) == list else lyr['num_layers']
                for lyrind in range(tune_num_layers):
                    tune_units = hp.Choice(f'units_{lyrind}', lyr['units']) if type(lyr['units']) == list else lyr['units']
                    addlayer = Dense(units = tune_units, activation = 'relu', 
                                    kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))
                model.add(addlayer)
            elif 'dense' in lyr['name']:
                tune_units = hp.Choice(lyr['name']+'units', lyr['units']) if type(lyr['units']) == list else lyr['units']
                addlayer = Dense(units = tune_units, activation = 'relu', 
                                kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))
                model.add(addlayer)
            elif 'flat' in lyr['name']:
                addlayer = Flatten()
                model.add(addlayer)
            elif 'dropout' in lyr['name']:
                addlayer = Dropout(lyr['ratio'])
                model.add(addlayer)
            else:
                print('Wrong layer type!!!')
                return None
        
        model.add(Dense(units = 1)) ## add the regression unit

        model.compile(loss = 'mean_squared_error',
                      optimizer = tf.keras.optimizers.Adam(learning_rate = tune_lr), # for 3D-CNNs
                      metrics = [RMSE()])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        para = self.para
        tune_epoch = hp.Choice('epochs', para['training_epochs']) if type(para['training_epochs']) == list else para['training_epochs']
        tune_bs = hp.Choice('batch_size', para['batch_size']) if type(para['batch_size']) == list else para['batch_size']
        return model.fit(*args, epochs = tune_epoch, batch_size = tune_bs, verbose = 2,
                         **kwargs)

def fire_module(x, s_1, e_1, e_3, fire_name, regularizer = None, reglambda = 0):
    """ 
    Parameters:
        x - input
        s_1 - number of filters in squeeze part (size of 1)
        e_1 - number of filters in expand part (size of 1)
        e_3 - number of filters in expand part (size of 3)
        fire_name - name of this fire module
        regularizer - whether to use l1/l2 regularizer
        reglambda - lambda for regularizer
    """
    # set kernel sizes according to dimensions
    if regularizer is not None:
        reglz = tf.keras.regularizers.L2(l2 = reglambda) if regularizer == 'l2' else tf.keras.regularizers.L1(l1 = reglambda)
        # squeeze part
        squeeze_x = Conv3D(kernel_size = 1, filters = s_1, padding = 'same', activation = 'relu', 
                                name = fire_name + '_s1',
                                kernel_regularizer = reglz)(x)
        # expand part
        expand_x_1 = Conv3D(kernel_size = 1, filters = e_1, padding = 'same', activation = 'relu', 
                                 name = fire_name + '_e1',
                                 kernel_regularizer = reglz)(squeeze_x)
        expand_x_3 = Conv3D(kernel_size = 3, filters = e_3, padding = 'same', activation = 'relu', 
                                 name = fire_name + '_e3',
                                 kernel_regularizer = reglz)(squeeze_x)
    else:
        # squeeze part
        squeeze_x = Conv3D(kernel_size = 1, filters = s_1, padding = 'same', activation = 'relu', name = fire_name + '_s1')(x)
        # expand part
        expand_x_1 = Conv3D(kernel_size = 1, filters = e_1, padding = 'same', activation = 'relu', name = fire_name + '_e1')(squeeze_x)
        expand_x_3 = Conv3D(kernel_size = 3, filters = e_3, padding = 'same', activation = 'relu', name = fire_name + '_e3')(squeeze_x)

    expand = tf.concat([expand_x_1, expand_x_3], axis = 4)
    return expand

class HyperMdl_SqueezeNet(kt.HyperModel):
    def __init__(self, sz = None, para = None):
        super().__init__()
        self.sz = sz
        self.para = para

    def build(self, hp):
        para = self.para['mdl_para']
        sz = self.sz

        tune_lr = hp.Choice('lr', para['lr']) if type(para['lr']) == list else para['lr']

        input1 = Input(shape = sz, dtype = 'float32')
        x = input1

        for lyr in para['layers']:
            if 'conv' in lyr['name']:
                tune_filter = hp.Choice(lyr['name']+'filters', lyr['filter']) if type(lyr['filter']) == list else lyr['filter']
                tune_kernel_size = hp.Choice(lyr['name']+'kernel_size', lyr['kernel_size']) if type(lyr['kernel_size']) == list else lyr['kernel_size']
                x = Conv3D(filters = tune_filter, kernel_size = tune_kernel_size, 
                        strides = lyr['strides'], activation = 'relu', padding = lyr['padding'],                    
                        # kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.001),
                        # use_bias = True, bias_initializer = tf.keras.initializers.Constant(value = 0.1),
                        kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))(x)
            elif 'maxpool' in lyr['name']:
                x = MaxPooling3D(pool_size = lyr['pool_size'], strides = lyr['strides'])(x)
            elif 'globalavgpool' in lyr['name']:
                x = GlobalAveragePooling3D()(x)
            elif 'avgpool' in lyr['name']:
                x = AveragePooling3D(pool_size = lyr['pool_size'], strides = lyr['strides'])(x)
            elif 'fire' in lyr['name']:
                x = fire_module(x, lyr['s1'], lyr['e1'], lyr['e3'], lyr['name'], regularizer = 'l2', reglambda = 0.01)
            elif 'dropout' in lyr['name']:
                x = Dropout(lyr['ratio'])(x)
            elif 'dense' in lyr['name']:
                tune_units = hp.Choice(lyr['name']+'units', lyr['units']) if type(lyr['units']) == list else lyr['units']
                x = Dense(units = tune_units, activation = 'relu', 
                        # kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 1/sqrt(lyr['units'])),
                        # use_bias = True, bias_initializer = tf.keras.initializers.Constant(value = 1.0),
                        kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))(x)
            elif 'flat' in lyr['name']:
                x = Flatten()(x)
            else:
                print('Wrong layer type!!!')
                return None
                
        y = Dense(units = 1)(x) ## add the regression unit         
        model = tf.keras.models.Model(inputs = input1, outputs = y)

        model.compile(loss = 'mean_squared_error',
                      optimizer = tf.keras.optimizers.Adam(learning_rate = tune_lr), # for 3D-CNNs
                      metrics = [RMSE()])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        para = self.para
        tune_epoch = hp.Choice('epochs', para['training_epochs']) if type(para['training_epochs']) == list else para['training_epochs']
        tune_bs = hp.Choice('batch_size', para['batch_size']) if type(para['batch_size']) == list else para['batch_size']
        return model.fit(*args, epochs = tune_epoch, batch_size = tune_bs, verbose = 2,
                         **kwargs)

def Train_HPMDL(x_train, y_train, validation_data, para):
    sz = x_train.shape[1:]
    # create a keras tuner (random-search mechanism) to search for the best hyperparameters
    if para['model'] == 'CNN':
        tuner = kt.RandomSearch(hypermodel = HyperMdl_CNN(sz = sz, para = para), 
                                objective = kt.Objective("root_mean_squared_error", direction = "min"),
                                max_trials = para['hptune_trials'], executions_per_trial = para['hptune_exe_per_trials'],
                                overwrite = True, directory = para['res_dir'], project_name = 'hptune_cnn')
    else:
        tuner = kt.RandomSearch(hypermodel = HyperMdl_SqueezeNet(sz = sz, para = para), 
                                objective = kt.Objective("root_mean_squared_error", direction = "min"),
                                max_trials = para['hptune_trials'], executions_per_trial = para['hptune_exe_per_trials'],
                                overwrite = True, directory = para['res_dir'], project_name = 'hptune_sqznet')

    tuner.search(x_train, y_train, validation_data = validation_data, 
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 40)])
    # retrieve the best hyperparameters and best model
    bst_hps = tuner.get_best_hyperparameters(num_trials = 1)
    bst_mdl = tuner.get_best_models(num_models = 1)
    retain_bst_mdl = bst_mdl[0]
    # retrain the best model using all data (training data + validation data)
    x_all = np.concatenate((x_train, validation_data[0]))
    y_all = np.concatenate((y_train, validation_data[1]))
    retain_bst_mdl.fit(x = x_all, y = y_all, epochs = 1)
    
    return (retain_bst_mdl, bst_mdl[0], bst_hps[0].values)
######################### network part ############################################


