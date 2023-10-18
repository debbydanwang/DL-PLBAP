# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:05 2023

@author: DDW
"""

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import hickle as hkl
import os
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
import keras_tuner as kt
from sklearn.metrics import mean_squared_error

######################### GCN input part ############################################
class Graph_inputs(object):    
    def __init__(self, 
                 fn_atomprop,  
                 labels,
                 int_cutoff = 4,
                 ID = None):
        """
        Initialization...
        Parameters:
            fn_atomprop - path to the file storing the atomic properties of the target complex
            ID - ID of the complex
            labels - atomic info table
            int_cutoff - distance cutoff for defining a binding area
        """
        self.ID = ID if ID is not None else "PL"
        self.cut = int_cutoff
        print('Constructing a Graph object for %s.........\n' % self.ID)

        lb = labels.loc[labels['id'] == ID,'affinity']
        self.label = lb[lb.index[0]] # lb is a data frame, lb.index[0] finds the index of its first row
        
        if os.path.isfile(fn_atomprop):
            rawAP = pd.read_csv(fn_atomprop)
            AP = rawAP[rawAP['atmnum'] > 1] # IGNORE hydrogen atoms
            proatmnum = AP['moltype'].tolist().count(0)
            AP_pro_all = AP[:proatmnum]
            AP_lig = AP[proatmnum:]
            coor = np.array(list(zip(AP['x'].tolist(), AP['y'].tolist(), AP['z'].tolist())))
            Coor_pro_all = coor[:proatmnum]
            Coor_lig = coor[proatmnum:]
            pdist = cdist(Coor_pro_all, Coor_lig, metric = 'euclidean')
            # filter atom pairs within a distance threshold --------------------------
            filtids_pro = sorted(np.unique(np.nonzero(pdist <= int_cutoff)[0]))
            AP_pro = AP_pro_all.iloc[filtids_pro]
            Coor_pro = Coor_pro_all[filtids_pro]
            # store properties and coordinates of filtered atoms ---------------------
            self.AP = pd.concat([AP_pro, AP_lig])
            self.coor = np.concatenate([Coor_pro, Coor_lig], axis = 0)
            self.bssz = self.coor.shape[0]
            # save the non-covalent-interaction adjacency matrix (with covalent-bond pairs assigned with 0)          
            self.pdist = cdist(self.coor, self.coor, metric = 'euclidean') ## keep all the distances after atom filtering (distance can be > self.cut)
            ## filtids = np.nonzero(self.pdist > self.cut)
            ## self.pdist[filtids[0], filtids[1]] = 0
            self.non_cov_adj = self.pdist
            # save the covalent-bond adjacency matrix (with bond types - 1, 1.5 and 2)
            nodesz = self.AP.shape[0]
            id_trans_dic = {(self.AP.moltype.tolist()[i], self.AP.id.tolist()[i]):i for i in range(nodesz)}
            self.cov_adj = np.zeros(shape = (nodesz, nodesz))
            for ni in range(nodesz):
                curatm = self.AP.iloc[ni]
                moltp = curatm.moltype
                nbrstr = curatm['neighbors(nbr:idx--anum--(sbond,dbond,tbond,arombond,ringbond))']
                if not pd.isna(nbrstr):
                    nbrs = nbrstr.split('//')[:-1]
                    for nbr in nbrs:
                        nbrdetails = nbr.split('--')
                        dicid = (moltp, int(nbrdetails[0][4:]))
                        if dicid in id_trans_dic:
                            nbrid = id_trans_dic[dicid]
                            edg_props = [int(prop) for prop in nbrdetails[2][1:-1].split(',')]
                            if edg_props[0] == 1:
                                self.cov_adj[ni, nbrid] = 1.0
                            elif edg_props[1] == 1:
                                self.cov_adj[ni, nbrid] = 1.5
                            else:
                                self.cov_adj[ni, nbrid] = 2
                            # turn of the value of this atom pair (with a covalent bond) in the non-covalent-interaction adjacency matrix
                            self.non_cov_adj[ni, nbrid] = 0  

    def Get_inputs_GraphBAR(self, 
                            feat_set = ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
                                        'hybridization', 'heavyneighbors', 'heteroneighbors', 'partialCH'],
                            max_complex_num_atoms = 200,
                            adj_type = 'norm',
                            covalent_adj = False,
                            covalent_num_adj = 1,
                            non_covalent_adj = True,
                            non_covalent_num_adj = 2,
                            non_covalent_adj_rg = False,
                            non_covalent_adj_ranges = [(0, 4), (4, 6), (6, 10)]):
        """
        Generate inputs (node features + several adj matrices) for the GraphBAR model.
        Each adj matrix is preprocessed by D^(-1/2) (adj+I) D^(-1/2).
        Parameters:
            feat_set - node features
            max_complex_num_atoms - maximum atoms at the binding site (for batch computations)
            adj_type - whether to output the original adjacency mat or the normalized mat (A_sharp)
            covalent_adj: sign of generating covalent adjacency matrices
            covalent_num_adj - number of covalent adjacency matrices (1 - weighted adj, -1 - binary adj, others - multiple binary adjs)
            non_covalent_adj: sign of generating non-covalent adjacency matrices (distance-based)
            non_covalent_num_adj - number of non-covalent adjacency matrices 
                               (1 - distance adj, -1 - distance adj with covalent bonds eliminated, 
                               others - > 0: multiple binary adjs,
                               others - < 0: multiple binary adjs with covalent bonds eliminated)
            non_covalent_adj_rg: sign of generating non-covalent adjacency matrices (range-based)
            non_covalent_adj_ranges - distance ranges for generating adj matrices
            """

        bs_size = self.AP.shape[0]
        throw = 1 if bs_size > max_complex_num_atoms else 0
        inputs = []
        
        if throw == 0:
            # --------------------- generate node-feature matrix ---------------------------
            features = np.zeros(shape = (max_complex_num_atoms, len(feat_set)))
            for nodei in range(bs_size):
                features[nodei] = np.array(self.AP.iloc[nodei][feat_set])
            inputs.append(np.expand_dims(features, axis = 0)) # node feature matrix
            # --------------------- generate covalent-bond adjacency matrix ------------------------------
            if covalent_adj:
                if abs(covalent_num_adj) == 1:
                    adjacency = np.zeros(shape = (max_complex_num_atoms, max_complex_num_atoms))
                    adjacency[0:bs_size, 0:bs_size] = self.cov_adj
                    if covalent_num_adj == -1:
                        filtids = np.nonzero(self.cov_adj > 0)
                        adjacency[filtids[0], filtids[1]] = 1
                    A_tilde = adjacency + np.eye(max_complex_num_atoms)
                    D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                    A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                    if adj_type == 'norm':
                        inputs.append(np.expand_dims(A_sharp, axis = 0))
                    else:
                        inputs.append(np.expand_dims(adjacency, axis = 0))
                else:
                    covbonds = [1, 1.5, 2]
                    for adj in range(len(covbonds)):
                        adjacency = np.zeros(shape = (max_complex_num_atoms, max_complex_num_atoms))
                        filtids = np.nonzero(self.cov_adj == covbonds[adj])
                        adjacency[filtids[0], filtids[1]] = 1
                        A_tilde = adjacency + np.eye(max_complex_num_atoms)
                        D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                        A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                        if adj_type == 'norm':
                            inputs.append(np.expand_dims(A_sharp, axis = 0))
                        else:
                            inputs.append(np.expand_dims(adjacency, axis = 0))
            # --------------------- generate non-covalent-bond adjacency matrix ------------------------------
            if non_covalent_adj:
                if abs(non_covalent_num_adj) == 1:
                    adjacency = np.zeros(shape = (max_complex_num_atoms, max_complex_num_atoms))
                    if non_covalent_num_adj == 1:
                        adjacency[0:bs_size, 0:bs_size] = self.pdist
                    else:
                        adjacency[0:bs_size, 0:bs_size] = self.non_cov_adj
                    A_tilde = adjacency + np.eye(max_complex_num_atoms)
                    D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                    A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                    if adj_type == 'norm':
                        inputs.append(np.expand_dims(A_sharp, axis = 0))
                    else:
                        inputs.append(np.expand_dims(adjacency, axis = 0))
                elif abs(non_covalent_num_adj) > 1:
                    cur_adj_num = abs(int(non_covalent_num_adj))
                    interval = self.cut / cur_adj_num
                    cur_distm = self.pdist if non_covalent_num_adj > 0 else self.non_cov_adj
                    for adj in range(cur_adj_num):
                        cur_range = (interval * adj, interval * (adj + 1))
                        adjacency = np.zeros(shape = (max_complex_num_atoms, max_complex_num_atoms))
                        filtids = np.nonzero((cur_distm > cur_range[0]) & (cur_distm <= cur_range[1]))
                        adjacency[filtids[0], filtids[1]] = 1
                        A_tilde = adjacency + np.eye(max_complex_num_atoms)
                        D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                        A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                        if adj_type == 'norm':
                            inputs.append(np.expand_dims(A_sharp, axis = 0))
                        else:
                            inputs.append(np.expand_dims(adjacency, axis = 0))
            # --------------------- generate non-covalent-bond adjacency matrix by ranges ------------------------------
            if non_covalent_adj_rg:
                for rg in non_covalent_adj_ranges:
                    if abs(rg[0]) <= self.cut and abs(rg[1]) <= self.cut:
                        adjacency = np.zeros(shape = (max_complex_num_atoms, max_complex_num_atoms))
                        cur_distm = self.pdist if sum(rg) > 0 else self.non_cov_adj
                        cur_range = rg if sum(rg) > 0 else (-rg[1], -rg[0])
                        filtids = np.nonzero((cur_distm > cur_range[0]) & (cur_distm <= cur_range[1]))
                        adjacency[filtids[0], filtids[1]] = 1
                        A_tilde = adjacency + np.eye(max_complex_num_atoms)
                        D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                        A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                        if adj_type == 'norm':
                            inputs.append(np.expand_dims(A_sharp, axis = 0))
                        else:
                            inputs.append(np.expand_dims(adjacency, axis = 0))    

        return inputs

def feature_engineering_GCN(ids, fn_labels, dt_folder,
                            para = {'feat': ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
                                             'hybridization', 'heavyneighbors', 'heteroneighbors', 'partialCH'],
                                    'max_complex_num_atoms': 200,                                          
                                    'int_cutoff': 4, 
                                    'adj_type': 'norm',
                                    'cov_adj': True, 'cov_adj_num': -1,
                                    'ncov_adj': True, 'ncov_adj_num': -1,
                                    'ncov_adj_rg': False, 'ncov_adj_num_ranges': []}):
    '''
    Extrat features for BAP.
    Parameters:
        ids - a list of samples (PDB IDs) for feature extraction
        fn_labels - path to the index file (in 'indexes' folder)
        dt_folder - the folder storing the atom-properties of the target complex
        para - parameters for feature extraction
    '''
    num_adj = 0
    if para['cov_adj']:
        num_adj += abs(para['cov_adj_num'])    
    if para['ncov_adj']:
        num_adj += int(abs(para['ncov_adj_num']))
    if para['ncov_adj_rg']:
        num_adj += len(para['ncov_adj_num_ranges'])   
    
    indexdf = pd.read_csv(fn_labels)
    features = [[] for inputi in range(num_adj + 1)] # +1 because of the feature matrix
    labels = []
    # extract features +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for index in range(len(ids)):
        ID = ids[index]
        print(str(index) + ': ' + ID + '......................................')

        try:
            fnp = dt_folder + ID + '/' + ID + '_atm_prop.txt'
            if os.path.isfile(fnp):
                tmp = Graph_inputs(fn_atomprop = fnp, labels = indexdf, int_cutoff = para['int_cutoff'], ID = ID)
                feat = tmp.Get_inputs_GraphBAR(feat_set = para['feat'],
                                                max_complex_num_atoms = para['max_complex_num_atoms'],
                                                adj_type = para['adj_type'],
                                                covalent_adj = para['cov_adj'],
                                                covalent_num_adj = para['cov_adj_num'],
                                                non_covalent_adj = para['ncov_adj'],
                                                non_covalent_num_adj = para['ncov_adj_num'],
                                                non_covalent_adj_rg = para['ncov_adj_rg'],
                                                non_covalent_adj_ranges = para['ncov_adj_num_ranges'])
                
                if len(feat) > 0:
                    # length of inputs > 0 indicates that the bs size is <= 200
                    for inputi in range(num_adj + 1):
                        features[inputi].append(feat[inputi])
                    labels.append(tmp.label)
            else:
                print('Atom-property file does not exist! Please generate that file first!')
                return []
        except:     
            print('Error')
            pass  

    final_feats = [np.concatenate(features[i], axis = 0) for i in range(len(features))]
    final_labels = np.array(labels)
    
    return (final_feats, final_labels)
######################### GCN input part ############################################
######################### network part ############################################
class GCNlayer(layers.Layer):
    def __init__(self, output_dim = 128, l2 = 0.01, **kwargs):
        super(GCNlayer, self).__init__()
        self.output_dim = output_dim
        self.l2 = l2
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'l2': self.l2,
        })
        return config
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name = 'W',
            shape = (input_shape[0][-1], self.output_dim),
            initializer = 'random_normal',
            trainable = True,
            regularizer = tf.keras.regularizers.L2(l2 = self.l2)
        )
    
    def call(self, inputs):
        X, A_sharp = inputs[0], inputs[1]
        X_bar = tf.nn.relu(tf.matmul(tf.matmul(A_sharp, X), self.W))
        return X_bar

def GraphBAR(sz, adj_num, gcn_lyrs, l2):
    input1 = Input(shape = sz[0], dtype = 'float32')
    X = Dense(units = 128)(input1)
    X = Dropout(0.5)(X)
    GCBs = [Input(shape = sz[i + 1], dtype = 'float32') for i in range(adj_num)]
    ys = []

    for blk_i in range(adj_num):   
        cur_input = X
        for lyr_i in range(gcn_lyrs - 1):
            gcl = GCNlayer(output_dim = 128, l2 = l2)([cur_input, GCBs[blk_i]])
            dense = Dense(units = 128)(gcl)
            cur_input = Dropout(0.5)(dense)

        gcl_last = GCNlayer(output_dim = 128, l2 = l2)([cur_input, GCBs[blk_i]])
        dense_last = Dense(units = 16 * adj_num)(gcl_last)
        drop_last = Dropout(0.5)(dense_last)
        gather = tf.reduce_sum(drop_last, axis = -2)

        ys.append(gather)

    z = tf.concat(ys, axis = -1)
    z = Dense(units = 128)(z)
    z = Dropout(0.5)(z)
    outp = Dense(units = 1)(z)

    mdl = Model(inputs = [input1] + GCBs, outputs = outp)

    mdl.compile(loss = 'mean_squared_error',
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                metrics = [RMSE()])
    return mdl

class HyperMdl_BAP(kt.HyperModel):
    def __init__(self, para = None):
        super().__init__()
        self.para = para

    def build(self, hp):
        para = self.para

        tune_gcn_depth = hp.Choice('gcn_depth', para['gcn_lyrs']) if type(para['gcn_lyrs']) == list else para['gcn_lyrs']
        tune_l2 = hp.Choice('l2_para', para['l2']) if type(para['l2']) == list else para['l2']

        model = GraphBAR(sz = para['sz'], 
                         adj_num = para['adj_num'],
                         gcn_lyrs = tune_gcn_depth, 
                         l2 = tune_l2)

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        para = self.para
        tune_epoch = hp.Choice('epochs', para['training_epochs']) if type(para['training_epochs']) == list else para['training_epochs']
        tune_bs = hp.Choice('batch_size', para['batch_size']) if type(para['batch_size']) == list else para['batch_size']
        return model.fit(*args, epochs = tune_epoch, batch_size = tune_bs, verbose = 2,
                         **kwargs)

def Train_GraphBAR(x_train, y_train, validation_data, para):
    # create a keras tuner (random-search mechanism) to search for the best hyperparameters
    tuner = kt.RandomSearch(hypermodel = HyperMdl_BAP(para = para), 
                            objective = kt.Objective("root_mean_squared_error", direction = "min"),
                            max_trials = para['hptune_trials'], executions_per_trial = para['hptune_exe_per_trials'],
                            overwrite = True, directory = para['res_dir'], project_name = 'hptune_gcn')

    tuner.search(x_train, y_train, validation_data = validation_data, 
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 40)])
    # retrieve the best hyperparameters and best model
    bst_hps = tuner.get_best_hyperparameters(num_trials = 1)
    bst_mdl = tuner.get_best_models(num_models = 1)
    retain_bst_mdl = bst_mdl[0]
    # retrain the best model using all data (training data + validation data)
    x_all = [np.concatenate((x_train[chi], validation_data[0][chi])) for chi in range(len(x_train))]
    y_all = np.concatenate((y_train, validation_data[1]))
    retain_bst_mdl.fit(x = x_all, y = y_all, epochs = 1)
    
    return (retain_bst_mdl, bst_mdl[0], bst_hps[0].values)

def validate_DL_model(mdl, test_dt_X, test_dt_y):
    pred_y = np.squeeze(mdl.predict(test_dt_X))
    # np.squeeze removes the last dimension of 1
    res = {'PC': np.corrcoef(pred_y, test_dt_y)[0, 1], 
           'rmse': mean_squared_error(pred_y, test_dt_y, squared = False)}
    return res
######################### network part ############################################


