# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:05 2023

@author: DDW
"""

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
##################################################
from scipy.spatial.distance import cdist
import numpy as np
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
            self.bssz = (Coor_pro.shape[0], self.coor.shape[0])
            # save the non-covalent-interaction adjacency matrix (with covalent-bond pairs assigned with 0)          
            self.pdist = cdist(self.coor, self.coor, metric = 'euclidean') ## keep all the distances after atom filtering (distance can be > self.cut)

    def Get_graph_inputs(self, 
                         feat_set = ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
                                     'hybridization', 'heavyneighbors', 'heteroneighbors', 
                                     'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring', 
                                     'partialCH'],
                         max_bs_num_atoms = 200,
                         adj_type = 'inter',
                         adj_ranges = [(2, 3), (3, 4)]):
        """
        Generate inputs (node features + several adj matrices) for the Graph Learning model.
        Parameters:
            feat_set - node features
            max_bs_num_atoms - maximum atoms at the binding site (for batch computations)
            adj_type - consider inter-molecular ('inter') interactions
                       both inter/intra-molecular interactions for the binding site, protein and ligand ('all')
                       both inter/intra-molecular interactions for the binding site and protein/ligand ('oth') 
            adj_ranges - distance ranges for generating adj matrices
        """

        bs_size = self.bssz[1]
        throw = 1 if bs_size > max_bs_num_atoms else 0
        inputs = [[], [], []]
        
        if throw == 0:
            # --------------------- generate node-feature matrix ---------------------------
            # 1. for binding site ----------------------------------------------------------
            features_bs = np.zeros(shape = (max_bs_num_atoms, len(feat_set)))
            for nodei in range(bs_size):
                features_bs[nodei] = np.array(self.AP.iloc[nodei][feat_set])
            inputs[0].append(np.expand_dims(features_bs, axis = 0))
            # 2. for protein fragment ------------------------------------------------------
            features_pro = np.zeros(shape = (max_bs_num_atoms, len(feat_set)))
            for nodei in range(self.bssz[0]):
                features_pro[nodei] = np.array(self.AP.iloc[nodei][feat_set])
            if adj_type == 'all':
                inputs[1].append(np.expand_dims(features_pro, axis = 0))
            # 3. for ligand atoms ----------------------------------------------------------
            features_lig = np.zeros(shape = (max_bs_num_atoms, len(feat_set)))
            for nodei in range(self.bssz[0], bs_size):
                features_lig[nodei - self.bssz[0]] = np.array(self.AP.iloc[nodei][feat_set])
            if adj_type == 'all':
                inputs[2].append(np.expand_dims(features_lig, axis = 0))
            # --------------------- generate adjacency matrix by ranges ------------------------------
            adjrgs = set(adj_ranges)
            if len(adjrgs) > 0:
                for rg in adjrgs:
                    if abs(rg[0]) <= self.cut and abs(rg[1]) <= self.cut:
                        pro_intra_rg = range(0, self.bssz[0])
                        lig_intra_rg = range(self.bssz[0], self.bssz[1])
                        filtids = np.nonzero((self.pdist > rg[0]) & (self.pdist <= rg[1]))
                        # 1. adjacency matrices -----------------------------------------------
                        adjacency = np.zeros(shape = (max_bs_num_atoms, max_bs_num_atoms))
                        # adjacency[filtids[0], filtids[1]] = 1
                        alists = [adjacency] * 4
                        for cont in range(filtids[0].shape[0]):
                            atm1 = filtids[0][cont]
                            atm2 = filtids[1][cont]
                            if (atm1 in pro_intra_rg) and (atm2 in pro_intra_rg):
                                alists[1][(atm1, atm2)] = 1
                                alists[3][(atm1, atm2)] = 1
                            elif (atm1 in lig_intra_rg) and (atm2 in lig_intra_rg):
                                alists[2][(atm1 - self.bssz[0], atm2 - self.bssz[0])] = 1
                                alists[3][(atm1, atm2)] = 1
                            else:
                                alists[0][(atm1, atm2)] = 1  
                            # note that all the connections are bi-directional (symmetric)  
                        ## normalize the adjacency matrices -----------------------------------
                        for adji in range(len(alists)):
                            A_tilde = alists[adji] + np.eye(max_bs_num_atoms)
                            D_tilde = np.diag(1/np.sqrt(np.sum(A_tilde, axis = 0)))                      
                            A_sharp = np.matmul(np.matmul(D_tilde, A_tilde), D_tilde) 
                            alists[adji] = A_sharp
                  
                        # include the alist for bs
                        inputs[0].append(np.expand_dims(alists[0], axis = 0))    
                        if adj_type == 'all':
                            inputs[1].append(np.expand_dims(alists[1], axis = 0)) 
                            inputs[2].append(np.expand_dims(alists[2], axis = 0))      
                        else:
                            inputs[1].append(np.expand_dims(alists[3], axis = 0)) 

                        # ------------------------------------------------------------------
                        # # 2. adjacency lists -----------------------------------------------
                        # alists = [[], [], []]
                        # for cont in range(filtids[0].shape[0]):
                        #     atm1 = filtids[0][cont]
                        #     atm2 = filtids[1][cont]
                        #     if (atm1 in pro_intra_rg) and (atm2 in pro_intra_rg):
                        #         alists[1].append((atm1, atm2))
                        #     elif (atm1 in lig_intra_rg) and (atm2 in lig_intra_rg):
                        #         alists[2].append((atm1 - self.bssz[0], atm2 - self.bssz[0]))
                        #     else:
                        #         alists[0].append((atm1, atm2))    
                        #     # note that all the connections are bi-directional (symmetric)                    
                        # # include the alist for bs
                        # inputs[0].append(np.expand_dims(alists[0], axis = 0))    
                        # if adj_type != 'inter':
                        #     inputs[1].append(np.expand_dims(alists[1], axis = 0)) 
                        #     inputs[2].append(np.expand_dims(alists[2], axis = 0))    
                        # # ------------------------------------------------------------------

        return inputs

def feature_engineering_AGIMA_Score(fn_labels, dt_folder, ids = None, 
                                    para = {'feat': ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
                                                    'hybridization', 'heavyneighbors', 'heteroneighbors', 
                                                    'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring', 
                                                    'partialCH'],
                                            'max_bs_num_atoms': 200,                                          
                                            'int_cutoff': 4, 
                                            'adj_type': 'inter',
                                            'adj_ranges': [(2, 3), (3, 4)]}):
    '''
    Extrat features for AGIMA-Score.
    Parameters:
        ids - a list of samples (PDB IDs) for feature extraction
        fn_labels - path to the index file (in 'indexes' folder)
        dt_folder - the folder storing the atom-properties of the target complex
        para - parameters for feature extraction
    '''
    num_adj = len(para['adj_ranges'])
    num_frag = 1 if para['adj_type'] == 'inter' else 3
    num_inp = (num_adj + 1) * num_frag
    
    indexdf = pd.read_csv(fn_labels)
    features = [[] for inputi in range(num_inp)]
    labels = []
    curids = indexdf['id'].tolist() if ids is None else ids
    # extract features +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for index in range(len(curids)):
        ID = curids[index]
        print(str(index) + ': ' + ID + '......................................')

        try:
            fnp = dt_folder + ID + '/' + ID + '_atm_prop.txt'
            tmp = Graph_inputs(fn_atomprop = fnp, labels = indexdf, 
                               int_cutoff = para['int_cutoff'], ID = ID)
            feat = tmp.Get_graph_inputs(feat_set = para['feat'],
                                        max_bs_num_atoms = para['max_bs_num_atoms'],
                                        adj_type = para['adj_type'],
                                        adj_ranges = para['adj_ranges'])
            
            if len(feat[0]) > 0:
                for inputi in range(num_frag):
                    strtid = inputi * (num_adj + 1)
                    for inputj in range(num_adj + 1):                            
                        features[strtid + inputj].append(feat[inputi][inputj])
                labels.append(tmp.label)
        except:     
            print('Error')
            pass  

    final_feats = [np.concatenate(features[i], axis = 0) for i in range(len(features))]
    final_labels = np.array(labels)
    
    return (final_feats, final_labels)

######################### GCN input part ############################################
######################### network part ############################################
# ----------------------------------------------------------------------------------
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

def AGIMA_Score(sz, adj_num, gcn_lyrs, l2):
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

    mnt = tf.concat(ys, axis = -1)
    z = Dense(units = 128)(mnt)
    z = Dropout(0.5)(z)
    outp = Dense(units = 1)(z)

    mdl = Model(inputs = [input1] + GCBs, outputs = [outp, mnt])

    mdl.compile(loss = 'mse',
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                metrics = [RMSE()])
    return mdl

class HyperMdl_AGIMA_Score(kt.HyperModel):
    def __init__(self, para = None):
        super().__init__()
        self.para = para

    def build(self, hp):
        para = self.para

        tune_gcn_depth = hp.Choice('gcn_depth', para['gcn_lyrs']) if type(para['gcn_lyrs']) == list else para['gcn_lyrs']
        tune_l2 = hp.Choice('l2_para', para['l2']) if type(para['l2']) == list else para['l2']

        model = AGIMA_Score(sz = para['sz'], 
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

def Train_AGIMA_Score(x_train, y_train, validation_data, para):
    # create a keras tuner (random-search mechanism) to search for the best hyperparameters
    tuner = kt.RandomSearch(hypermodel = HyperMdl_AGIMA_Score(para = para), 
                            objective = kt.Objective("root_mean_squared_error", direction = "min"),
                            max_trials = para['hptune_trials'], executions_per_trial = para['hptune_exe_per_trials'],
                            overwrite = True, directory = para['res_dir'], project_name = 'hptune_gcn')

    mod_y_train = [y_train, np.zeros((y_train.shape[0], 64))] # two adj, one with a 32bit embedding length
    mod_validation_data = (validation_data[0], 
                           [validation_data[1], np.zeros((validation_data[1].shape[0], 64))])
    tuner.search(x_train, mod_y_train, 
                 validation_data = mod_validation_data)
    # retrieve the best hyperparameters and best model
    bst_hps = tuner.get_best_hyperparameters(num_trials = 1)
    bst_mdl = tuner.get_best_models(num_models = 1)
    retain_bst_mdl = bst_mdl[0]
    # retrain the best model using all data (training data + validation data)
    x_all = [np.concatenate((x_train[chi], validation_data[0][chi])) for chi in range(len(x_train))]
    y_all = np.concatenate((y_train, validation_data[1]))
    retain_bst_mdl.fit(x = x_all, y = y_all, epochs = 1)
    
    return (retain_bst_mdl, bst_mdl[0], bst_hps[0].values)

def validate_AGIMA_Score(mdl, test_dt_X, test_dt_y):
    pred_y = np.squeeze(mdl.predict(test_dt_X)[0])
    # np.squeeze removes the last dimension of 1
    res = {'PC': np.corrcoef(pred_y, test_dt_y)[0, 1], 
           'rmse': mean_squared_error(pred_y, test_dt_y, squared = False)}
    return res
######################### network part(wm) ############################################


# ############################# EXAMPLE 1 (featurization) ####################################
# ids =  ['1a9m', '1a30', '1apv']
# fn_labels = './indexes/rs_index.csv'
# dt_folder = './examples/'

# fe = feature_engineering_AGIMA_Score(ids = ids, fn_labels = fn_labels, dt_folder = dt_folder)
# # fe[0] is the feature representation of three parts:
#         # 1. node-feature matrix for each complex (num_of_complexes*200*num_of_features)
#         # 2. adjacency matrix 1 (2A~3A) for each complex (num_of_complexes*200*200)
#         # 3. adjacency matrix 2 (3A~4A) for each complex (num_of_complexes*200*200)
# # fe[1] is the list of labels (affinity) for each complex
# ###########################################################################################


