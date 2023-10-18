# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:17:20 2022

@author: debby
"""
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import itertools
import os
from collections import defaultdict
import hickle as hkl
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
import keras_tuner as kt
from scipy.stats import pearsonr

######################### input part ############################################
class IMC(object):
    def __init__(self, fn_atomprop, ID, labels):
        """
        Initialize an IMC class.

        Parameters:
            fn_atomprop - path to the file storing the atomic properties of the target complex
            ID - ID for complex
            labels - dataframe storing the IDs and affinities (labels) for the complexes
        """
        self.ID = ID if ID is not None else "PL"
        print('Constructing an IMC object for %s.........\n' % self.ID)

        lb = labels.loc[labels['id'] == ID,'affinity']
        self.label = lb[lb.index[0]] # lb is a data frame, lb.index[0] finds the index of its first row
            
        if os.path.isfile(fn_atomprop):
            # read in data
            self.AP = pd.read_csv(fn_atomprop)

    def gen_imc_desp(self, bins = [], proatm_tpdic = {}, ligatm_tpdic = {}, ext = 1, channel = 1):
        '''
        Extract IMC-like descriptors (e.g. OninonNet descriptors).
        Parameters:
            bins - distance bins for counting interacting atom pairs
            proatm_tpdic - atom types for protein atoms
            ligatm_tpdic - atom types for ligand atoms
            ext - whether to consider default atom types
            channel - property channels for IMCs (1 - only counts, 2 - counts and avg dist)
        '''
        # 1. contructing a list of distance bins ------------------------------------------
        if len(bins) == 0:
            cur_bins = [(0, 1)]
            for shell in np.arange(2, 61):
                cur_bins.append((1 + (shell - 2) * 0.5, 1 + (shell - 1) * 0.5))  # By default, use the bins in OnionNet
        else:
            cur_bins = bins
        # 2.1. selecting atoms types for protein ---------------------------------------------
        if len(proatm_tpdic) == 0:
            # OnionNet
            cur_proatm_tpdic = defaultdict(lambda: 100, 
                                           {6: 0, 7: 1, 8: 2, 1: 3, 15: 4, 16: 5, 9: 6, 17: 6, 35: 6, 53: 6})
            # # RF-Score
            # cur_proatm_tpdic = {6: 0, 7: 1, 8: 2, 16: 3} 
        else:
            cur_proatm_tpdic = proatm_tpdic
         # 2.2. selecting atoms types for ligand ---------------------------------------------
        if len(ligatm_tpdic) == 0:
            # OnionNet
            cur_ligatm_tpdic = defaultdict(lambda: 100, 
                                           {6: 0, 7: 1, 8: 2, 1: 3, 15: 4, 16: 5, 9: 6, 17: 6, 35: 6, 53: 6})
            # # RF-Score
            # cur_ligatm_tpdic = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8} 
        else:
            cur_ligatm_tpdic = ligatm_tpdic
        # 2.3. using ext to indicate whether the rest of atom types are considered --------------------        
        if ext == 1:
            proatmtps = list(set(cur_proatm_tpdic.values())) + [100]
            ligatmtps = list(set(cur_ligatm_tpdic.values())) + [100]
        else:
            proatmtps = list(set(cur_proatm_tpdic.values()))
            ligatmtps = list(set(cur_ligatm_tpdic.values()))           
        # atom-pair types -------------------------------------------------------------
        aptplst = list(itertools.product(proatmtps, ligatmtps))
        
        # 3. generate descriptors --------------------------------------------------------
        dt = np.zeros(shape = (len(cur_bins), len(proatmtps) * len(ligatmtps), channel))
        APpro = self.AP.loc[self.AP['moltype'] == 0]
        APlig = self.AP.loc[self.AP['moltype'] == 1]
        points_pro = np.array([[x, y, z] for (x, y, z) in zip(APpro['x'].tolist(), APpro['y'].tolist(), APpro['z'].tolist())])
        points_lig = np.array([[x, y, z] for (x, y, z) in zip(APlig['x'].tolist(), APlig['y'].tolist(), APlig['z'].tolist())])
        pdist = cdist(points_pro, points_lig, metric = 'euclidean')
        for bn in cur_bins:
            # print(bn)
            filtids = np.nonzero((pdist < bn[1]) & (pdist >= bn[0]))
            ## returns (array1([...]), array2([...])), array1 gives the first index i of an element fulfiling the conditions and array2 gives the second index j; e.g. pdist[i][j]           
            apnum = filtids[0].shape[0]
            if apnum > 0:
                for apid in range(apnum):
                    proaid = filtids[0][apid]
                    ligaid = filtids[1][apid]
                    proanum = APpro.iloc[proaid][1] # returns the atomic number of protein atom
                    liganum = APlig.iloc[ligaid][1] # returns the atomic number of ligand atom
                    if proanum in cur_proatm_tpdic and liganum in cur_ligatm_tpdic:
                        aptps = (cur_proatm_tpdic[proanum], cur_ligatm_tpdic[liganum])
                        if aptps in aptplst:
                            dt[(cur_bins.index(bn), aptplst.index(aptps))][0] += 1
                            if channel == 2:
                                dt[(cur_bins.index(bn), aptplst.index(aptps))][1] += pdist[proaid][ligaid] # accumulating distances (ICMP)                  
                # get average distance for each bin-aptp cell
                if channel == 2:
                    for curaptp in aptplst:
                        if dt[(cur_bins.index(bn), aptplst.index(curaptp))][1] > 0:
                            dt[(cur_bins.index(bn), aptplst.index(curaptp))][1] /= dt[(cur_bins.index(bn), aptplst.index(curaptp))][0]
        feat = np.expand_dims(dt, axis = 0) if len(cur_bins) > 1 else dt
        # if num of bins == 1, feat of size 1*NumOfAtomPairTypes*channel; else feat of size 1*NumOfBins*NumOfAtomPairTypes*channel
         
        return (feat, self.label)

def feature_engineering_IMC_CNN(ids, fn_labels, dt_folder,
                                para = {'bins': [], 'proatm_tpdic': {}, 'ligatm_tpdic': {}, 'ext': 1, 'channel': 1}):
    '''
    Extrat features for BAP.
    Parameters:
        ids - a list of samples (PDB IDs) for feature extraction
        fn_labels - path to the index file (in 'indexes' folder)
        dt_folder - the folder storing the atom-properties of the target complex
        para - parameters for feature extraction
    '''
    print('Generating feature representations for IMC-CNN models........................................\n\n')
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
                test = IMC(fn_atomprop = fnp, ID = ID, labels = indexdf)
                feat = test.gen_imc_desp(bins = para['bins'], proatm_tpdic = para['proatm_tpdic'], 
                                            ligatm_tpdic = para['ligatm_tpdic'], ext = para['ext'], channel = para['channel'])

                features.append(feat[0])
                labels.append(feat[1])
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

        for lyr in para['layers']:
            if 'conv' in lyr['name']:
                tune_filter = hp.Choice(lyr['name']+'filters', lyr['filter']) if type(lyr['filter']) == list else lyr['filter']
                tune_kernel_size = hp.Choice(lyr['name']+'kernel_size', lyr['kernel_size']) if type(lyr['kernel_size']) == list else lyr['kernel_size']
                addlayer = Conv2D(filters = tune_filter, kernel_size = tune_kernel_size, 
                                  strides = lyr['strides'], activation = 'relu', 
                                  padding = lyr['padding'],
                                  kernel_regularizer = tf.keras.regularizers.L2(l2 = 0.01))
                model.add(addlayer)
            elif 'maxpool' in lyr['name']:
                addlayer = MaxPooling2D(pool_size = lyr['pool_size'], strides = lyr['strides'])
                model.add(addlayer)
            elif 'avgpool' in lyr['name']:
                addlayer = AveragePooling2D(pool_size = lyr['pool_size'], strides = lyr['strides'])
                model.add(addlayer)
            elif 'globalavgpool' in lyr['name']:
                addlayer = GlobalAveragePooling2D()
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

        # curopt = tf.keras.optimizers.Adam(learning_rate = 0.001)
        model.compile(loss = 'mean_squared_error',
                      optimizer = 'adam',
                      metrics = [RMSE()])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        para = self.para
        tune_epoch = hp.Choice('epochs', para['training_epochs']) if type(para['training_epochs']) == list else para['training_epochs']
        tune_bs = hp.Choice('batch_size', para['batch_size']) if type(para['batch_size']) == list else para['batch_size']
        return model.fit(*args, epochs = tune_epoch, batch_size = tune_bs, verbose = 2,
                         **kwargs)

def Train_MDL(x_train, y_train, validation_data, para):
    sz = x_train.shape[1:]
    # create a keras tuner (random-search mechanism) to search for the best hyperparameters
    tuner = kt.RandomSearch(hypermodel = HyperMdl_CNN(sz = sz, para = para), 
                            objective = kt.Objective("root_mean_squared_error", direction = "min"),
                            max_trials = para['hptune_trials'], executions_per_trial = para['hptune_exe_per_trials'],
                            overwrite = True, directory = para['res_dir'], project_name = 'hptune_cnn')
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



