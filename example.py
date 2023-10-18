# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:29:05 2023

@author: DDW
"""
######################################### EXAMPLE 1 ###################################################
from atmprop_extra_ob import ComProp

ID = '1a9m'
dt_folder = './examples/'
fn_pro = dt_folder + ID + '/' + ID + '_protein.pdb' #path to protein pdb file
fn_lig = dt_folder + ID + '/' + ID + '_ligand.pdb' #path to ligand pdb file

test = ComProp(fn_pro, fn_lig, 'pdb', 'pdb')
propdf = test.extract_atom_prop()
fn = dt_folder + ID + '/' + ID + '_atm_prop.txt'
propdf.to_csv(fn, index = False)
# -----------------------------------------------------------------
########################################################################################################


######################################### EXAMPLE 2 ###################################################
# from graph_gcn import feature_engineering_GCN
# import hickle as hkl
# import os

# ids =  ['1a9m', '1a30', '1apv']
# fn_labels = './indexes/rs_index.csv'
# dt_folder = './examples/'
# storefolder = './example_res/graphgcn'
  
# feat_para = {'feat': ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
#                       'hybridization', 'heavyneighbors', 'heteroneighbors', 
#                       'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring', 
#                       'partialCH'],
#              'max_complex_num_atoms': 200,
#              'int_cutoff': 4, 
#              'adj_type': 'norm',
#              'cov_adj': True, 'cov_adj_num': -1,
#              'ncov_adj': True, 'ncov_adj_num': 1,
#              'ncov_adj_rg': False, 'ncov_adj_num_ranges': []}

# if not os.path.exists(storefolder):
#     os.makedirs(storefolder)
# fe = feature_engineering_GCN(ids = ids, fn_labels = fn_labels, dt_folder = dt_folder, para = feat_para)

# hkl.dump(fe[0], storefolder + '/Xtrain_gzip.hkl', mode = 'w', compression = 'gzip')
# hkl.dump(fe[1], storefolder + '/ytrain_gzip.hkl', mode = 'w', compression = 'gzip')
########################################################################################################



######################################### EXAMPLE 3 ###################################################
# from graph_gcn import Train_GraphBAR, validate_DL_model
# import hickle as hkl
# import os

# storefolder = './example_res/graphgcn'
# feat_para = {'feat': ['atmB', 'atmC', 'atmN', 'atmO', 'atmP', 'atmS', 'atmSe', 'atmHalogen', 'atmMetal',
#                       'hybridization', 'heavyneighbors', 'heteroneighbors', 
#                       'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring', 
#                       'partialCH'],
#              'max_complex_num_atoms': 200,
#              'int_cutoff': 4, 
#              'adj_type': 'norm',
#              'cov_adj': True, 'cov_adj_num': -1,
#              'ncov_adj': True, 'ncov_adj_num': 1,
#              'ncov_adj_rg': False, 'ncov_adj_num_ranges': []}

# Xtrain_data = hkl.load(storefolder + '/Xtrain_gzip.hkl')
# ytrain_data = hkl.load(storefolder + '/ytrain_gzip.hkl')
# validation_data = (hkl.load(storefolder + '/Xvalid_gzip.hkl'), hkl.load(storefolder + '/yvalid_gzip.hkl'))

# mdl_para  = [{'sz': [Xtrain_data[i][0].shape for i in range(len(Xtrain_data))],
#               'adj_num': len(Xtrain_data) - 1,
#               'gcn_lyrs': [3, 4, 5],
#               'l2': 0.01,
#               'training_epochs': [50, 100, 150, 200],
#               'batch_size': [5, 64, 128],
#               'hptune_trials': 40,
#               'hptune_exe_per_trials': 1,
#               'res_dir': storefolder}]
 
# cur_res = {'hp': None}
# print('Hyper-parameter tuning.....................................')
# # Tuning the parameters using Keras Tuner
# # ----------------------------------------------------------------------------------
# (retrain_model, cur_model, cur_res['hp']) = Train_GraphBAR(x_train = Xtrain_data, y_train = ytrain_data, 
#                                                             validation_data = validation_data, para = mdl_para)

# cur_res['train_res'] = validate_DL_model(mdl = cur_model, test_dt_X = Xtrain_data, test_dt_y = ytrain_data)
# cur_res['val_res'] = validate_DL_model(mdl = cur_model, test_dt_X = validation_data[0], test_dt_y = validation_data[1])

# print(cur_res)
# retrain_model.save(storefolder + '/trained_mdl.keras')
########################################################################################################

 
 