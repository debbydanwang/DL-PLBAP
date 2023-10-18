#############################################################################
#                NOTEs for the DL-PLBAP repository                          #
#############################################################################

1. Requirements
   DL-PLBAP currently supports a Linux system with Python 3.7 (or above), and requires main dependency packages as follows. 
   (1) For generating molecular representations:
       - openbabel (https://open-babel.readthedocs.io/en/latest/UseTheLibrary/PythonInstall.html)
       - rdkit (https://www.rdkit.org/)
       - open3d (http://www.open3d.org/docs/release/getting_started.html)
   (2) For training deep-learning models:
       - tensorflow-gpu with version 2.6.0 or above (https://www.tensorflow.org/)
       - keras_tuner (https://keras.io/keras_tuner/)
       - scipy (https://www.scipy.org/)
       - sklearn (https://scikit-learn.org/stable/)
   (3) General packages:
       - numpy (https://numpy.org/)		
       - pandas (https://pandas.pydata.org/)
       - hickle (https://pypi.org/project/hickle/)

2. Modules in DL-PLBAP repository 
   - atmprop_extra_ob: Extrating the atom-properties for molecules.
   - acnn: Generating feature representations and training an ACNN PLBAP model.
   - imc_cnn: Generating IMC representations and training an IMC-CNN PLBAP model.
   - grid_cnn: Generating grid representations and training a GRID-CNN PLBAP model.
   - graph_gcn: Generating graph representations and training a GRAPH-GCN PLBAP model.

3. Data downloading and preprocessing
	(1) Downloading:
	    - Model construction: Download the PDBbind refined set (http://www.pdbbind.org.cn/) to a folder (e.g. 'refined').
	    - Validation (hyperparameter tuning): Download the PDBbind core set (http://www.pdbbind.org.cn/) to a folder (e.g. 'core').
	    - Testing: Download the CSAR-HiQ sets 1 and 2 (http://www.csardock.org/) to a folder (e.g. 'test1' and 'test2').

	(2) Preprocessing:
	    - PDBbind refined/core sets: Save the ligand files as PDB files (e.g. using software like UCSF Chimera).
	    - CSAR-HiQ sets: Save the protein and ligand in each complex as PDB files (e.g. using software like UCSF Chimera), 
	   		     and name these files as those in PDBbind refined/core sets (e.g. 1ax1_protein.pdb, 1ax1_ligand.pdb in '1ax1' folder).
	
	(3) Data folder format:
	    - Each data folder ('refined', 'core', 'test1' and 'test2') includes subfolders corresponding to the protein-ligand complexes (indexed by PDB ID).
	      For example, 'refined' folder includes '1a9m' subfolder that further includes the PDB files for the protein ('1a9m_protein.pdb') and ligand ('1a9m_ligand.pdb').
	   		   
	(4) Index files:
	    - The 'indexes' folder inside this repository includes the label information for all the complexes.
	      For example, 'rs_index.csv' file includes the information (e.g. ID, -logKd/Ki and protein name) for all complexes in the PDBbind refined set.

3. Procedure for training a deep-learning PLBAP model
   (1) Starting from a data folder, we can first extract the atom properties for each complex in this data set. This saves much time in the model-construction process.
       - Import ComProp class from the atmprop_extra_ob package, and generate the atom properties for a given complex using customized parameters.
         A text file will be created in order to store the atom properties (e.g. '1a9m_atm_prop.txt').
   (2) Generate molecular representations (e.g. grids or graphs):
       - Import the feature representation class (e.g. feature_engineering_GCN) from the corresponding package (e.g. graph_gcn), and generate the representations to be learned by deep-learning models.
         You can save the generated tensors (representations) using hickle, such as 'Xtrain_gzip.hkl' and 'ytrain_gzip.hkl' for the data set, for easier training of multiple models later.
   (3) Training a deep-learning model:
       - Import the model-training and -validation modules (e.g. Train_GraphBAR and validate_DL_model) from the corresponding package (e.g. graph_gcn), and train a model using  customized parameters. NOTE that you need to provide a validation set (e.g. PDBbind core set) for parameter tuning when training a model, and remember to remove those complexes residing in the training set.
         You can save the trained model (e.g. 'trained_mdl.keras') for easier reloading later.
	   
4. Example codes are provided in the 'example.py' in this repository
   (1) 'examples' folder - It provides a small data set of 3 complexes ('1a9m', '1a30' and '1apv'), and each subfolder stores the protein and ligand PDB file for the complex.
   (2) 'example.py':
       - Example 1: Generating the atom-property file for a complex.
       - Example 2: Generating the graph representations for the small data set in the 'examples' folder. The generated tensors can be saved to 'example_res' folder.
       - Example 3: Training a GRAPH-GCN model for PLBAP. The trained model can be saved to 'example_res' folder.
   
   
   
   
   
