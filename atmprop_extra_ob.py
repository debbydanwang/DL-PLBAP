#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:53:50 2021

@author: debbywang
"""
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import PeriodicTable
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_molecule(molecule_file, format = 'pdb'):
  """Load a molecule.
  Parameters:
    molecule_file - filename for molecule
    format - file format
  Returns a pybel object.
  Note This function requires openbabel to be installed.
  """
  try:
    mol = next(pybel.readfile(format = format, filename = molecule_file))
  except:
    raise ValueError('Error loading molecular file: %s' % molecule_file)

  mol.removeh() # remove the hydrogens first
  mol.addh() # add hydrogens using openbabel
  return mol

class ComProp(object):
    def __init__(self, fn_pro, fn_lig, 
                 fmat_pro = 'pdb', fmat_lig = 'pdb', 
                 promol = None,
                 ID = None):
        """
        Initialize an ComProp class.
        Parameters:
            fn_pro - file name of the protein
            fn_lig - file name of the ligand
            fmat_pro - file format for the protein
            fmat_lig - file format for the ligand
            promol - molecule loaded before in case of large protein molecule
            ID - ID of the complex
        """
        self.ID = ID if ID is not None else "PL"
        print('Constructing an ComProp object for %s.........\n' % self.ID)
        # read in protein and ligand files ------------------------
        if promol is not None:
           self.mols = (promol, load_molecule(fn_lig, fmat_lig))
        else:
           self.mols = (load_molecule(fn_pro, fmat_pro), load_molecule(fn_lig, fmat_lig))
        # center at the ligand center -----------------------------
        coor = (np.array([a.coords for a in self.mols[0].atoms]),
                np.array([b.coords for b in self.mols[1].atoms]))
        self.ligcenter = np.mean(coor[1], axis = 0) 
        self.coor = (coor[0] - self.ligcenter, coor[1] - self.ligcenter)
        # # identify interacting area
        # self.pd = cdist(self.coor[0], self.coor[1], metric = 'euclidean')

    def extract_atom_prop(self, molid = None):
        """
        Extract atom properties and save the informaiton.
        Parameters:
            molid - 0: protein, 1: ligand, None: both
        """
        dtdic = {'id': [], 'atmnum': [], 'x': [], 'y': [], 'z': [], 
                 'atmB': [], 'atmC': [], 'atmN': [], 'atmO': [], 'atmP': [], 'atmS': [], 
                 'atmSe': [], 'atmHalogen': [], 'atmMetal': [], 'atmMetallic': [], # 'atmMetal': all metals, 'atmMetallic': Mg, Zn, Mn, Ca or Fe
                 'hybridization': [],
                 'heavyneighbors': [], 'heteroneighbors': [],
                 'hydrophobic': [], 'aromatic': [], 'acceptor': [], 'donor': [], 'ring': [], 
                 'partialCH': [],
                 'posionizable': [], 'negionizable': [], 'exlvolume': [], 'vdwrad': [], 
                 # excluded volume is 4 times the actual volume (4/3*pi*r^3, r is atom radius)
                 'moltype': [],
                 'neighbors(nbr:idx--anum--(sbond,dbond,tbond,arombond,ringbond))': []}
        nms = {0: 'protein', 1: 'ligand'}

        if molid is not None:
          print('characterizing %s....................' % nms[molid])
          mol = self.mols[molid]
          mol_coor = self.coor[molid]
          ss = pybel.Smarts('[c,C]')
          hydrophobic_atms = [kk[0] for kk in ss.findall(mol)]
          for i in range(mol_coor.shape[0]):
              if i % 1000 == 0:
                print('processing %d out of %d atoms.........' % (i, mol_coor.shape[0]))
              atm = mol.atoms[i]
              resname = atm.OBAtom.GetResidue().GetName()
              if resname != 'HOH':
                  dtdic['id'].append(atm.idx)
                  atm_anum = atm.atomicnum
                  dtdic['atmnum'].append(atm_anum)
                  dtdic['x'].append('%.3f' % mol_coor[i, 0])
                  dtdic['y'].append('%.3f' % mol_coor[i, 1])
                  dtdic['z'].append('%.3f' % mol_coor[i, 2])
                  # moltype (ligand or protein) ----------------------------------------------------------
                  dtdic['moltype'].append(molid)
                  # extract specific atom types -----------------------------------------------------------
                  atm_B = 1 if atm_anum == 5 else 0
                  dtdic['atmB'].append(atm_B)
                  atm_C = 1 if atm_anum == 6 else 0
                  dtdic['atmC'].append(atm_C)
                  atm_N = 1 if atm_anum == 7 else 0
                  dtdic['atmN'].append(atm_N)
                  atm_O = 1 if atm_anum == 8 else 0
                  dtdic['atmO'].append(atm_O)
                  atm_P = 1 if atm_anum == 15 else 0
                  dtdic['atmP'].append(atm_P)
                  atm_S = 1 if atm_anum == 16 else 0
                  dtdic['atmS'].append(atm_S)
                  atm_Se = 1 if atm_anum == 34 else 0
                  dtdic['atmSe'].append(atm_Se)
                  atm_halogen = 1 if atm_anum in (9, 17, 35, 53, 85, 117) else 0
                  dtdic['atmHalogen'].append(atm_halogen)
                  ck = (atm_anum in [3, 4, 11, 12, 13, 19, 20, 37, 38, 55, 56, 87, 88] or 22<=atm_anum<=32 or 40<=atm_anum<=51 or 72<=atm_anum<=84 or 104<=atm_anum<=116)
                  atm_metal = 1 if ck else 0
                  dtdic['atmMetal'].append(atm_metal)
                  # atom hybridization -------------------------------------------------------------------
                  dtdic['hybridization'].append(atm.hyb)
                  # connections to heavy atoms and heteroatoms -------------------------------------------
                  dtdic['heavyneighbors'].append(atm.heavydegree)
                  dtdic['heteroneighbors'].append(atm.heterodegree)                       
                  # pharmacorphoric prop - hydrophobic, aromatic, acceptor, donor and ring --------------------
                  dtdic['hydrophobic'].append(int(i in hydrophobic_atms))
                  dtdic['aromatic'].append(int(atm.OBAtom.IsAromatic()))
                  dtdic['donor'].append(int(atm.OBAtom.IsHbondDonor()))
                  dtdic['acceptor'].append(int(atm.OBAtom.IsHbondAcceptor()))
                  dtdic['ring'].append(int(atm.OBAtom.IsInRing()))
                  # partial charge -----------------------------------------------------------------------
                  pch = atm.partialcharge
                  dtdic['partialCH'].append('%.3f' % pch)
                  # extract specific atom types -----------------------------------------------------------
                  dtdic['atmMetallic'].append(int(atm_anum in [12, 20, 25, 26, 30]))
                  # partial charge 2 ----------------------------------------------------------------------
                  dtdic['posionizable'].append(int(pch > 0))
                  dtdic['negionizable'].append(int(pch < 0))
                  # excluded volume of atom --------------------------------------------------------------
                  rad = PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atm_anum)
                  dtdic['exlvolume'].append('%.3f' % (4*4/3*np.pi*rad**3))
                  dtdic['vdwrad'].append(rad)
                  # record neighbors ----------------------------------------------------------------------
                  nbrs_str = ''
                  for bd in ob.OBAtomBondIter(atm.OBAtom): 
                      nbr = bd.GetNbrAtom(atm.OBAtom)
                      nbridx = nbr.GetIdx()
                      nbr_anum = nbr.GetAtomicNum()
                      bdod = bd.GetBondOrder()
                      bd_single = 1 if (bdod == 1) else 0
                      bd_double = 1 if (bdod == 2) else 0
                      bd_triple = 1 if (bdod == 3) else 0
                      bd_aromatic = int(bd.IsAromatic())
                      bd_ring = int(bd.IsInRing())
                      nbrs_str += ('nbr:%d--%d--(%d,%d,%d,%d,%d)//' % (nbridx, nbr_anum, bd_single, bd_double, bd_triple, bd_aromatic, bd_ring))

                  dtdic['neighbors(nbr:idx--anum--(sbond,dbond,tbond,arombond,ringbond))'].append(nbrs_str)
        else: 
          for mol_ind in range(2):
              print('characterizing %s....................' % nms[mol_ind])
              mol = self.mols[mol_ind]
              mol_coor = self.coor[mol_ind]
              ss = pybel.Smarts('[c,C]')
              hydrophobic_atms = [kk[0] for kk in ss.findall(mol)]
              for i in range(mol_coor.shape[0]):
                  if i % 1000 == 0:
                    print('processing %d out of %d atoms.........' % (i, mol_coor.shape[0]))
                  atm = mol.atoms[i]
                  resname = atm.OBAtom.GetResidue().GetName()
                  if resname != 'HOH':
                      dtdic['id'].append(atm.idx)
                      atm_anum = atm.atomicnum
                      dtdic['atmnum'].append(atm_anum)
                      dtdic['x'].append('%.3f' % mol_coor[i, 0])
                      dtdic['y'].append('%.3f' % mol_coor[i, 1])
                      dtdic['z'].append('%.3f' % mol_coor[i, 2])
                      # moltype (ligand or protein) ----------------------------------------------------------
                      dtdic['moltype'].append(mol_ind)
                      # extract specific atom types -----------------------------------------------------------
                      atm_B = 1 if atm_anum == 5 else 0
                      dtdic['atmB'].append(atm_B)
                      atm_C = 1 if atm_anum == 6 else 0
                      dtdic['atmC'].append(atm_C)
                      atm_N = 1 if atm_anum == 7 else 0
                      dtdic['atmN'].append(atm_N)
                      atm_O = 1 if atm_anum == 8 else 0
                      dtdic['atmO'].append(atm_O)
                      atm_P = 1 if atm_anum == 15 else 0
                      dtdic['atmP'].append(atm_P)
                      atm_S = 1 if atm_anum == 16 else 0
                      dtdic['atmS'].append(atm_S)
                      atm_Se = 1 if atm_anum == 34 else 0
                      dtdic['atmSe'].append(atm_Se)
                      atm_halogen = 1 if atm_anum in (9, 17, 35, 53, 85, 117) else 0
                      dtdic['atmHalogen'].append(atm_halogen)
                      ck = (atm_anum in [3, 4, 11, 12, 13, 19, 20, 37, 38, 55, 56, 87, 88] or 22<=atm_anum<=32 or 40<=atm_anum<=51 or 72<=atm_anum<=84 or 104<=atm_anum<=116)
                      atm_metal = 1 if ck else 0
                      dtdic['atmMetal'].append(atm_metal)
                      # atom hybridization -------------------------------------------------------------------
                      dtdic['hybridization'].append(atm.hyb)
                      # connections to heavy atoms and heteroatoms -------------------------------------------
                      dtdic['heavyneighbors'].append(atm.heavydegree)
                      dtdic['heteroneighbors'].append(atm.heterodegree)                       
                      # pharmacorphoric prop - hydrophobic, aromatic, acceptor, donor and ring --------------------
                      dtdic['hydrophobic'].append(int(i in hydrophobic_atms))
                      dtdic['aromatic'].append(int(atm.OBAtom.IsAromatic()))
                      dtdic['donor'].append(int(atm.OBAtom.IsHbondDonor()))
                      dtdic['acceptor'].append(int(atm.OBAtom.IsHbondAcceptor()))
                      dtdic['ring'].append(int(atm.OBAtom.IsInRing()))
                      # partial charge -----------------------------------------------------------------------
                      pch = atm.partialcharge
                      dtdic['partialCH'].append('%.3f' % pch)
                      # extract specific atom types -----------------------------------------------------------
                      dtdic['atmMetallic'].append(int(atm_anum in [12, 20, 25, 26, 30]))
                      # partial charge 2 ----------------------------------------------------------------------
                      dtdic['posionizable'].append(int(pch > 0))
                      dtdic['negionizable'].append(int(pch < 0))
                      # excluded volume of atom --------------------------------------------------------------
                      rad = PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atm_anum)
                      dtdic['exlvolume'].append('%.3f' % (4*4/3*np.pi*rad**3))
                      dtdic['vdwrad'].append(rad)
                      # record neighbors ----------------------------------------------------------------------
                      nbrs_str = ''
                      for bd in ob.OBAtomBondIter(atm.OBAtom): 
                          nbr = bd.GetNbrAtom(atm.OBAtom)
                          nbridx = nbr.GetIdx()
                          nbr_anum = nbr.GetAtomicNum()
                          bdod = bd.GetBondOrder()
                          bd_single = 1 if (bdod == 1) else 0
                          bd_double = 1 if (bdod == 2) else 0
                          bd_triple = 1 if (bdod == 3) else 0
                          bd_aromatic = int(bd.IsAromatic())
                          bd_ring = int(bd.IsInRing())
                          nbrs_str += ('nbr:%d--%d--(%d,%d,%d,%d,%d)//' % (nbridx, nbr_anum, bd_single, bd_double, bd_triple, bd_aromatic, bd_ring))

                      dtdic['neighbors(nbr:idx--anum--(sbond,dbond,tbond,arombond,ringbond))'].append(nbrs_str)

        return pd.DataFrame(dtdic)      

    def mod_dtdict_for_protein(self, dtdict):
        """
        In screening tasks, we just calculate the properties for the target once, but need to modify the coordinates (located in each ligand center).
        Parameters:
            dtdict - the drived feature dictionary for the protein
        """
        ids = [i-1 for i in dtdict['id'].tolist()]
        tmpdict = dtdict
        tmpdict['x'] = ['%.3f' % crd for crd in self.coor[0][ids, 0]]
        tmpdict['y'] = ['%.3f' % crd for crd in self.coor[0][ids, 1]]
        tmpdict['z'] = ['%.3f' % crd for crd in self.coor[0][ids, 2]]

        return tmpdict
