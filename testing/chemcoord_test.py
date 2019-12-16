import chemcoord as cc
import os
from os.path import join

molecules_dir = os.path.abspath(join(os.path.dirname(__file__), "../molecules"))
testing_dir = os.path.abspath(join(os.path.dirname(__file__), "../testing"))

cc.settings['defaults']['viewer'] = 'ase-gui'

molecule = cc.Cartesian.read_xyz(join(molecules_dir,"dihedral_tm1.xyz"))

zmat = molecule.get_zmat()

#zmat.safe_loc[0,'angle'] = 180

