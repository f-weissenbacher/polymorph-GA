import os
import chemcoord as cc
from collections import Collection

import numpy as np

from Utilities import checkAtomDistances
from Polymorph import Polymorph
from Mutators import FullRangeMutator, PlaceboMutator


class PolymorphFactory:
    def __init__(self, base_structure_filepath: str, default_mutation_rate=1e-2, default_crossover_rate=1e-3,
                 polymer_name="Polymer"):
        # Load base structure as cc.Cartesian
        if os.path.isfile(base_structure_filepath):
            filename, ext = os.path.splitext(base_structure_filepath)
            if ext == ".xyz":
                self.base_structure = cc.Cartesian.read_xyz(base_structure_filepath)
            elif ext == ".json":
                self.base_structure = cc.Cartesian.read_cjson(base_structure_filepath)
            else:
                raise ValueError(f"Unknown file extension {ext} for base structure file. " + \
                                 "Accepted formats are '.xyz' and '.json'")
        else:
            raise FileNotFoundError("Unable to load base molecular structure. " + \
                                    f"File {base_structure_filepath} cannot be found.")
        
        self.zmat_base = self.base_structure.get_zmat()
        self.polymer_name = polymer_name
        # Defaults for mutation behavior
        self.default_mutation_rate = default_mutation_rate
        self.default_crossover_rate = default_crossover_rate
        
        # Degrees of freedom -> Define genome
        self.resetDegreesOfFreedom()
        self.n_atoms = len(self.zmat_base.index)
        
        # Allowed ranges for bond lengths, angles and dihedrals used in polymorph generation
        self.bond_value_range = [1.0, 3.0]  # Angstrom
        self.angle_value_range = [0, 180]  # Degrees
        self.dihedral_value_range = [-180, 180]  # Degrees
        
        # Allowed ranges for single mutations of bond lengths, angles and dihedrals
        self.bond_mutation_range = [0.5, 1.5]
        self.angle_mutation_range = [-30, 30]
        self.dihedral_mutation_range = [-30, 30]
        
        self.bond_mutator = None
        self.angle_mutator = None
        self.dihedral_mutator = None
        
        Polymorph.resetIdCounter()
        self._createBasePolymorph()
    
    @property
    def mutable_bonds(self):
        return zip(self._mutable_bonds_idxs, self.zmat_base.loc[self._mutable_bonds_idxs, 'b'])
    
    @property
    def mutable_angles(self):
        return zip(self._mutable_angles_idxs, self.zmat_base.loc[self._mutable_angles_idxs, ['b', 'a']])
    
    @property
    def mutable_dihedrals(self):
        return zip(self._mutable_dihedrals_idxs, self.zmat_base.loc[self._mutable_dihedrals_idxs, ['b', 'a', 'd']])
    
    def resetDegreesOfFreedom(self):
        self._mutable_bonds_idxs = self.zmat_base.index[1:]  # All bonds (first bond appears in line 2)
        self._mutable_angles_idxs = self.zmat_base.index[2:]  # All angles (first angle appears in line 3)
        self._mutable_dihedrals_idxs = self.zmat_base.index[3:]  # All dihedrals (first dihedral is in line 4)
    
    # Setting degrees of freedom ----------------------------------------------------
    def freezeBonds(self, bonds_to_freeze):
        if bonds_to_freeze == 'all':
            self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop(self._mutable_bonds_idxs)
        
        elif isinstance(bonds_to_freeze, Collection):
            for bond in bonds_to_freeze:
                if isinstance(bond, Collection) and len(bond) == 2:  # bond == Pair of atom indices
                    for atom1, atom2 in self.mutable_bonds:
                        bond = tuple(bond)
                        if (atom1, atom2) == bond or (atom2, atom1) == bond:
                            self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop([atom1])
                
                elif isinstance(bond, int) and bond in self._mutable_bonds_idxs:  # bond that belongs to atom <bond>
                    self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop([bond])
    
    def freezeAngles(self, angles_to_freeze):
        if angles_to_freeze == 'all':
            self._mutable_angles_idxs = self._mutable_angles_idxs.drop(self._mutable_angles_idxs)
        
        elif isinstance(angles_to_freeze, Collection):
            for angle in angles_to_freeze:
                if isinstance(angle, Collection) and len(angle) == 3:  # bond == Pair of atom indices
                    for free_angle in self.mutable_angles:
                        if np.all(np.in1d(angle, free_angle)):
                            self._mutable_angles_idxs = self._mutable_angles_idxs.drop([free_angle[0]])
                
                elif isinstance(angle, int) and angle in self._mutable_angles_idxs:  # bond that belongs to atom <bond>
                    self._mutable_angles_idxs = self._mutable_angles_idxs.drop([angle])
    
    def freezeDihedrals(self, dihedrals_to_freeze):
        if dihedrals_to_freeze == 'all':
            self._mutable_dihedrals_idxs = self._mutable_dihedrals_idxs.drop(self._mutable_dihedrals_idxs)
        
        elif isinstance(dihedrals_to_freeze, Collection):
            for dihedral in dihedrals_to_freeze:
                if isinstance(dihedral, Collection) and len(dihedral) == 4:  # bond == Pair of atom indices
                    for free_dihedral in self.mutable_dihedrals:
                        if np.all(np.in1d(dihedral, free_dihedral)):
                            self._mutable_dihedrals_idxs = self._mutable_dihedrals_idxs.drop([free_dihedral[0]])
                
                elif isinstance(dihedral,
                                int) and dihedral in self._mutable_dihedrals_idxs:  # bond that belongs to atom <bond>
                    self._mutable_dihedrals_idxs = self._mutable_bonds_idxs.drop([dihedral])
    
    # Mutators ---------------------------------------------------------------------
    def setupDefaultMutators(self):
        self.bond_mutator = FullRangeMutator('bond', self.bond_value_range)
        self.angle_mutator = FullRangeMutator('angle', self.angle_value_range)
        self.dihedral_mutator = FullRangeMutator('dihedral', self.dihedral_value_range)
    
    # Generation of polymorphs -----------------------------------------------------
    def generateRandomPolymorph(self, valid_structure_only=True, n_max_restarts=2):
        
        for k in range(n_max_restarts):
            zmat = self.zmat_base.copy()
            mutation_succeeded = False
            
            # Set bonds randomly
            for bond_index in self._mutable_bonds_idxs:
                old_length = zmat.loc[bond_index, 'bond']
                new_length = self.bond_value_range[0] + np.random.rand() * np.diff(self.bond_value_range)[0]
                zmat.safe_loc[bond_index, 'bond'] = new_length
                if not valid_structure_only or checkAtomDistances(zmat):
                    mutation_succeeded = True
                else:
                    zmat.safe_loc[bond_index, 'bond'] = old_length
            
            # Set mutable angles randomly
            for angle_index in self._mutable_angles_idxs:
                old_angle = zmat.loc[angle_index, 'angle']
                new_angle = self.angle_value_range[0] + np.random.rand() * np.diff(self.angle_value_range)[0]
                zmat.safe_loc[angle_index, 'angle'] = new_angle
                if not valid_structure_only or checkAtomDistances(zmat):
                    mutation_succeeded = True
                else:
                    zmat.safe_loc[angle_index, 'angle'] = old_angle
            
            # Set mutable dihedrals randomly
            for dihedral_index in self._mutable_dihedrals_idxs:
                old_dihedral = zmat.loc[dihedral_index, 'dihedral']
                new_dihedral = self.dihedral_value_range[0] + np.random.rand() * np.diff(self.dihedral_value_range)[0]
                zmat.safe_loc[dihedral_index, 'dihedral'] = new_dihedral
                if not valid_structure_only or checkAtomDistances(zmat):
                    mutation_succeeded = True
                else:
                    zmat.safe_loc[dihedral_index, 'dihedral'] = old_dihedral
            
            if mutation_succeeded:
                return Polymorph(zmat, self.bond_mutator, self.angle_mutator, self.dihedral_mutator,
                                 self._mutable_bonds_idxs, self._mutable_angles_idxs, self._mutable_dihedrals_idxs,
                                 self.default_crossover_rate, name=self.polymer_name)
        else:
            print(
                f"Warning: Unable to generate random Polymorph. Reached maximum number of restarts ({n_max_restarts})")
            return None
    
    def _createBasePolymorph(self):
        self.base_polymorph = Polymorph(self.zmat_base,
                                        PlaceboMutator('bond'), PlaceboMutator('angle'), PlaceboMutator('dihedral'),
                                        mutable_bonds=[], mutable_angles=[], mutable_dihedrals=[],
                                        crossover_rate=0.0, name="base structure", generation_number=-1)