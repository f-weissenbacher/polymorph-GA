import os
import chemcoord as cc
from collections import Collection

import numpy as np

from Utilities import checkAtomDistances, minimalDistanceThreshold
from Polymorph import Polymorph
from Mutators import Mutator, PlaceboMutator, IncrementalMutator, MultiplicativeMutator

import configparser

class PolymorphFactory:
    
    generator_defaults = {'bond_scaling_range': [0.5, 2.0],
                          'angle_value_range': [0, 180],  # Degrees
                          'dihedral_value_range': [-180, 180],  # Degrees
                         }
    
    def __init__(self, base_structure_filepath: str, mutation_rate=0.05, crossover_rate=1e-3,
                 polymer_name="Polymer", bond_mutator=None, angle_mutator=None, dihedral_mutator=None,
                 bond_scaling_range=None, angle_value_range=None, dihedral_value_range=None):
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
        
        # Degrees of freedom -> Define genome
        self.resetDegreesOfFreedom()
        self.n_atoms = len(self.zmat_base.index)

        # Defaults for mutation behavior
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Settings for polymorph generator
        if bond_scaling_range is None:
            bond_scaling_range = PolymorphFactory.generator_defaults['bond_scaling_range']
        
        if angle_value_range is None:
            angle_value_range = PolymorphFactory.generator_defaults['angle_value_range']
            
        if dihedral_value_range is None:
            dihedral_value_range = PolymorphFactory.generator_defaults['dihedral_value_range']
            
        self.bond_scaling_range = bond_scaling_range
        self.angle_value_range = angle_value_range
        self.dihedral_value_range = dihedral_value_range

        # Set default mutators
        self.setupDefaultMutators()
        
        # Override default mutators if custom mutators are given
        if isinstance(bond_mutator, Mutator):
            self.bond_mutator = bond_mutator
            
        if isinstance(angle_mutator, Mutator):
            self.angle_mutator = angle_mutator
            
        if isinstance(dihedral_mutator, Mutator):
            self.dihedral_mutator = dihedral_mutator
            
        Polymorph.resetIdCounter()
        self._createBasePolymorph()
        
        print("Running DFT calculation for base polymorph")
        self.base_polymorph.evaluate()
    
    @property
    def mutable_bonds(self):
        return zip(self._mutable_bonds_idxs, self.zmat_base.loc[self._mutable_bonds_idxs, 'b'])
    
    @property
    def mutable_angles(self):
        return zip(self._mutable_angles_idxs, self.zmat_base.loc[self._mutable_angles_idxs, ['b', 'a']])
    
    @property
    def mutable_dihedrals(self):
        return zip(self._mutable_dihedrals_idxs, self.zmat_base.loc[self._mutable_dihedrals_idxs, ['b', 'a', 'd']])
    
    # Saving / Loading --------------------------------------------------------------
    
    def save(self, folder):
        
        configfile_path = os.path.join(folder, "factory.settings")
        structure_path = os.path.join(folder, "base_structure.xyz")
        
        config = configparser.ConfigParser()
        
        factory_settings = {'polymer_name': self.polymer_name}
        config['General Settings'] = factory_settings
        
        
        
    

    
    # Setting degrees of freedom ----------------------------------------------------

    def resetDegreesOfFreedom(self):
        self._mutable_bonds_idxs = self.zmat_base.index[1:]  # All bonds (first bond appears in line 2)
        self._mutable_angles_idxs = self.zmat_base.index[2:]  # All angles (first angle appears in line 3)
        self._mutable_dihedrals_idxs = self.zmat_base.index[3:]  # All dihedrals (first dihedral is in line 4)

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
        
        elif dihedrals_to_freeze == 'all-improper':
            proper_dihedrals_idxs = self.base_polymorph.selectDihedralsByType('proper')
            self._mutable_dihedrals_idxs = proper_dihedrals_idxs
        
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
        self.bond_mutator = PlaceboMutator('bond')
        self.angle_mutator = IncrementalMutator('angle', [0, 180], [-60, 60],
                                                gene_is_periodic=False, mutation_rate=self.mutation_rate)
        self.dihedral_mutator = IncrementalMutator('dihedral', [-180, 180], [-60, 60],
                                                   gene_is_periodic=True, mutation_rate=self.mutation_rate)
    
    # Generation of polymorphs -----------------------------------------------------
    def generateRandomPolymorph(self, valid_structure_only=True, n_max_restarts=10):
        """ Generates a random polymorph by running full range mutations on the base polymorph """
        
        for k in range(n_max_restarts):
            zmat = self.zmat_base.copy()
            mutation_succeeded = False
            
            # Set bonds randomly
            for bond_index in self._mutable_bonds_idxs:
                old_length = zmat.loc[bond_index, 'bond']
                new_length = old_length * ( self.bond_scaling_range[0] +
                                            np.random.rand() * np.diff(self.bond_scaling_range)[0])
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
                                 self.crossover_rate, name=self.polymer_name)
        else:
            print(f"Warning: Unable to generate random Polymorph. Reached maximum number of restarts ({n_max_restarts})")
            return None
    
    def _createBasePolymorph(self):
        self.base_polymorph = Polymorph(self.zmat_base,
                                        PlaceboMutator('bond'), PlaceboMutator('angle'), PlaceboMutator('dihedral'),
                                        mutable_bonds=[], mutable_angles=[], mutable_dihedrals=[],
                                        crossover_rate=0.0, name="base polymorph")