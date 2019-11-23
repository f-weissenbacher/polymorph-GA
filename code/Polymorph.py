"""
Class describing a polymorph of a molecule by its internal coordinates.
Provides functions that mutate the genome (== structure) of the polymorph as well as functionality to mate the
polymorph with other instances of the Polymorph class (for gene crossing, inheritance and etc.)

Also provides functions that evaluate different properties (dipole moment, total energy etc.) of the polymorph (based
on QM-calculations)

"""

import itertools
import os
import chemcoord as cc
from collections import Iterable, Collection
import numpy as np

from Mutators import FullRangeMutator

import ase
import pyscf

# keys for property dictionaries
TOTAL_ENERGY = "total_energy"
DIPOLE_MOMENT = "dipole_moment"
IONIZATION_ENERGY = "ionization_energy"


class GeneticAlgorithm:
    def __init__(self):
        self.generation_number = 0
        self.current_generation = list() # Current generation of polymorphs


    def removeLeastFittest(self):
        """"Removes least fittest polymorphs from the current generation"""
        pass

    def selectFromDistribution(self, type='fermi'):
        """ Selects polymorphs based on a probability distribution """
        pass


class PolymorphFactory:
    def __init__(self, base_structure_filepath : str, default_mutation_rate=1e-2, default_crossover_rate=1e-3):
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

        self.zmat_base = self.base_structure.to_zmat()
        # Defaults for mutation behavior
        self.default_mutation_rate = default_mutation_rate
        self.default_crossover_rate = default_crossover_rate

        # Degrees of freedom -> Define genome
        self.resetDegreesOfFreedom()

        self.n_atoms = len(self.zmat_base.index)

        # Allowed ranges for bond lengths, angles and dihedrals used in polymorph generation
        self.bond_value_range = [1.0, 3.0] # Angstrom
        self.angle_value_range = [0, 180] # Degrees
        self.dihedral_value_range = [-180, 180] # Degrees

        # Allowed ranges for single mutations of bond lengths, angles and dihedrals
        self.bond_mutation_range = [0.5, 1.5]
        self.angle_mutation_range = [-30, 30]
        self.dihedral_mutation_range = [-30, 30]

        self.bond_mutator = None
        self.angle_mutator = None
        self.dihedral_mutator = None

    @property
    def mutable_bonds(self):
        return zip(self._mutable_bonds_idxs, self.zmat_base.loc[self._mutable_bonds_idxs, 'b'])

    @property
    def mutable_angles(self):
        return zip(self._mutable_angles_idxs, self.zmat_base.loc[self._mutable_angles_idxs,['b','a']])

    @property
    def mutable_dihedrals(self):
        return zip(self._mutable_dihedrals_idxs, self.zmat_base.loc[self._mutable_dihedrals_idxs,['b','a','d']])

    def resetDegreesOfFreedom(self):
        self._mutable_bonds_idxs = self.zmat_base.index[1:] # All bonds (first bond appears in line 2)
        self._mutable_angles_idxs = self.zmat_base.index[2:] # All angles (first angle appears in line 3)
        self._mutable_dihedrals_idxs = self.zmat_base.index[3:] # All dihedrals (first dihedral is in line 4)

    # Setting degrees of freedom ----------------------------------------------------
    def freezeBonds(self, bonds_to_freeze):
        if bonds_to_freeze == 'all':
            self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop(self._mutable_bonds_idxs)

        elif isinstance(bonds_to_freeze, Collection):
            for bond in bonds_to_freeze:
                if isinstance(bond, Collection) and len(bond) == 2: # bond == Pair of atom indices
                    for atom1, atom2 in self.mutable_bonds:
                        bond = tuple(bond)
                        if (atom1, atom2) == bond or (atom2, atom1) == bond:
                            self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop([atom1])

                elif isinstance(bond, int) and bond in self._mutable_bonds_idxs: # bond that belongs to atom <bond>
                    self._mutable_bonds_idxs = self._mutable_bonds_idxs.drop([bond])

    def freezeAngles(self, angles_to_freeze):
        if angles_to_freeze == 'all':
            self._mutable_angles_idxs = self._mutable_angles_idxs.drop(self._mutable_angles_idxs)

        elif isinstance(angles_to_freeze, Collection):
            for angle in angles_to_freeze:
                if isinstance(angle, Collection) and len(angle) == 3: # bond == Pair of atom indices
                    for free_angle in self.mutable_angles:
                        if np.all(np.in1d(angle, free_angle)):
                            self._mutable_angles_idxs = self._mutable_angles_idxs.drop([free_angle[0]])

                elif isinstance(angle, int) and angle in self._mutable_angles_idxs: # bond that belongs to atom <bond>
                    self._mutable_angles_idxs = self._mutable_angles_idxs.drop([angle])

    def freezeDihedrals(self, dihedrals_to_freeze):
        if dihedrals_to_freeze == 'all':
            self._mutable_dihedrals_idxs = self._mutable_dihedrals_idxs.drop(self._mutable_dihedrals_idxs)

        elif isinstance(dihedrals_to_freeze, Collection):
            for dihedral in dihedrals_to_freeze:
                if isinstance(dihedral, Collection) and len(dihedral) == 4: # bond == Pair of atom indices
                    for free_dihedral in self.mutable_dihedrals:
                        if np.all(np.in1d(dihedral, free_dihedral)):
                            self._mutable_dihedrals_idxs = self._mutable_dihedrals_idxs.drop([free_dihedral[0]])

                elif isinstance(dihedral, int) and dihedral in self._mutable_dihedrals_idxs: # bond that belongs to atom <bond>
                    self._mutable_dihedrals_idxs = self._mutable_bonds_idxs.drop([dihedral])

    # Mutators ---------------------------------------------------------------------
    def setupDefaultMutators(self):
        self.bond_mutator = FullRangeMutator('bond', self.bond_value_range)
        self.angle_mutator = FullRangeMutator('angle', self.angle_value_range)
        self.dihedral_mutator = FullRangeMutator('dihedral', self.dihedral_value_range)

    # Generation of polymorphs -----------------------------------------------------
    def generateRandomPolymorph(self):
        zmat = self.zmat_base.copy()

        # Set bonds randomly
        for bond_index in self._mutable_bonds_idxs:
            new_length = self.bond_value_range[0] + np.random.rand() * np.diff(self.bond_value_range)[0]
            zmat.safe_loc[bond_index, 'bond'] = new_length
        # Set mutable angles randomly
        for angle_index in self._mutable_angles_idxs:
            new_angle = self.angle_value_range[0] + np.random.rand() * np.diff(self.angle_value_range)[0]
            zmat.safe_loc[angle_index, 'angle'] = new_angle
        # Set mutable dihedrals randomly
        for dihedral_index in self._mutable_dihedrals_idxs:
            new_dihedral = self.dihedral_value_range[0] + np.random.rand() * np.diff(self.dihedral_value_range)[0]
            zmat.safe_loc[dihedral_index, 'dihedral'] = new_dihedral

        return Polymorph(zmat, self.bond_mutator, self.angle_mutator, self.dihedral_mutator,
                         self._mutable_bonds_idxs, self._mutable_angles_idxs, self._mutable_dihedrals_idxs,
                         self.default_crossover_rate)


class Polymorph:
    # Function to generate new, unique IDs
    _generate_id = itertools.count().__next__

    def __init__(self, zmatrix, bond_mutator, angle_mutator, dihedral_mutator,
                 mutable_bonds=None, mutable_angles=None, mutable_dihedrals=None,
                 crossover_rate=1e-3, name="", generation_number=-1):

        self.name = name
        self.id = Polymorph._generate_id()
        self.generation_number = generation_number
        self.properties = dict()
        self.bond_mutator = bond_mutator
        self.angle_mutator = angle_mutator
        self.dihedral_mutator = dihedral_mutator
        self.crossover_probability = crossover_rate
        self.zmat = zmatrix
        self._mutable_bonds = mutable_bonds
        self._mutable_angles = mutable_angles
        self._mutable_dihedrals = mutable_dihedrals

        if mutable_bonds is None:
            self._mutable_bonds = self.zmat.index[1:]

        if mutable_angles is None:
            self._mutable_angles = self.zmat.index[2:]

        if mutable_dihedrals is None:
            self._mutable_dihedrals = self.zmat.index[3:]


    @property
    def genome(self):
        return {'bondlengths' : self.zmat.loc[self._mutable_bonds,'bond'],
                'angles' : self.zmat.loc[self._mutable_angles, 'angle'],
                'dihedrals' : self.zmat.loc[self._mutable_dihedrals,'dihedral']}

    @property
    def bonds(self):
        return zip(self.zmat.index[1:], self.zmat.loc[1:,'b'])

    @property
    def structure(self):
        return self.zmat.get_cartesian()

    @property
    def zmat_string(self):
        return self.zmat.to_string(header=False, upper_triangle=False, index=False)

    def __str__(self):
        text = f"Polymorph '{self.name}', ID: {self.id}"
        return text

    def saveStructure(self, filename):
        self.structure.to_xyz(filename)

    def mateWith(self, partner):
        """ Creates an offspring polymorph by mating two polymorphs
        Both involved polymorphs are assumed to share the same type of genome """

        new_zmat = self.zmat.copy()

        for bond_index in self._mutable_bonds:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[bond_index, 'bond'] = partner.zmat.loc[bond_index, 'bond']

        for angle_index in self._mutable_angles:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[angle_index, 'angle'] = partner.zmat.loc[angle_index, 'angle']

        for dihedral_index in self._mutable_dihedrals:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[dihedral_index, 'dihedral'] = partner.zmat.loc[dihedral_index, 'dihedral']

        return Polymorph(new_zmat, self.bond_mutator, self.angle_mutator, self.dihedral_mutator,
                         self._mutable_bonds, self._mutable_angles, self._mutable_dihedrals,
                         self.crossover_probability)


    def crossoverGenesWith(self, partner):
        """ Attempts gene crossovers with other polymorph 'partner' based on crossover probability. Changes
        are directly written to the zmatrices of both involved polymorphs"""

        for bond_index in self._mutable_bonds:
            if np.random.rand() < self.crossover_probability:
                own_bondlength = self.zmat.loc[bond_index, 'bond']
                self.zmat.safe_loc[bond_index, 'bond'] = partner.zmat.loc[bond_index, 'bond']
                partner.zmat.safe_loc[bond_index, 'bond'] = own_bondlength

        for angle_index in self._mutable_angles:
            if np.random.rand() < 0.5:
                own_angle = self.zmat.loc[angle_index, 'angle']
                self.zmat.safe_loc[angle_index, 'angle'] = partner.zmat.loc[angle_index, 'angle']
                partner.zmat.safe_loc[angle_index, 'angle'] = own_angle

        for dihedral_index in self._mutable_dihedrals:
            if np.random.rand() < 0.5:
                own_dihedral = self.zmat.loc[dihedral_index, 'dihedral']
                self.zmat.safe_loc[dihedral_index, 'dihedral'] = partner.zmat.loc[dihedral_index, 'dihedral']
                partner.zmat.safe_loc[dihedral_index, 'dihedral'] = own_dihedral


    def checkDistances(self):
        structure = self.structure


    def runHartreeFock(self):
        zmat_str = self.zmat_string


    def runSemiempiricalQM(self):
        zmat_str = self.zmat_string





  
  
  
  
  

if __name__ == "__main__":
  # Test polymorph class
  pass
#  for k in range(10):
#    pm = Polymorph(f"Polymorph {k}")
#    print(pm)