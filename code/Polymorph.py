"""
Class describing a polymorph of a molecule by its internal coordinates.
Provides functions that mutate the genome (== structure) of the polymorph as well as functionality to mate the
polymorph with other instances of the Polymorph class (for gene crossing, inheritance and etc.)

Also provides functions that evaluate different properties (dipole moment, total energy etc.) of the polymorph (based
on QM-calculations)

Units
-----

Energy : Hartree
Dipole moment : Debye

"""

import itertools
import os
import chemcoord as cc
from collections import Iterable, Collection
import numpy as np

import ase
from ase import data
from ase import visualize
import imolecule
#from ase.visualize import view

from Utilities import checkAtomDistances
from Mutators import FullRangeMutator

import pyscf

# keys for property dictionaries
TOTAL_ENERGY = "total_energy"
DIPOLE_MOMENT = "dipole_moment"
IONIZATION_ENERGY = "ionization_energy"

GENE_TYPES = ('bond', 'angle', 'dihedral')

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

        self.zmat_base = self.base_structure.get_zmat()
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
                                 self.default_crossover_rate)
        else:
            print(f"Warning: Unable to generate random Polymorph. Reached maximum number of restarts ({n_max_restarts})")
            return None




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
        self.n_atoms = len(self.zmat.index)
        self._mutable_bonds = mutable_bonds
        self._mutable_angles = mutable_angles
        self._mutable_dihedrals = mutable_dihedrals
        self.charge = 0 # Electronic charge
        self.multiplicity = 1 # Spin multiplicity

        if mutable_bonds is None:
            self._mutable_bonds = self.zmat.index[1:]

        if mutable_angles is None:
            self._mutable_angles = self.zmat.index[2:]

        if mutable_dihedrals is None:
            self._mutable_dihedrals = self.zmat.index[3:]
            
    def __str__(self):
        text = f"Polymorph '{self.name}', ID: {self.id}"
        return text

    # Dynamic properties---------------------------------------------------------------------------------------------- #
    @property
    def genome(self):
        return {'bondlengths' : self.zmat.loc[self._mutable_bonds,'bond'],
                'angles' : self.zmat.loc[self._mutable_angles, 'angle'],
                'dihedrals' : self.zmat.loc[self._mutable_dihedrals,'dihedral']}

    @property
    def bondpairs(self):
        return zip(self.zmat.index[1:], self.zmat.loc[1:,'b'])
    
    @property
    def calculation_needed(self):
        return self.properties == {}

    @property
    def structure(self):
        return self.zmat.get_cartesian()
    
    @property
    def gzmat_string(self):
        gzmat_text = "# gzmat created from Polymorph\n"
        gzmat_text += "\n" + f"Name: {self.name}, ID:{self.id}, Gen:{self.generation_number}\n" + "\n"
        gzmat_text += f"{self.charge:d} {self.multiplicity:d}\n"
        gzmat_text += self.zmat_string
        return gzmat_text

    @property
    def zmat_string(self):
        return self.zmat.to_zmat(upper_triangle=False)

    def saveStructure(self, filename):
        self.structure.to_xyz(filename)


    # Mating and Crossover ------------------------------------------------------------------------------------------- #
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


    def crossoverGenesWith(self, partner, validate_updates=False):
        """ Attempts gene crossovers with other polymorph 'partner' based on crossover probability. Changes
        are directly written to the zmatrices of both involved polymorphs """

        genomes_altered = False

        own_zmatrix = self.zmat.copy()
        partner_zmatrix = partner.zmat.copy()

        for bond_index in self._mutable_bonds:
            if np.random.rand() < self.crossover_probability:
                genomes_altered = True
                own_bondlength = self.zmat.loc[bond_index, 'bond']
                own_zmatrix.safe_loc[bond_index, 'bond'] = partner.zmat.loc[bond_index, 'bond']
                partner_zmatrix.safe_loc[bond_index, 'bond'] = own_bondlength

        for angle_index in self._mutable_angles:
            if np.random.rand() < self.crossover_probability:
                genomes_altered = True
                own_angle = self.zmat.loc[angle_index, 'angle']
                own_zmatrix.safe_loc[angle_index, 'angle'] = partner.zmat.loc[angle_index, 'angle']
                partner_zmatrix.safe_loc[angle_index, 'angle'] = own_angle

        for dihedral_index in self._mutable_dihedrals:
            if np.random.rand() < self.crossover_probability:
                genomes_altered = True
                own_dihedral = self.zmat.loc[dihedral_index, 'dihedral']
                own_zmatrix.safe_loc[dihedral_index, 'dihedral'] = partner.zmat.loc[dihedral_index, 'dihedral']
                partner_zmatrix.safe_loc[dihedral_index, 'dihedral'] = own_dihedral
                
        if genomes_altered:
            self.applyMutation(own_zmatrix, validate_updates)
            partner.applyMutation(partner_zmatrix, validate_updates)
          
    # Mutations ------------------------------------------------------------------------------------------------------ #

    def applyMutation(self, new_zmatrix, validate_first=True):
        if validate_first:
            update_is_valid = checkAtomDistances(new_zmatrix)
        else:
            update_is_valid = True
            
        if update_is_valid:
            self.zmat = new_zmatrix
            self.resetProperties()


    def mutable_genes(self, gene_type):
        if gene_type == 'bond':
            return self._mutable_bonds
        elif gene_type == 'angle':
            return self._mutable_angles
        elif gene_type == 'dihedral':
            return self._mutable_dihedrals
        else:
            raise ValueError("Unknown gene type. Valid options are: " + str(GENE_TYPES))


    def mutateBonds(self):
        """
        Attempts to mutate bond length of each mutable bond. If one or more bonds are altered, the calculated properties
        of the polymorph are reset.
        """
        new_zmat, bondlengths_altered = self.bond_mutator.mutate(self.zmat, self._mutable_bonds)
        if bondlengths_altered:
            self.applyMutation(new_zmat)
            
    def mutateAngles(self):
        new_zmat, angles_altered = self.angle_mutator.mutate(self.zmat, self._mutable_angles)
        if angles_altered:
            self.applyMutation(new_zmat)
    
    def mutateDihedrals(self):
        new_zmat, dihedrals_altered = self.dihedral_mutator.mutate(self.zmat, self._mutable_dihedrals)
        if dihedrals_altered:
            self.applyMutation(new_zmat)
            
    def mutateGenome(self):
        """ Attempts to mutate all mutable bond lengths, angles and dihedrals of the polymorph (in that order) """
        self.mutateBonds()
        self.mutateAngles()
        self.mutateDihedrals()

    def resetProperties(self):
        self.properties = dict()

    # SCF Calculations ----------------------------------------------------------------------------------------------- #
    def setupGeometryForCalculation(self, basis='sto-3g'):
        zmat_str = self.zmat_string
        mol = pyscf.gto.Mole()
        mol.atom = zmat_str
        mol.basis = 'sto-3g'
        mol.charge = 0 # Neutral molecule
        mol.spin = None # Guess multiplicity
        mol.unit = 'Angstrom'
        mol.verbose = 5 # set output level to 'info'
        mol.max_memory = 1000
        mol.build()
        return mol
    
    def runHartreeFock(self):
        mol = self.setupGeometryForCalculation()
        print("Starting restricted Hartree-Fock calculation ...")
        calc = pyscf.scf.RHF(mol)
       
        if calc.converged:
            energy = calc.e_tot
            print(f"E(RHF) = {energy:g} Hartree")
            dipole_moment = calc.dip_moment()
        else:
            energy = np.nan
            dipole_moment = np.nan * np.ones(3)
            print(f"Polymorph {self.id}: HF calculation did not converge!")
        
        self.properties[TOTAL_ENERGY] = energy
        self.properties[DIPOLE_MOMENT] = dipole_moment

#    def runSemiempiricalQM(self):
#        mol = self.setupGeometryForCalculation()

    # Visualization -------------------------------------------------------------------------------------------------
    
    def visualize(self):
        atoms = self.structure.get_ase_atoms()
        ase.visualize.view(atoms)
    
    


if __name__ == "__main__":
  # Test polymorph class
  pass
#  for k in range(10):
#    pm = Polymorph(f"Polymorph {k}")
#    print(pm)