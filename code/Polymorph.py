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
from collections import Iterable, Collection, defaultdict
import numpy as np
import pandas as pd
import ase
from ase import data
from ase import visualize
import imolecule
#from ase.visualize import view

from Utilities import checkAtomDistances
from Mutators import FullRangeMutator

import pyscf

HARTREE_IN_EV = 27.211386245988

class Polymorph:
    # Function to generate new, unique IDs
    _generate_id = itertools.count().__next__

    # keys for property dictionaries
    TOTAL_ENERGY = "total_energy"
    DIPOLE_MOMENT = "dipole_moment"
    DIPOLE_VEC_FIELDS = ('mu_x', 'mu_y', 'mu_z')
    IONIZATION_ENERGY = "ionization_energy"
    ELECTRON_AFFINITY = "electron_affinity"
    DATA_FIELDS = (TOTAL_ENERGY, DIPOLE_MOMENT) + DIPOLE_VEC_FIELDS + (IONIZATION_ENERGY, ELECTRON_AFFINITY)
    
    GENE_TYPES = ('bond', 'angle', 'dihedral')
    
    @classmethod
    def resetIdCounter(cls):
        cls._generate_id = itertools.count().__next__

    def __init__(self, zmatrix, bond_mutator, angle_mutator, dihedral_mutator,
                 mutable_bonds=None, mutable_angles=None, mutable_dihedrals=None,
                 crossover_rate=1e-3, name="", generation_number=-1):

        self.name = name
        self.id = Polymorph._generate_id()
        self.generation_number = generation_number
        self.bond_mutator = bond_mutator
        self.angle_mutator = angle_mutator
        self.dihedral_mutator = dihedral_mutator
        self.crossover_probability = crossover_rate
        self.zmat = zmatrix
        self.n_atoms = len(self.zmat.index)
        self._mutable_bonds = mutable_bonds
        self._mutable_angles = mutable_angles
        self._mutable_dihedrals = mutable_dihedrals
        
        self.data_fields = Polymorph.DATA_FIELDS
        self.properties = defaultdict(lambda:np.nan).fromkeys(Polymorph.DATA_FIELDS)
        self.needs_evaluation = True
        
        # Default SCF settings:
        self.scf_verbosity = 0 # 0: no output, 5: info level (good for debugging etc)
        self.scf_basis = 'sto-3g'
        self.charge = 0 # Electronic charge
        self.spin = 0 # Spin multiplicity
        self.last_calculation = None

        if mutable_bonds is None:
            self._mutable_bonds = self.zmat.index[1:]

        if mutable_angles is None:
            self._mutable_angles = self.zmat.index[2:]

        if mutable_dihedrals is None:
            self._mutable_dihedrals = self.zmat.index[3:]
            
    def __str__(self):
        text = f"{self.name}, ID: {self.id}"
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
    def structure(self):
        return self.zmat.get_cartesian()
    
    @property
    def total_energy(self):
        return self.properties[Polymorph.TOTAL_ENERGY]
        
    @property
    def dipole_moment_vec(self):
        dipole_vec = np.nan * np.ones(3)
        for k, field in enumerate(Polymorph.DIPOLE_VEC_FIELDS):
            dipole_vec[k] = self.properties[field]
        return dipole_vec
        
    @property
    def dipole_moment(self):
        return self.properties[Polymorph.DIPOLE_MOMENT]
    
    @property
    def gzmat_string(self):
        gzmat_text = "# gzmat created from Polymorph\n"
        gzmat_text += "\n" + f"Name: {self.name}, ID:{self.id}, Gen:{self.generation_number}\n" + "\n"
        gzmat_text += f"{self.charge:d} {(2*self.spin+1):d}\n"
        gzmat_text += self.zmat_string
        return gzmat_text

    @property
    def zmat_string(self):
        return self.zmat.to_zmat(upper_triangle=False)

    def saveStructure(self, filename):
        self.structure.to_xyz(filename)
        
    def resetProperties(self):
        self.properties = defaultdict(lambda:np.nan).fromkeys(Polymorph.DATA_FIELDS)
        self.needs_evaluation = True
        

    # Mating and Crossover ------------------------------------------------------------------------------------------- #
    def mateWith(self, partner, verbose=False):
        """ Creates an offspring polymorph by mating two polymorphs
        Both involved polymorphs are assumed to share the same type of genome """

        new_zmat = self.zmat.copy()
        
        if verbose:
            print(f"Mating polymorphs {self.id} and {partner.id}")

        for bond_index in self._mutable_bonds:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[bond_index, 'bond'] = partner.zmat.loc[bond_index, 'bond']

        for angle_index in self._mutable_angles:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[angle_index, 'angle'] = partner.zmat.loc[angle_index, 'angle']

        for dihedral_index in self._mutable_dihedrals:
            if np.random.rand() < 0.5:
                new_zmat.safe_loc[dihedral_index, 'dihedral'] = partner.zmat.loc[dihedral_index, 'dihedral']
      
        name = f"Child of {self.id} & {partner.id}"
        generation_number = max(self.generation_number, partner.generation_number) + 1
      
        new_polymorph = Polymorph(new_zmat, self.bond_mutator, self.angle_mutator, self.dihedral_mutator,
                         self._mutable_bonds, self._mutable_angles, self._mutable_dihedrals,
                         self.crossover_probability, name=name, generation_number=generation_number)
        
        print(f"--> Child polymorph: {new_polymorph.id}")
        
        return new_polymorph
        


    def crossoverGenesWith(self, partner, validate_updates=False, verbose=False):
        """ Attempts gene crossovers with other polymorph 'partner' based on crossover probability. Changes
        are directly written to the zmatrices of both involved polymorphs """

        genomes_altered = False
        own_zmatrix = self.zmat.copy()
        partner_zmatrix = partner.zmat.copy()
        
        if verbose:
            print(f"Attempting gene crossover between polymorphs {self.id} and {partner.id}")

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
            
        return update_is_valid


    def mutable_genes(self, gene_type):
        if gene_type == 'bond':
            return self._mutable_bonds
        elif gene_type == 'angle':
            return self._mutable_angles
        elif gene_type == 'dihedral':
            return self._mutable_dihedrals
        else:
            raise ValueError("Unknown gene type. Valid options are: " + str(Polymorph.GENE_TYPES))

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


    # SCF Calculations ----------------------------------------------------------------------------------------------- #
    def setupGeometryForCalculation(self, basis=None, charge=None, spin=None, verbosity=None):
        if basis is None:
            basis = self.scf_basis
        if verbosity is None:
            verbosity = self.scf_verbosity
        if charge is None:
            charge = self.charge
        if spin is None:
            spin = self.spin
            
        mol = pyscf.gto.Mole()
        mol.atom = self.zmat_string
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = 'Angstrom'
        mol.verbose = verbosity
        mol.max_memory = 1000
        mol.build()
        return mol
    
    def runHartreeFock(self, basis=None, charge=None, spin=None, verbosity=None):
        mol = self.setupGeometryForCalculation(basis, charge, spin, verbosity)
        print("Starting restricted Hartree-Fock calculation ...")
        calc = pyscf.scf.RHF(mol)
        calc.kernel() # Needed to collect results of the calculation (I guess)
       
        if calc.converged:
            energy = calc.e_tot * HARTREE_IN_EV
            print(f"E(RHF) = {energy:g} eV")
            dipole_moment = calc.dip_moment()
            
            self.properties[Polymorph.TOTAL_ENERGY] = energy
            self.properties[Polymorph.DIPOLE_MOMENT] = np.linalg.norm(dipole_moment)
            for k in range(3):
                self.properties[Polymorph.DIPOLE_VEC_FIELDS[k]] = dipole_moment[k]
                
            self.last_calculation = calc
            self.needs_evaluation = False
        else:
            print(f"Polymorph {self.id}: Hartree-Fock calculation did not converge!")
        

        
    def calculateElectronAffinity(self, anion_spin=None):
        if anion_spin is None:
            anion_spin = (self.spin + 1) % 2
            
        if self.needs_evaluation:
            self.runHartreeFock()
            
        neutral_energy = self.total_energy
        
        if neutral_energy == np.nan:
            print(f"Polymorph {self.id}: Energy of neutral molecule cannot be determined, " + \
                  "no use in running calculation for anion.")
            self.properties[Polymorph.ELECTRON_AFFINITY] = np.nan
            return
            
        anion = self.setupGeometryForCalculation(charge=self.charge-1, spin=anion_spin)
        print("Starting Hartree-Fock calculation for anion ...")
        calc = pyscf.scf.RHF(anion)
        calc.kernel()  # Needed to collect results of the calculation (I guess)

        if calc.converged:
            anion_energy = calc.e_tot * HARTREE_IN_EV
            print(f"Anion: E(ROHF) = {anion_energy:g} eV")
            electron_affinity = neutral_energy - anion_energy # > 0
            print(f"Electron affinity: {electron_affinity} eV")
            self.properties[Polymorph.ELECTRON_AFFINITY] = electron_affinity
            
        else:
            print(f"Polymorph {self.id}: Hartree-Fock calculation for anion did not converge!")
            self.properties[Polymorph.ELECTRON_AFFINITY] = np.nan
            return

    def calculateIonizationEnergy(self, cation_spin=None):
        if cation_spin is None:
            cation_spin = (self.spin + 1) % 2
    
        if self.needs_evaluation:
            self.runHartreeFock()
    
        neutral_energy = self.total_energy
    
        if neutral_energy == np.nan:
            print(f"Polymorph {self.id}: Energy of neutral molecule cannot be determined, " + \
                  "no use in running calculation for cation.")
            self.properties[Polymorph.IONIZATION_ENERGY] = np.nan
            return None
    
        cation = self.setupGeometryForCalculation(charge=self.charge + 1, spin=cation_spin)
        print("Starting Hartree-Fock calculation for cation ...")
        calc = pyscf.scf.RHF(cation)
        calc.kernel()  # Needed to collect results of the calculation (I guess)
    
        if calc.converged:
            cation_energy = calc.e_tot * HARTREE_IN_EV
            print(f"Cation: E(ROHF) = {cation_energy:g} eV")
            ionization_energy = cation_energy - neutral_energy  # > 0
            print(f"Ionization energy: {ionization_energy} eV")
            self.properties[Polymorph.IONIZATION_ENERGY] = ionization_energy
    
        else:
            print(f"Polymorph {self.id}: Hartree-Fock calculation for cation did not converge!")
            self.properties[Polymorph.IONIZATION_ENERGY] = np.nan
            return None


    # Visualization -------------------------------------------------------------------------------------------------
    
    def visualize(self):
        atoms = self.structure.get_ase_atoms()
        ase.visualize.view(atoms, name=f"ID_{self.id}_")
    
    


if __name__ == "__main__":
  # Test polymorph class
  pass
#  for k in range(10):
#    pm = Polymorph(f"Polymorph {k}")
#    print(pm)