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

from ase import visualize, neighborlist
import imolecule
#from ase.visualize import view

import pybel

from Utilities import checkAtomDistances
from Mutators import Mutator

import pyscf

HARTREE_IN_EV = 27.211386245988

cc.settings['defaults']['viewer'] = 'ase-gui'

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
    DATA_UNITS = {TOTAL_ENERGY:'eV',
                  DIPOLE_MOMENT:'Debye',
                  DIPOLE_VEC_FIELDS[0]: 'Debye', DIPOLE_VEC_FIELDS[1]: 'Debye', DIPOLE_VEC_FIELDS[2]: 'Debye',
                  IONIZATION_ENERGY: 'eV', ELECTRON_AFFINITY: 'eV'}
    
    GENE_TYPES = ('bond', 'angle', 'dihedral')
    
    # Class methods -------------------------------------------------------------------------------------------------- #
    @classmethod
    def resetIdCounter(cls, first_id=0):
        cls._generate_id = itertools.count(first_id).__next__

    # Constructor and other __functions__ ---------------------------------------------------------------------------- #
    def __init__(self, zmatrix, bond_mutator, angle_mutator, dihedral_mutator,
                 mutable_bonds=None, mutable_angles=None, mutable_dihedrals=None,
                 crossover_rate=1e-3, bond_map=None, name="", custom_id=None):

        self.name = name
        if custom_id is None:
            self.id = Polymorph._generate_id()
        else:
            self.id = custom_id

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
        
        if bond_map is None:
            self.bond_map = self._buildBondMap()

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
    def structure(self):
        return self.zmat.get_cartesian().sort_index()
    
    @property
    def real_structure(self):
        """ Geometry in cartesian coordinates without any virtual atoms """
        structure = self.zmat.get_cartesian()
        virtual_atoms = structure['atom'] == 'X'
        return structure[~virtual_atoms].sort_index()
    
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
        gzmat_text += "\n" + f"Name: {self.name}, ID:{self.id}\n" + "\n"
        gzmat_text += f"{self.charge:d} {(2*self.spin+1):d}\n"
        gzmat_text += self.zmat_string
        return gzmat_text

    @property
    def zmat_string(self):
        return self.zmat.to_zmat(upper_triangle=False)
    
    # Getters -------------------------------------------------------------------------------------------------------- #
    def bond_atoms(self, mutable_only=False):
        if mutable_only:
            return self.zmat.loc[self._mutable_bonds, 'b']
        else:
            return self.zmat.iloc[1:].loc[:, 'b']

    def angle_atoms(self, mutable_only=False):
        if mutable_only:
            angles_df = self.zmat.loc[self._mutable_angles]
        else:
            angles_df = self.zmat.iloc[2:]
        # Sort values
        angles_df = angles_df.loc[:, ['b', 'a']].sort_index()
        angles_df = angles_df.rename(columns={'b': 'B', 'a': 'C'}).rename_axis("A", axis="columns")
        return angles_df

    @property
    def dihedral_atoms(self, mutable_only=False):
        if mutable_only:
            dihedrals_df = self.zmat.loc[self._mutable_dihedrals]
        else:
            dihedrals_df = self.zmat.iloc[3:]
        # Sort values and rename columns
        dihedrals_df = dihedrals_df.loc[:, ['b', 'a', 'd']].sort_index()
        dihedrals_df = dihedrals_df.rename(columns={'b': 'B', 'a': 'C', 'd': 'D'}).rename_axis("A", axis="columns")
        return dihedrals_df

    # Private / internal functions ----------------------------------------------------------------------------------- #

    def _buildBondMap(self):
        ase_atoms = self.real_structure.get_ase_atoms()
        cutoffs = ase.neighborlist.natural_cutoffs(ase_atoms)
        neighbor_list = ase.neighborlist.neighbor_list('ij', ase_atoms, cutoffs)
        ase_pairs = np.array(neighbor_list).transpose()
        
        bond_map = defaultdict(set)
        for i, j in ase_pairs:
            bond_map[i].add(j)
            bond_map[j].add(i)

        bond_atoms = self.bond_atoms(mutable_only=False)
        for a,b in zip(bond_atoms.index, bond_atoms):
            bond_map[a].add(b)
            bond_map[b].add(a)
                
        return bond_map

    # Loading/Saving ------------------------------------------------------------------------------------------------- #
    def saveStructure(self, filename):
        self.structure.to_xyz(filename)
        
    #def save(self, ):
    
    def resetProperties(self):
        self.properties = defaultdict(lambda:np.nan).fromkeys(Polymorph.DATA_FIELDS)
        self.needs_evaluation = True
        

    # Mating and Crossover ------------------------------------------------------------------------------------------- #
    
    def calculateGenomeDifference(self, partner, comparison_mode='average'):
        own_genome = self.genome
        partner_genome = partner.genome
        
        bond_mismatch = np.abs((own_genome['bondlengths'] - partner_genome['bondlengths']) / \
                               (own_genome['bondlengths'] + partner_genome['bondlengths']))
        angles_mismatch = np.abs(own_genome['angles'] - partner_genome['angles'])
        dihedrals_mismatch = np.abs(own_genome['dihedrals'] - partner_genome['dihedrals'])
        L360 = dihedrals_mismatch > 180
        dihedrals_mismatch[L360] = 360 - dihedrals_mismatch[L360]
        
        if comparison_mode == 'average':
            return np.mean(bond_mismatch), np.mean(angles_mismatch), np.mean(dihedrals_mismatch)
        elif comparison_mode == 'maximum':
            return np.max(bond_mismatch), np.max(angles_mismatch), np.max(dihedrals_mismatch)
        else:
            return ValueError("Invalid comparison mode. Valid options: 'average', 'maximum'")
    
    def mateWith(self, partner, validate_child=True, verbose=False):
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
      
        if validate_child:
            if not checkAtomDistances(new_zmat):
                if verbose:
                    print("Resulting child was not valid!")
                return None
      
        new_polymorph = Polymorph(new_zmat, self.bond_mutator, self.angle_mutator, self.dihedral_mutator,
                                  self._mutable_bonds, self._mutable_angles, self._mutable_dihedrals,
                                  self.crossover_probability, name=name)
        
        if verbose:
            print(f"--> Child polymorph: {new_polymorph.id}")
        
        return new_polymorph


    def crossoverGenesWith(self, partner, validate_updates=False, verbose=False):
        """ Attempts gene crossovers with other polymorph 'partner' based on crossover probability. Changes
        are directly written to the zmatrices of both involved polymorphs """

        genomes_altered = False
        own_zmatrix = self.zmat.copy()
        partner_zmatrix = partner.zmat.copy()
        
        if verbose:
            print(f"Trying gene crossover between polymorphs {self.id} and {partner.id}")

        for bond_index in self._mutable_bonds:
            if np.random.rand() < self.crossover_probability:
                own_bondlength = self.zmat.loc[bond_index, 'bond']
                partner_bondlength = partner.zmat.loc[bond_index, 'bond']
                if not np.isclose(own_bondlength, partner_bondlength):
                    own_zmatrix.safe_loc[bond_index, 'bond'] = partner_bondlength
                    partner_zmatrix.safe_loc[bond_index, 'bond'] = own_bondlength
                    genomes_altered = True
                
        for angle_index in self._mutable_angles:
            if np.random.rand() < self.crossover_probability:
                own_angle = self.zmat.loc[angle_index, 'angle']
                partner_angle = partner.zmat.loc[angle_index, 'angle']
                if not np.isclose(own_angle, partner_angle):
                    own_zmatrix.safe_loc[angle_index, 'angle'] = partner_angle
                    partner_zmatrix.safe_loc[angle_index, 'angle'] = own_angle
                    genomes_altered = True

        for dihedral_index in self._mutable_dihedrals:
            if np.random.rand() < self.crossover_probability:
                own_dihedral = self.zmat.loc[dihedral_index, 'dihedral']
                partner_dihedral =  partner.zmat.loc[dihedral_index, 'dihedral']
                if not np.isclose(own_dihedral, partner_dihedral):
                    own_zmatrix.safe_loc[dihedral_index, 'dihedral'] = partner_dihedral
                    partner_zmatrix.safe_loc[dihedral_index, 'dihedral'] = own_dihedral
                    genomes_altered = True
                
        if genomes_altered:
            success_self = self.applyMutation(own_zmatrix, validate_updates)
            success_partner = partner.applyMutation(partner_zmatrix, validate_updates)
            if verbose:
                text = f"Transfer {self.id} -> {partner.id}: "
                if success_partner:
                    text += "successful"
                else:
                    text += "not successful"
                text += f"; Transfer {partner.id} -> {self.id}:"
                if success_self:
                    text += "successful"
                else:
                    text += "not successful"
                print(text)
        elif verbose:
            print("No crossover event triggered")
            
          
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


    def _mutate_gene(self, mutator:Mutator, mutable_genes, verbose=False):
        target_gene = mutator.target_gene
        if verbose:
            print(f"Polymorph {self.id}: Attempting to mutate {target_gene}(s) ...")
        new_zmat, genes_altered = mutator.mutate(self.zmat, mutable_genes, verbose=verbose)
        
        if genes_altered:
            mutation_successful = self.applyMutation(new_zmat)
            if verbose:
                if mutation_successful:
                    print(f"Polymorph {self.id}: Mutation of {target_gene}(s) was applied successfully")
                else:
                    print(f"Polymorph {self.id}: Proposed mutation of {target_gene}(s) was invalid")
        elif verbose:
            print(f"Polymorph {self.id}: No mutation of {target_gene}(s) triggered")
            
            
    def mutateBonds(self, verbose=False):
        """
        Attempts to mutate bond length of each mutable bond. If one or more bonds are altered, the calculated properties
        of the polymorph are reset.
        """
        
        self._mutate_gene(self.bond_mutator, self._mutable_bonds, verbose)
            
    def mutateAngles(self, verbose=False):
        """
        Attempts to mutate each mutable angle. If one or more angles are altered, the calculated properties
        of the polymorph are reset.
        """
        self._mutate_gene(self.angle_mutator, self._mutable_angles, verbose)
            
    def mutateDihedrals(self, verbose=False):
        """
        Attempts to mutate each mutable dihedral. If one or more dihedrals are altered, the calculated properties
        of the polymorph are reset.
        """
        self._mutate_gene(self.dihedral_mutator, self._mutable_dihedrals, verbose)
            
    def mutateGenome(self, verbose=False):
        """ Attempts to mutate all mutable bond lengths, angles and dihedrals of the polymorph (in that order) """
        self.mutateBonds(verbose=verbose)
        self.mutateAngles(verbose=verbose)
        self.mutateDihedrals(verbose=verbose)
   

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
        mol.atom = self.real_structure.to_string(index=False, header=False)
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = 'Angstrom'
        mol.verbose = verbosity
        mol.max_memory = 1000
        mol.build()
        return mol
    
    def runHartreeFock(self, basis=None, charge=None, spin=None, verbosity=None, run_dir=None):
        """
        Sets up and runs a (restricted) Hartree-Fock calculation
        :param basis:
        :param charge:
        :param spin:
        :param verbosity:
        :param run_dir: Path to folder to which the chkfile is written to. Default: os.getcwd()
        """
        
        if run_dir is None:
            run_dir = os.getcwd()
        
        mol = self.setupGeometryForCalculation(basis, charge, spin, verbosity)
        print("Starting restricted Hartree-Fock calculation ...")
        calc = pyscf.scf.RHF(mol)
        
        calc.chkfile = os.path.join(run_dir,f"pm-{self.id}_ch{mol.charge}_s{mol.spin}.chk")
        calc.kernel() # Executes the calculation
       
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
        
    def calculateElectronAffinity(self, anion_spin=None, run_dir=None):
        if anion_spin is None:
            anion_spin = (self.spin + 1) % 2
            
        if run_dir is None:
            run_dir = os.getcwd()
            
        if self.needs_evaluation:
            self.runHartreeFock(run_dir=run_dir)
            
        neutral_energy = self.total_energy
        
        if neutral_energy == np.nan:
            print(f"Polymorph {self.id}: Energy of neutral molecule cannot be determined, " + \
                  "no use in running calculation for anion.")
            self.properties[Polymorph.ELECTRON_AFFINITY] = np.nan
            return
            
        anion = self.setupGeometryForCalculation(charge=self.charge-1, spin=anion_spin)
        print("Starting Hartree-Fock calculation for anion ...")
        calc = pyscf.scf.RHF(anion)
        calc.chkfile = os.path.join(run_dir,f"pm-{self.id}_ch{anion.charge}_s{anion.spin}.chk")
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
        
    def calculateIonizationEnergy(self, cation_spin=None, run_dir=None):
        if cation_spin is None:
            cation_spin = (self.spin + 1) % 2
            
        if run_dir is None:
            run_dir = os.getcwd()
    
        if self.needs_evaluation:
            self.runHartreeFock(run_dir=run_dir)
    
        neutral_energy = self.total_energy
    
        if neutral_energy == np.nan:
            print(f"Polymorph {self.id}: Energy of neutral molecule cannot be determined, " + \
                  "no use in running calculation for cation.")
            self.properties[Polymorph.IONIZATION_ENERGY] = np.nan
            return None
    
        cation = self.setupGeometryForCalculation(charge=self.charge + 1, spin=cation_spin)
        print("Starting Hartree-Fock calculation for cation ...")
        calc = pyscf.scf.RHF(cation)
        calc.chkfile = os.path.join(run_dir, f"pm-{self.id}_ch{cation.charge}_s{cation.spin}.chk")
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

    def evaluate(self, fitness_property=TOTAL_ENERGY):
        """ Determines the fitness of the polymorph by running the corresponding SCF calculations """

        if fitness_property == Polymorph.IONIZATION_ENERGY:
            self.calculateIonizationEnergy()
        elif fitness_property == Polymorph.ELECTRON_AFFINITY:
            self.calculateElectronAffinity()
        else:
            self.runHartreeFock()
    
    # Visualization -------------------------------------------------------------------------------------------------
    def visualize(self, viewer='ase'):
        if viewer == 'ase':
            atoms = self.structure.get_ase_atoms()
            ase.visualize.view(atoms, name=f"ID_{self.id}_")
        else:
            pybel.Ipython_3d = True
            ob_mol = pybel.readstring('gzmat', self.gzmat_string)
            return ob_mol
    
    # Misc ----------------------------------------------------------------------------------------------------------
    def selectDihedralsByType(self, type='proper'):
        valid_dihedral_idxs = self.zmat.index[3:]
        proper_dihedrals = []
        
        for a in valid_dihedral_idxs:
            b,c,d = list(self.zmat.loc[a,['b', 'a', 'd']])
            if b in self.bond_map[a] and c in self.bond_map[b] and d in self.bond_map[c]:
               proper_dihedrals.append(a)
               
        if type == 'proper':
            return pd.Int64Index(proper_dihedrals)
        elif type == 'improper':
            return valid_dihedral_idxs.drop(proper_dihedrals)


if __name__ == "__main__":
    import chemcoord as cc
    from Mutators import PlaceboMutator
    structure = cc.Cartesian.read_xyz("../molecules/CF3-CH3.xyz")
    zmat = structure.get_zmat()
    bond_mutator = PlaceboMutator('bond')
    angle_mutator = PlaceboMutator('angle')
    dihedral_mutator = PlaceboMutator('dihedral')
    pm = Polymorph(zmat, bond_mutator, angle_mutator, dihedral_mutator)
    
    #proper_dihedrals = pm.selectDihedralsByType()
    