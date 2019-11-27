import ase
from ase import data
import numpy as np

from chemcoord import Zmat

def checkAtomDistances(zmatrix: Zmat, threshold_factor=0.8):
    structure = zmatrix.get_cartesian()
    n_atoms = len(zmatrix)
    
    for ind_a in range(n_atoms):
        element_a = structure.loc[[ind_a], 'atom'].to_string(index=False).strip()
        position_a = structure.loc[[ind_a], ('x', 'y', 'z')].to_numpy().flatten()
        radius_a = ase.data.covalent_radii[ase.data.atomic_numbers[element_a]] * threshold_factor
        for ind_b in range(ind_a + 1, n_atoms):
            element_b = structure.loc[[ind_b], 'atom'].to_string(index=False).strip()
            position_b = structure.loc[[ind_b], ('x', 'y', 'z')].to_numpy().flatten()
            radius_b = ase.data.covalent_radii[ase.data.atomic_numbers[element_b]] * threshold_factor
            
            distance = np.linalg.norm(position_b - position_a)
            if distance < radius_a + radius_b:
                return False
    
    else:  # All distances are valid
        return True