import numpy as np
import ase
from ase import data

from Polymorph import Polymorph, checkAtomDistances

class Mutator:
    """ Base class of molecular structure mutators"""
    GENE_TYPES = Polymorph.GENE_TYPES

    def __init__(self, target_gene: str, gene_value_range, gene_is_periodic=False, mutation_rate=0.001):
        if target_gene not in self.GENE_TYPES:
            raise ValueError(f"Invalid target gene '{target_gene}'. Valid gene types are 'bond', 'angle' and 'dihedral'")

        self.target_gene = target_gene
        self.mutation_probability = mutation_rate
        if len(gene_value_range) != 2 or np.diff(gene_value_range)[0] < 0:
            raise ValueError("gene_value_range must be of length 2 and monotonically increasing")
        self.gene_value_range = gene_value_range
        self.gene_value_span = np.diff(gene_value_range)[0]
        self.gene_is_periodic = gene_is_periodic

    def validateValue(self, value):
        if self.gene_is_periodic:
            valid_value = (value - self.gene_value_range[0]) % self.gene_value_span
        else:
            if value < self.gene_value_range[0]:
                valid_value = self.gene_value_range[0]
            else:
                valid_value = self.gene_value_range[1]

        return valid_value
    
    def mutable_genes(self, polymorph):
    
   
    # Abstract function definition, needs to implemented in child classes
    def mutate(self, polymorph):
        raise NotImplementedError("Mutator.mutate: This function is virtual and needs to be implemented in child classes")


class IncrementalMutator(Mutator):
    """
    Mutator that adds a (uniformly chosen) value from the interval defined by **mutation_range** to the current value
    of the target gene
    """
    def __init__(self, target_gene: str, gene_value_range, mutation_range, gene_is_periodic=False, mutation_rate=1e-3):

        super().__init__(target_gene, gene_value_range, gene_is_periodic, mutation_rate)

        self.mutation_range = mutation_range
        self.mutation_span = np.diff(mutation_range)[0]

    def mutate(self, polymorph):
        """
        :param Polymorph polymorph: The polymorph that shall be mutated
        :param mutable_genes_indices: Indices of the polymorph's genes (bonds, angles or dihedrals) that can be altered
        :return bool genes_altered: 'True' if one ore more of the targeted genes have been mutated, 'False' if not
        """
        genes_altered = False
        new_zmatrix = polymorph.zmat.copy()
        for gene_index in polymorph.:
            if np.random.rand() < self.mutation_probability:
                new_value = zmatrix.loc[gene_index, self.target_gene] + np.random.rand() * self.mutation_span
                new_value = self.validateValue(new_value)
                new_zmatrix.safe_loc[gene_index, self.target_gene] = new_value
                genes_altered = True
                
        if genes_altered:
            update_valid = checkAtomDistances(new_zmatrix)
            if update_valid:
                zmatrix = new_zmatrix
            
                
                
                

        


class MultiplicativeMutator(Mutator):
    """
    Mutator that mutates gene by multiplying its value with a (uniformly chosen) factor from the interval defined by
    **mutation_scaling_range**. Rescaled value is mapped to **gene_value_range** before applying change to the zmatrix.
    """
    def __init__(self, target_gene: str, gene_value_range, mutation_scaling_range, gene_is_periodic=False,
                 mutation_rate=1e-3):

        super().__init__(target_gene, gene_value_range, gene_is_periodic, mutation_rate)

        self.scaling_range = mutation_scaling_range
        self.scaling_span = np.diff(mutation_scaling_range)

    def mutate(self, zmatrix, mutable_genes_indices):
        genes_altered = False
        for gene_index in mutable_genes_indices:
            if np.random.rand() < self.mutation_probability:
                factor =  self.scaling_range[0] + np.random.rand() * self.scaling_span
                new_value = self.validateValue(zmatrix.loc[gene_index, self.target_gene] * factor)
                zmatrix.safe_loc[gene_index, self.target_gene] = new_value
                genes_altered = True

        return genes_altered

class FullRangeMutator(Mutator):
    """
    Mutator that mutates the target gene (bond lengths, angles or dihedrals) by assigning a new value that is chosen
    uniformly in the interval defined by **gene_value_range**
    """

    def __init__(self, target_gene: str, gene_value_range, mutation_rate=1e-3):

        super().__init__(target_gene, gene_value_range, False, mutation_rate)


    def mutate(self, zmatrix, mutable_genes_indices):
        old_zmatrix = zmatrix.copy()
        genes_altered = False
        for gene_index in mutable_genes_indices:
            if np.random.rand() < self.mutation_probability:
                new_value = self.gene_value_range[0] + np.random.rand() * self.gene_value_span
                zmatrix.safe_loc[gene_index, self.target_gene] = new_value
                genes_altered = True

        return genes_altered

if __name__ == "__main__":
    m = Mutator('bond', (1.5,2.0), False)
    mm = MultiplicativeMutator('bond', (0.2, 2.0), (0.75,1.25))
    frm = FullRangeMutator('angle', (0,180))
    im = IncrementalMutator('angle', (0,180), (-10,10), True)