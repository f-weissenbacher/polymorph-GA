import imolecule
from imolecule import format_converter
from Utilities import checkAtomDistances
import Polymorph
from Polymorph import Polymorph
from PolymorphFactory import PolymorphFactory

from Mutators import FullRangeMutator, IncrementalMutator, PlaceboMutator

import pandas as pd
import numpy as np
import time
import pybel

pybel.ipython_3d = True

from IPython.display import display, HTML

class GeneticAlgorithm:
    def __init__(self):
        self.generation_number = 0
        self.polymorphs = dict() # Current generation of polymorphs, polymorph id's are keys
        self.properties = pd.DataFrame
        
        self.fitness_property = Polymorph.TOTAL_ENERGY
        
        
    def saveState(self, folder):
        pass
    
    
    @classmethod
    def loadState(cls, folder):
        pass
        
        
    


    def removeLeastFittest(self):
        """"Removes least fittest polymorphs from the current generation"""
        polymorphs.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        showGenerationState(polymorphs)

        ids_to_keep = polymorphs.index[:generation_size]
        polymorphs = polymorphs.loc[ids_to_keep, :]
        

    def selectFromDistribution(self, type='fermi'):
        """ Selects polymorphs based on a probability distribution """
        pass
  
    
    def renderCurrentGeneration(self, shader='basic'):
        
        renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                                  shader=shader, display_html=False, resizeable=False) \
                   for p in polymorphs_df.pm)
        columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
        display(HTML('<div class="row">{}</div>'.format("".join(columns))))
    
    
    def collectGenerationProperties(polymorphs_df):
        for p in polymorphs_df.pm:
            properties_vec = [p.total_energy,
                              p.dipole_moment[0],
                              p.dipole_moment[1],
                              p.dipole_moment[2]]
            polymorphs_df.loc[p.id, 1:] = properties_vec
    
    
    def showGenerationState(polymorphs_df):
        if not np.any(np.isnan(polymorphs_df.total_energy)):
            return polymorphs_df.style.background_gradient(cmap='RdYlGn_r', subset=['total_energy'])
        else:
            return polymorphs_df
    
    
    def evaluateGeneration(polymorphs_df):
        for p in polymorphs_df.pm:
            if p.needsEvaluation():
                print(f"Polymorph {p.id} was mutated, needs evaluation")
                p.runHartreeFock()
            else:
                print(f"Polymorph {p.id} unchanged")

    ### Mate polymorphs
    def generateOffsprings(self):
        n_pairs = int(np.floor(len(polymorphs) / 2))
        pair_indices = np.random.permutation(polymorphs.index)
        pair_indices = pair_indices[:2 * n_pairs]
        pair_indices = np.reshape(pair_indices, (n_pairs, 2))
       
        child_polymorphs_dict = dict()
        for ind_a, ind_b in pair_indices:
            partner_a = polymorphs.pm[ind_a]
            partner_b = polymorphs.pm[ind_b]
            child = partner_a.mateWith(partner_b)
            child_polymorphs_dict[child.id] = child.asDataDict()
    
    
        child_polymorphs = pd.DataFrame.from_dict(child_polymorphs_dict, orient='index')
    
        polymorphs = polymorphs.append(child_polymorphs)
        
        