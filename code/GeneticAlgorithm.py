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


def renderPolymorphs(polymorphs, shader='basic'):
    """ Displays geometries of all polymorphs in the current generation """
    renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                              shader=shader, display_html=False, resizeable=False) \
               for p in polymorphs)
    columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
    display(HTML('<div class="row">{}</div>'.format("".join(columns))))

class GeneticAlgorithm:
    def __init__(self, factory: PolymorphFactory, generation_size=10, fitness_property=Polymorph.TOTAL_ENERGY):
        
        self.factory = factory
        self.generation_number = 0
        self.polymorphs = dict() # Current generation of polymorphs, polymorph id's are keys
        self.properties = pd.DataFrame(columns=Polymorph.DATA_FIELDS, dtype=float)
        
        if fitness_property not in Polymorph.DATA_FIELDS:
            raise ValueError("Unknown fitness property. Valid options: " + \
                             "".join([f"'{p}'" for p in Polymorph.DATA_FIELDS]))
        
        self.fitness_property = fitness_property
        self.generation_size = generation_size
        
    def saveState(self, folder):
        pass
    
    
    @classmethod
    def loadState(cls, folder):
        pass


    def removeLeastFittest(self):
        """"Removes least fittest polymorphs from the current generation"""
        self.properties.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        #showGenerationState(polymorphs)
        polymorphs_to_drop = self.properties.index[self.generation_size:]

        for polymorph_id in polymorphs_to_drop:
            self.polymorphs.pop(polymorph_id)
            
        self.properties = self.properties.drop(polymorphs_to_drop, axis=0)
        

    def selectFromDistribution(self, type='fermi'):
        """ Selects polymorphs based on a probability distribution """
        pass
  
    
    def renderCurrentGeneration(self, shader='basic'):
        """ Displays geometries of all polymorphs in the current generation """
        renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                                  shader=shader, display_html=False, resizeable=False) \
                   for p in self.polymorphs)
        columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
        display(HTML('<div class="row">{}</div>'.format("".join(columns))))
        
    def viewCurrentGeneration(self):
        for p in self.polymorphs.values():
            p.visualize()
    
    
    def collectGenerationProperties(self):
        for p in self.polymorphs.values():
            self.properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
    
    
    def listGenerationProperties(self):
        if not np.any(self.properties[self.fitness_property] is None or self.properties[self.fitness_property] == np.nan):
            return self.properties.style.background_gradient(cmap='RdYlGn_r', subset=[self.fitness_property])
        else:
            return self.properties
    
    
    def evaluateGeneration(self):
        for p in self.polymorphs.values():
            if p.needs_evaluation:
                print(f"Polymorph {p.id} was mutated, needs evaluation")
                p.runHartreeFock()
                # Update Properties
                self.properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
            else:
                print(f"Polymorph {p.id} unchanged")

    def evaluateIonizationEnergies(self):
        for p in self.polymorphs.values():
            if p.properties[Polymorph.IONIZATION_ENERGY] in [np.nan, None]:
                p.calculateIonizationEnergy()
                
        self.collectGenerationProperties()

    def evaluateElectronAffinities(self):
        for p in self.polymorphs.values():
            if p.properties[Polymorph.ELECTRON_AFFINITY] in [np.nan, None]:
                p.calculateElectronAffinity()
    
        self.collectGenerationProperties()

    #### Generate polymorphs
    
    # TODO: Make sure that enough polymorphs are generated
    def fillGeneration(self):
        n_existing = len(self.polymorphs)
        new_polymorphs_dict = dict()
        for k in range(self.generation_size - n_existing):
            p = self.factory.generateRandomPolymorph()
            if p is not None:
                self.polymorphs[p.id] = p
                new_polymorphs_dict[p.id] = p.properties
                
        new_properties_df = pd.DataFrame.from_dict(new_polymorphs_dict, orient='index', dtype=float)
        self.properties = self.properties.append(new_properties_df)


    #### Mate polymorphs
    def generateOffsprings(self, verbose=False):
        n_pairs = int(np.floor(len(self.polymorphs) / 2))
        pair_indices = np.random.permutation(list(self.polymorphs.keys()))
        pair_indices = pair_indices[:2 * n_pairs]
        pair_indices = np.reshape(pair_indices, (n_pairs, 2))
       
        children_properties_dict = dict()
        for ind_a, ind_b in pair_indices:
            partner_a = self.polymorphs[ind_a]
            partner_b = self.polymorphs[ind_b]
            child = partner_a.mateWith(partner_b, verbose=verbose)
            self.polymorphs[child.id] = child
            children_properties_dict[child.id] = child.properties
    
        children_properties = pd.DataFrame.from_dict(children_properties_dict, orient='index', dtype=float)
        self.properties = self.properties.append(children_properties)
        
    def attemptCrossovers(self, valid_crossovers_only=False, verbose=False):
        n_pairs = int(np.floor(len(self.polymorphs) / 2))
        pair_indices = np.random.permutation(list(self.polymorphs.keys()))
        pair_indices = pair_indices[:2 * n_pairs]
        pair_indices = np.reshape(pair_indices, (n_pairs, 2))
        
        for ind_a, ind_b in pair_indices:
            partner_a = self.polymorphs[ind_a]
            partner_b = self.polymorphs[ind_b]
            partner_a.crossoverGenesWith(partner_b, valid_crossovers_only, verbose=verbose)
            
        self.collectGenerationProperties()
        
    def mutateAll(self, verbose=False):
        for p in self.polymorphs.values():
            p.mutateGenome()
            
        self.collectGenerationProperties()
        
  
if __name__ is "__main__":
    import os
    from os.path import join
    import matplotlib.pyplot as plt
    molecules_dir = os.path.abspath(join(os.path.dirname(__file__),"../molecules"))
    testing_dir = os.path.abspath(join(os.path.dirname(__file__),"../testing"))
    structure_filepath = join(molecules_dir, "CF3-CH3.xyz")
    
    os.chdir(testing_dir)
    mutation_rate = 0.05
    crossover_rate = 0.0
    factory = PolymorphFactory(structure_filepath, mutation_rate, crossover_rate)
    factory.freezeBonds('all')
    factory.setupDefaultMutators()
    ga = GeneticAlgorithm(factory, generation_size=6)
    ga.fillGeneration()

    energies_timeline = list()
    polymorphs_timeline = list()
    
    ga.evaluateGeneration()
    
    ga.properties.sort_values(ga.fitness_property, axis=0, inplace=True, ascending=True)
    energies_timeline.append(ga.properties.total_energy)
    
    for k in range(10):
        ga.mutateAll()
        ga.attemptCrossovers(verbose=True)
        ga.generateOffsprings(verbose=True)
        ga.evaluateGeneration()
        ga.removeLeastFittest()
        ga.properties.sort_values(ga.fitness_property, axis=0, inplace=True, ascending=True)
        energies_timeline.append(ga.properties.total_energy.copy())
        polymorphs_timeline.append(ga.polymorphs.copy())
    
    energies_array = np.array(energies_timeline, dtype=float)
    
    plt.imshow(energies_array.transpose(), cmap="RdYlGn_r")
    plt.colorbar()
    plt.show()