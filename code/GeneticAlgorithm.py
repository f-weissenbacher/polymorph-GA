import imolecule
from imolecule import format_converter
from Utilities import checkAtomDistances, fermiDistribution
import Polymorph
from Polymorph import Polymorph
from PolymorphFactory import PolymorphFactory

from Mutators import FullRangeMutator, IncrementalMutator, PlaceboMutator

import pandas as pd
import numpy as np
import time
import pybel
import shutil
import os
import pickle

import configparser

pybel.ipython_3d = True

from IPython.display import display, HTML


def makeSureDirectoryExists(dirpath, overwrite=False):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    elif overwrite:
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)

def renderPolymorphs(polymorphs, shader='basic'):
    """ Displays geometries of all polymorphs in the current generation """
    renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                              shader=shader, display_html=False, resizeable=False) \
               for p in polymorphs)
    columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
    display(HTML('<div class="row">{}</div>'.format("".join(columns))))

class GeneticAlgorithm:
    def __init__(self, factory: PolymorphFactory, generation_size=10, fitness_property=Polymorph.TOTAL_ENERGY,
                 work_dir=None):
        
        self.factory = factory
        self.current_generation_number = 0
        self.polymorphs = dict() # Current generation of polymorphs, polymorph id's are keys
        self.properties = pd.DataFrame(columns=Polymorph.DATA_FIELDS, dtype=float)
        
        if fitness_property not in Polymorph.DATA_FIELDS:
            raise ValueError("Unknown fitness property. Valid options: " + \
                             "".join([f"'{p}'" for p in Polymorph.DATA_FIELDS]))
        
        self.fitness_property = fitness_property
        self.generation_size = generation_size
        
        self.generation_timeline = []
        
        if work_dir is None:
            work_dir = os.getcwd()
            
        self.work_dir = work_dir
        self.pyscf_calc_dir = os.path.join(work_dir, "pyscf-calculations")
        makeSureDirectoryExists(self.pyscf_calc_dir)
        
        
    def saveState(self, folder):
        makeSureDirectoryExists(folder)
        
        factory_filepath = os.path.join(folder, "polymorph_factory.pkl")
        timeline_filepath = os.path.join(folder, "generation_timeline.pkl")
        config_filepath = os.path.join(folder, "settings.config")
        
        # Config file:
        config = configparser.ConfigParser()
        config['Algorithm Settings'] = {'fitness_property' : self.fitness_property,
                                        'generation_size' : self.generation_size,
                                        'current_generation_number' : self.current_generation_number }
        
        with open(config_filepath, 'w') as config_file:
            config.write(config_file)
        
        # Polymorph factory
        with open(factory_filepath, 'wb') as factory_file:
            pickle.dump(self.factory, factory_file)
            
        # Generation timeline (Polymorphs of each generation
        with open(timeline_filepath, 'wb') as timeline_file:
            pickle.dump(self.generation_timeline, timeline_file)

    
    @classmethod
    def loadState(cls, folder):
        factory_filepath = os.path.join(folder, "polymorph_factory.pkl")
        timeline_filepath = os.path.join(folder, "generation_timeline.pkl")
        config_filepath = os.path.join(folder, "settings.config")

        config = configparser.ConfigParser()
        config.read(config_filepath)
    
        generation_size = config['Algorithm Settings']['generation_size']
        generation_number = config['Algorithm Settings']['current_generation_number']
        fitness_property = config['Algorithm Settings']['fitness_property']

    
        # Polymorph factory
        with open(factory_filepath, 'rb') as factory_file:
            factory = pickle.load(factory_file)

        # Generation timeline (Polymorphs of each generation
        with open(timeline_filepath, 'rb') as timeline_file:
            generation_timeline = pickle.load(timeline_file)
            
        ga = cls.__init__(factory, generation_size, fitness_property, work_dir=folder)
        
        ga.current_generation_number = generation_number
        ga.generation_timeline = generation_timeline
        ga.polymorphs = ga.generation_timeline[-1].copy()
        
        return ga
    
    
    def removePolymorphs(self, polymorphs_to_drop):
        """ Removes specified polymorphs from the current generation. Properties DataFrame is updated accordingly
        :param polymorphs_to_drop: list or array of IDs of the polymorphs that shall be removed
        """
        for polymorph_id in polymorphs_to_drop:
            self.polymorphs.pop(polymorph_id)
    
        self.properties = self.properties.drop(polymorphs_to_drop, axis=0)


    def discardLeastFittest(self):
        """" Shrinks current generation back to the default generation size by removing the least fittest polymorphs """
        self.properties.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        #showGenerationState(polymorphs)
        polymorphs_to_drop = self.properties.index[self.generation_size:]
        self.removePolymorphs(polymorphs_to_drop)
        
    
    def discardRandomPolymorphs(self):
        """" Shrinks current generation back to the default generation size by randomly removing polymorphs """
        n_to_drop = len(self.polymorphs)-self.generation_size
        
        if n_to_drop > 0:
            polymorphs_to_drop = np.random.choice(self.properties.index, (n_to_drop,), replace=False)
            self.removePolymorphs(polymorphs_to_drop)

    def discardByFermiDistribution(self, sigma=None):
        """" Shrinks current generation back to the default generation size by removing polymorphs based on a
        probability distribution
        """

        n_to_drop = len(self.polymorphs)-self.generation_size
        
        if n_to_drop <= 0:
            print("Current generation size is less or equal then the default size, " + \
                  "therefore no need to discard polymopphs")
            return
        
        # Sort by fitness_property
        self.properties.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        fitness_values = self.properties[self.fitness_property]
        min_value = np.nanmin(self.properties[self.fitness_property])
        max_value = np.nanmax(self.properties[self.fitness_property])
        cutoff_value = 0.5 * (fitness_values.iloc[-(n_to_drop+1)] + fitness_values.iloc[-n_to_drop])
        
        if sigma is None:
            sigma = 0.05 * (max_value - min_value)

        polymorphs_to_drop = []
        polymorphs_to_keep = list(self.polymorphs.keys())
        
        while len(polymorphs_to_drop) < n_to_drop:
            for pm_id in polymorphs_to_keep:
                occupation_chance = fermiDistribution(fitness_values.loc[pm_id], cutoff_value, sigma)
                if np.random.rand() > occupation_chance:
                    # Drop this polymorph
                    polymorphs_to_drop.append(pm_id)
                    polymorphs_to_keep.remove(pm_id)
                if len(polymorphs_to_drop) == n_to_drop:
                    break
                    
        self.removePolymorphs(polymorphs_to_drop)
            
    
    def renderCurrentGeneration(self, shader='basic'):
        """ Displays geometries of all polymorphs in the current generation """
        renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                                  shader=shader, display_html=False, resizeable=False) \
                   for p in self.polymorphs.values())
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
                p.runHartreeFock(run_dir=self.pyscf_calc_dir)
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
            p.mutateGenome(verbose=verbose)
            
        self.collectGenerationProperties()
        
    def doGenerationStep(self, discard_mode='least-fittest', verbose=True):
        self.mutateAll(verbose=verbose)
        self.attemptCrossovers(verbose=verbose)
        self.generateOffsprings(verbose=verbose)
        self.evaluateGeneration()
        
        if discard_mode == 'least-fittest':
            self.discardLeastFittest()
        elif discard_mode == 'random':
            self.discardRandomPolymorphs()
        elif discard_mode == 'fermi':
            self.discardByFermiDistribution()
        else:
            ValueError("Unknown discard mode. Valid options are: 'least-fittest', 'random' and 'fermi'")
            
        self.generation_timeline.append(self.polymorphs.copy())
        self.current_generation_number += 1
        
  
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
    ga = GeneticAlgorithm(factory, generation_size=10)
    ga.fillGeneration()

    energies_timeline = list()
    polymorphs_timeline = list()
    
    ga.evaluateGeneration()
    
    ga.properties.sort_values(ga.fitness_property, axis=0, inplace=True, ascending=True)
    energies_timeline.append(ga.properties.total_energy)
    
    for k in range(10):
        print(f"Iteration {k}")
        ga.mutateAll(verbose=True)
        ga.attemptCrossovers(verbose=False)
        ga.generateOffsprings(verbose=True)
        ga.evaluateGeneration()
        #ga.discardByFermiDistribution(5.0)
        ga.discardLeastFittest()
        ga.properties.sort_values(ga.fitness_property, axis=0, inplace=True, ascending=True)
        energies_timeline.append(ga.properties.total_energy.copy())
        polymorphs_timeline.append(ga.polymorphs.copy())
        print("=" * 60)
    
    energies_array = np.array(energies_timeline, dtype=float)
    
    plt.imshow(energies_array.transpose(), cmap="RdYlGn_r")
    plt.colorbar()
    plt.show()