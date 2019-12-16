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

from matplotlib.ticker import MaxNLocator

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
                 work_dir=None, default_discard_mode='fermi'):
        
        self.factory = factory

        self.current_generation = dict() # Current generation of polymorphs, polymorph id's are keys
        self.current_properties = pd.DataFrame(columns=Polymorph.DATA_FIELDS, dtype=float) # Properties of current generation
        
        if fitness_property not in Polymorph.DATA_FIELDS:
            raise ValueError("Unknown fitness property. Valid options: " + \
                             "".join([f"'{p}'" for p in Polymorph.DATA_FIELDS]))
        
        self.fitness_property = fitness_property
        self.generation_size = generation_size
        self.current_generation_number = 0
        self.population_timeline = []
        self.properties_timeline = []
        self.default_discard_mode = default_discard_mode
        
        if work_dir is None:
            work_dir = os.getcwd()
        else:
            makeSureDirectoryExists(work_dir)
            
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
            pickle.dump(self.population_timeline, timeline_file)

    
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
        ga.population_timeline = generation_timeline
        ga.current_generation = ga.population_timeline[-1].copy()
        
        return ga
    
    
    def removePolymorphs(self, polymorphs_to_drop):
        """ Removes specified polymorphs from the current generation. Properties DataFrame is updated accordingly
        :param polymorphs_to_drop: list or array of IDs of the polymorphs that shall be removed
        """
        for polymorph_id in polymorphs_to_drop:
            self.current_generation.pop(polymorph_id)
    
        self.current_properties = self.current_properties.drop(polymorphs_to_drop, axis=0)


    def discardLeastFittest(self):
        """" Shrinks current generation back to the default generation size by removing the least fittest polymorphs """
        self.current_properties.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        #showGenerationState(polymorphs)
        polymorphs_to_drop = self.current_properties.index[self.generation_size:]
        self.removePolymorphs(polymorphs_to_drop)
        
    
    def discardRandomPolymorphs(self):
        """" Shrinks current generation back to the default generation size by randomly removing polymorphs """
        n_to_drop = len(self.current_generation) - self.generation_size
        
        if n_to_drop > 0:
            polymorphs_to_drop = np.random.choice(self.current_properties.index, (n_to_drop,), replace=False)
            self.removePolymorphs(polymorphs_to_drop)

    def discardByFermiDistribution(self, sigma=None):
        """" Shrinks current generation back to the default generation size by removing polymorphs based on a
        probability distribution
        """

        n_to_drop = len(self.current_generation) - self.generation_size
        
        if n_to_drop <= 0:
            print("Current generation size is less or equal then the default size, " + \
                  "therefore no need to discard polymopphs")
            return
        
        # Sort by fitness_property
        self.current_properties.sort_values(self.fitness_property, axis=0, inplace=True, ascending=True)
        fitness_values = self.current_properties[self.fitness_property]
        min_value = np.nanmin(fitness_values)
        max_value = np.nanmax(fitness_values)
        #cutoff_value = 0.5 * (fitness_values.iloc[-(n_to_drop+1)] + fitness_values.iloc[-n_to_drop])
        median_fitness = 0.5 * (max_value + min_value)
        
        if sigma is None:
            sigma = 0.1 * (max_value - min_value)

        polymorphs_to_drop = []
        polymorphs_to_keep = list(self.current_generation.keys())
        
        mean_fitness = np.mean(fitness_values)
        
        while len(polymorphs_to_drop) < n_to_drop:
            for pm_id in polymorphs_to_keep:
                occupation_chance = fermiDistribution(fitness_values.loc[pm_id], median_fitness, sigma)
                if np.random.rand() > occupation_chance:
                    # Drop this polymorph
                    polymorphs_to_drop.append(pm_id)
                    polymorphs_to_keep.remove(pm_id)
                if len(polymorphs_to_drop) == n_to_drop:
                    break
                    
        self.removePolymorphs(polymorphs_to_drop)
            
    
    def renderGeneration(self, generation_number=-1, shader='lambert'):
        """ Displays geometries of all polymorphs in the current generation """
        if generation_number > self.current_generation_number:
            raise ValueError(f"Generation {generation_number} does not exist (yet)")

        generation = self.population_timeline[generation_number]
        renders = (imolecule.draw(p.gzmat_string, format='gzmat', size=(200, 150),
                                  shader=shader, display_html=False, resizeable=False) \
                   for p in generation.values())
        columns = ('<div class="col-xs-6 col-sm-3">{}</div>'.format(r) for r in renders)
        display(HTML('<div class="row">{}</div>'.format("".join(columns))))
        
    def viewGeneration(self, generation_number=-1):
        if generation_number > self.current_generation_number:
            raise ValueError(f"Generation {generation_number} does not exist (yet)")
        
        generation = self.population_timeline[generation_number]
        
        for p in generation.values():
            p.visualize()
    
    
    def collectGenerationProperties(self):
        for p in self.current_generation.values():
            self.current_properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
    
    
    def listGenerationProperties(self):
        if not np.any(self.current_properties[self.fitness_property] is None or self.current_properties[self.fitness_property] == np.nan):
            return self.current_properties.style.background_gradient(cmap='RdYlGn_r', subset=[self.fitness_property])
        else:
            return self.current_properties
    
    
    def evaluateGeneration(self):
        print("Evaluating current generation ...")
        for p in self.current_generation.values():
            if p.needs_evaluation:
                print(f"Polymorph {p.id} was mutated, needs evaluation")
                p.runHartreeFock(run_dir=self.pyscf_calc_dir)
                # Update Properties
                self.current_properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
            else:
                print(f"Polymorph {p.id} unchanged")

    def evaluateIonizationEnergies(self):
        for p in self.current_generation.values():
            if p.properties[Polymorph.IONIZATION_ENERGY] in [np.nan, None]:
                p.calculateIonizationEnergy()
                
        self.collectGenerationProperties()

    def evaluateElectronAffinities(self):
        for p in self.current_generation.values():
            if p.properties[Polymorph.ELECTRON_AFFINITY] in [np.nan, None]:
                p.calculateElectronAffinity()
    
        self.collectGenerationProperties()

    #### Generate polymorphs
    
    # TODO: Make sure that enough polymorphs are generated
    def fillGeneration(self):
        n_existing = len(self.current_generation)
        new_polymorphs_dict = dict()
        for k in range(self.generation_size - n_existing):
            p = self.factory.generateRandomPolymorph()
            if p is not None:
                self.current_generation[p.id] = p
                new_polymorphs_dict[p.id] = p.properties
                
        new_properties_df = pd.DataFrame.from_dict(new_polymorphs_dict, orient='index', dtype=float)
        self.current_properties = self.current_properties.append(new_properties_df)


    #### Mate polymorphs
    def generateOffsprings(self, mode='random', verbose=False):
        n_pairs = int(np.floor(len(self.current_generation) / 2))
        pair_indices = np.random.permutation(list(self.current_generation.keys()))
        pair_indices = pair_indices[:2 * n_pairs]
        pair_indices = np.reshape(pair_indices, (n_pairs, 2))
       
        if verbose:
            print("Generating offspring ...")
        children_properties_dict = dict()
        for ind_a, ind_b in pair_indices:
            partner_a = self.current_generation[ind_a]
            partner_b = self.current_generation[ind_b]
            child = partner_a.mateWith(partner_b, verbose=verbose)
            self.current_generation[child.id] = child
            children_properties_dict[child.id] = child.properties
    
        children_properties = pd.DataFrame.from_dict(children_properties_dict, orient='index', dtype=float)
        self.current_properties = self.current_properties.append(children_properties)
        
    def attemptCrossovers(self, valid_crossovers_only=False, verbose=False):
        n_pairs = int(np.floor(len(self.current_generation) / 2))
        pair_indices = np.random.permutation(list(self.current_generation.keys()))
        pair_indices = pair_indices[:2 * n_pairs]
        pair_indices = np.reshape(pair_indices, (n_pairs, 2))
        
        print("Attempting gene crossovers ...")
        for ind_a, ind_b in pair_indices:
            partner_a = self.current_generation[ind_a]
            partner_b = self.current_generation[ind_b]
            partner_a.crossoverGenesWith(partner_b, valid_crossovers_only, verbose=verbose)
            
        self.collectGenerationProperties()
        
    def mutateAll(self, verbose=False):
        print("Mutating all polymorphs of current generation...")
        for p in self.current_generation.values():
            p.mutateGenome(verbose=verbose)
            
        self.collectGenerationProperties()
        
    def doGenerationStep(self, discard_mode='default', verbose=True):
        if discard_mode == 'default':
            discard_mode = self.default_discard_mode

        self.current_generation_number += 1
        print("#" + "-" * 60)
        print(f"#  Generation {self.current_generation_number}")
        print("#" + "-" * 60)
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
            
        self.population_timeline.append(self.current_generation.copy())
        self.properties_timeline.append(self.current_properties.copy())
        
    def doMultipleGenerationSteps(self, n_steps, discard_mode='default', verbose=False):
        if discard_mode == 'default':
            discard_mode = self.default_discard_mode
            
        for k in range(n_steps):
            self.doGenerationStep(discard_mode, verbose)
  
    def collectTimelineFor(self, property_key):
        if property_key not in Polymorph.DATA_FIELDS:
            raise ValueError("Invalid data field / property type.")
        
        values_list = list()
        
        for generation_properties in self.properties_timeline:
            values_list.append(generation_properties[property_key])
            
        values = np.array(values_list).transpose()
        return values
    
    def plotTimeline(self, property_key=None, ax=None, cmap="RdYlGn_r", start_with=0):
        if property_key is None:
            property_key = self.fitness_property
            
        values = self.collectTimelineFor(property_key)
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        image = ax.imshow(values[:,start_with:], cmap=cmap)
        plt.colorbar(image, ax=ax)
        ax.set_title(f"Evolution of {property_key}")
        ax.set_xlabel("Number of generations")
        ax.set_ylabel("Polymorphs ranked by fitness")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    def analyzeTimeline(self, property_key=None, ax=None, yscale='linear'):
        if property_key is None:
            property_key = self.fitness_property
            
        values = self.collectTimelineFor(property_key)
        
        mid_index = int(np.floor(self.generation_size/2))
        
        mean_value = np.mean(values, axis=0)
        median_value = values[mid_index,:]
        max_value = np.max(values, axis=0)
        min_value = np.min(values, axis=0)
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        ax.set_yscale(yscale)
        ax.set_title(f"Time evolution for property '{property_key}'")
        ax.set_xlabel("Generation number")
        ax.set_ylabel(f"{property_key} / {Polymorph.DATA_UNITS[property_key]}")
        # Plot timeline for statistical quantities
        ax.plot(mean_value, 'C0', label="average")
        ax.plot(median_value, 'C4:', label="median")
        ax.plot(max_value, 'C3', label="max")
        ax.plot(min_value, 'C2', label="min")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return ax
        
  
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
    factory.freezeAngles('all')
    factory.freezeDihedrals('all-improper')
    ga = GeneticAlgorithm(factory, generation_size=20, default_discard_mode='fermi')
    ga.fillGeneration()
    
    ga.doMultipleGenerationSteps(20, verbose=True)
    
    ga.plotTimeline(Polymorph.TOTAL_ENERGY)
    
    ga.analyzeTimeline(yscale='linear')
    
    bp = ga.factory.base_polymorph
    