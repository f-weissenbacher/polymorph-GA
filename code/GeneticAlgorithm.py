import imolecule
from imolecule import format_converter
from Utilities import checkAtomDistances, fermiDistribution
import Polymorph
from Polymorph import Polymorph
from PolymorphFactory import PolymorphFactory

from Mutators import FullRangeMutator, IncrementalMutator, PlaceboMutator, MultiplicativeMutator

import pandas as pd
import numpy as np
import time
import pybel
import shutil
import os
import pickle

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

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


def assignMatingPairs(n_pairs, mating_pool):
    """ Randomly assigns polymorphs from the mating pool into mating pairs.
    :param int n_pairs: Number of mating pairs to form
    :param mating_pool: List or Iterable of IDs of possible mating partners
    :return pair_indices: Pairs of polymorph IDs in form of numpy array of shape (n_pairs, 2)
    """
    
    mating_pool_size = len(mating_pool)
    
    pair_indices = None
    for k in range(int(np.ceil(2 * n_pairs / mating_pool_size))):
        if k == 0:
            pair_indices = np.random.permutation(mating_pool)
        else:
            pair_indices = np.append(pair_indices, np.random.permutation(mating_pool))
    
    pair_indices = pair_indices[:2 * n_pairs]
    pair_indices = np.reshape(pair_indices, (n_pairs, 2))
    
    return pair_indices


class GeneticAlgorithm:
    def __init__(self, factory: PolymorphFactory, generation_size=10, fitness_property=Polymorph.TOTAL_ENERGY,
                 fitness_goal='minimize', fraction_parents_survive=0.2, work_dir=None,
                 discard_mode='least-fittest', matchmaking_mode='roulette'):
        
        self.factory = factory
        
        self.current_generation = dict() # Current generation of polymorphs, polymorph id's are keys
        self.current_properties = pd.DataFrame(columns=Polymorph.DATA_FIELDS, dtype=float) # Properties of current generation
        
        if fitness_property not in Polymorph.DATA_FIELDS:
            raise ValueError("Unknown fitness property. Valid options: " + \
                             "".join([f"'{p}'" for p in Polymorph.DATA_FIELDS]))
        
        self.fitness_property = fitness_property
        self.fitness_goal = fitness_goal
        self.fitness_sort_ascending = False # Default behavior: Maximize fitness value
        if self.fitness_goal == 'minimize':
            self.fitness_sort_ascending = True
        
        self.generation_size = generation_size
        #self.current_generation_number = 0
        self.population_timeline = []
        self.properties_timeline = []
        self.discard_mode = discard_mode
        self.matchmaking_mode = matchmaking_mode
        self.n_surviving_parents = int(np.ceil(self.generation_size * fraction_parents_survive))
        
        if work_dir is None:
            work_dir = os.getcwd()
        else:
            makeSureDirectoryExists(work_dir)
            
        self.work_dir = work_dir
        self.pyscf_calc_dir = os.path.join(work_dir, "pyscf-calculations")
        makeSureDirectoryExists(self.pyscf_calc_dir)
        
        self.fillCurrentGeneration()
        
    @property
    def current_generation_number(self):
        return len(self.population_timeline)
    

    def fitnessRanking(self, generation_number=None):
        """ Returns a pandas.Series object that contains the fitness value for all polymorphs in the specified
        generation, ranked from fittest to least fittest. Entries are labeled with the corresponding polymorph ID"""
        if generation_number is None:
            properties = self.current_properties
        else:
            properties = self.properties_timeline[generation_number]
            
        return properties[self.fitness_property].sort_values(ascending=self.fitness_sort_ascending)
     
    def firstValueIsFitter(self, value1, value2):
        if np.isnan(value1):
            return False
        if np.isnan(value2):
            return True
        if self.fitness_sort_ascending:
            return value1 < value2
        else:
            return value1 > value2
        
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

    def discardLeastFittest(self, n):
        """" Removes the *n* least fittest polymorphs from the current generation"""
        polymorphs_to_drop = self.fitnessRanking().index[-n:]
        self.removePolymorphs(polymorphs_to_drop)
    
    def discardRandomPolymorphs(self, n):
        """" Remove *n* random polymorphs """
        if 0 < n <= self.generation_size:
            polymorphs_to_drop = np.random.choice(self.current_properties.index, (n,), replace=False)
            self.removePolymorphs(polymorphs_to_drop)

    def discardByFermiDistribution(self, n, sigma=0.2):
        """" Shrinks current generation by removing *n* polymorphs based on a Fermi probability distribution """
        
        if n < 1 or n > self.generation_size:
            raise ValueError()

        fitness_values = self.fitnessRanking()
        min_value = np.nanmin(fitness_values)
        max_value = np.nanmax(fitness_values)
        mean_fitness = np.nanmean(fitness_values)

        sigma = (max_value - min_value) * sigma

        polymorphs_to_keep = list(self.current_generation.keys())
        polymorphs_to_drop = []

        if self.fitness_sort_ascending:
            sign = 1
        else:
            sign = -1
        
        while len(polymorphs_to_drop) < n:
            for pm_id in polymorphs_to_keep:
                occupation_chance = fermiDistribution(fitness_values.loc[pm_id], mean_fitness, sigma, sign)
                if np.random.rand() > occupation_chance:
                    # Drop this polymorph
                    polymorphs_to_drop.append(pm_id)
                    polymorphs_to_keep.remove(pm_id)
                if len(polymorphs_to_drop) == n:
                    break
                    
        self.removePolymorphs(polymorphs_to_drop)
            
    
    # Generation properties -------------------------------------------
    
    def collectGenerationProperties(self):
        for p in self.current_generation.values():
            self.current_properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
    
    
    def listGenerationProperties(self):
        if not np.any(self.current_properties[self.fitness_property] is None or self.current_properties[self.fitness_property] == np.nan):
            return self.current_properties.style.background_gradient(cmap='RdYlGn_r', subset=[self.fitness_property])
        else:
            return self.current_properties
    
    
    def evaluateCurrentGeneration(self):
        print(f"Evaluating current generation (# {self.current_generation_number}) ...")
        for p in self.current_generation.values():
            if p.needs_evaluation:
                print(f"Polymorph {p.id} was mutated, needs evaluation")
                p.evaluate(self.fitness_property)
                # Update Properties
                self.current_properties.loc[[p.id]] = pd.DataFrame.from_dict({p.id: p.properties}, orient='index', dtype=float)
            else:
                print(f"Polymorph {p.id} unchanged")


    #### Polymorph generation / Initialization of a generation ---------------------------------------------------------
    # TODO: Make sure that enough polymorphs are generated
    def fillCurrentGeneration(self):
        n_existing = len(self.current_generation)
        new_polymorphs_dict = dict()
        for k in range(self.generation_size - n_existing):
            p = self.factory.generateRandomPolymorph()
            if p is not None:
                self.current_generation[p.id] = p
                new_polymorphs_dict[p.id] = p.properties
                
        new_properties_df = pd.DataFrame.from_dict(new_polymorphs_dict, orient='index', dtype=float)
        self.current_properties = self.current_properties.append(new_properties_df)

    #### Genetic diversity
    
    def analyzeGeneticDiversity(self, comparison_mode='average', bonds_eps=1e-2, angles_eps=0.1, dihedrals_eps=0.1):
        families = []
        polymorph_ids = list(self.current_generation.keys())
        
        families.append([polymorph_ids.pop()])
        
        for pm1_id in polymorph_ids:
            pm1 = self.current_generation[pm1_id]
            pm1_found_family = False
            for family in families:
                for pm2_id in family:
                    pm2 = self.current_generation[pm2_id]
                    mismatch = pm1.calculateGenomeDifference(pm2, comparison_mode)
                    if mismatch[0] < bonds_eps and mismatch[1] < angles_eps and mismatch[2] < dihedrals_eps:
                        family.append(pm1)
                        pm1_found_family = True
                        break
                if pm1_found_family:
                    break
            else:
                families.append([pm1_id])
                
        return families

    #### Mating process -----------------------------------------------------------------------------------------------

    def selectRandomPolymorphs(self, n_pairs):
        """ Draws random mating pairs from all polymorphs of the current generation """
        mating_pool = list(self.current_generation.keys())
        return assignMatingPairs(n_pairs, mating_pool)
    
    def selectFromMostFittest(self, n_pairs, n_most_fittest):
        """ Selects the *n_most_fittest* polymorphs from the current generation and randomly assigns them to
        *n_pairs* mating pairs. If n_pairs > n_most_fittest / 2, some polymorphs will mate more than once.
        
        :param int n_pairs:
        :param int n_most_fittest: Number of potential mating partners. Valid range: 2 <= n_most_fittest <= generation_size
        :return mating_pairs:
        """
        fitness_ranking = self.fitnessRanking()
        mating_pool = fitness_ranking.index[:n_most_fittest].to_numpy()
        
        return assignMatingPairs(n_pairs, mating_pool)
    
    def selectFromLeastFittest(self, n_pairs, n_least_fittest):
        """ Selects the *n_least_fittest* polymorphs from the current generation and randomly assigns them to
        *n_pairs* mating pairs. If n_pairs > n_least_fittest / 2, some polymorphs will mate more than once.

        :param int n_pairs:
        :param int n_least_fittest: Number of potential mating partners. Valid range: 2 <= n_least_fittest <= generation_size
        :return mating_pairs:
        """
        fitness_ranking = self.fitnessRanking()
        mating_pool = fitness_ranking.index[-n_least_fittest:].to_numpy()

        return assignMatingPairs(n_pairs, mating_pool)
    
    def selectByFermiDistribution(self, n_pairs, mating_pool_size, sigma=0.1, drawing_mode='regular'):
        """ Selects *mating_pool_size* potential mating partners based on a Fermi distribution and their respective
        fitness value. Mating pairs are then drawn randomly from all polymorphs of the resulting mating pool.
        The acceptance probability for a polymorph with fitness 'x' is given by:
          
          p(x) = 1 / (1 + exp(+-(x - mu)/((x_max -x_min) * sigma)) ,
          
        where mu is equal to the average fitness, x_max and x_min are the maximum/minimum fitness value and sigma
        controls the slope of the distribution function. The sign in the exponent depends on whether the fitness value
        is to be maximized or minimized.
        """
        
        fitness_ranking = self.fitnessRanking()
        # Remove polymorphs with fitness = nan
        fitness_ranking = fitness_ranking[~np.isnan(fitness_ranking)]

        # Limit size of mating pool
        mating_pool_size = min(mating_pool_size, len(fitness_ranking))

        # Determine range of values
        min_value = np.nanmin(fitness_ranking)
        max_value = np.nanmax(fitness_ranking)
        mean_fitness = np.nanmean(fitness_ranking)

        sigma = (max_value - min_value) * sigma

        polymorph_list = list(fitness_ranking.index)
        mating_pool = []
        
        if self.fitness_sort_ascending:
            sign = 1
        else:
            sign = -1
            
        if drawing_mode == 'inverted':
            sign *= -1

        while len(mating_pool) < mating_pool_size:
            for pm_id in polymorph_list:
                occupation_chance = fermiDistribution(fitness_ranking.loc[pm_id], mean_fitness, sigma, sign)
                if np.random.rand() < occupation_chance:
                    # Add to mating pool
                    mating_pool.append(pm_id)
                    polymorph_list.remove(pm_id)
                # Check if mating pool is complete
                if len(mating_pool) == mating_pool_size:
                    break

        return assignMatingPairs(n_pairs, mating_pool)
    
    def generateOffsprings(self, n_offsprings, mode='random', mating_pool_size=None, validate_children=True,
                           prohibit_inbreeding=True, verbose=False):
        """ Generates offsprings by mating polymorphs of the current generation with a process defined by
        argument 'mode'
        
        :param int n_offsprings:
        :param str mode: Sets match making type
        :param int mating_pool_size:
        :param bool validate_children: When 'True', only valid children are accepted
        :param bool verbose:
        :return dict child_polymorphs:
        :return DataFrame child_properties
        """
        if mating_pool_size is None:
            mating_pool_size = self.generation_size
        
        if mode == 'random':
            mating_pairs = self.selectRandomPolymorphs(n_offsprings)
        elif mode == 'most-fittest':
            mating_pairs = self.selectFromMostFittest(n_offsprings, mating_pool_size)
        elif mode == 'least-fittest':
            mating_pairs = self.selectFromLeastFittest(n_offsprings, mating_pool_size)
        elif mode == 'roulette':
            mating_pairs = self.selectByFermiDistribution(n_offsprings, mating_pool_size, drawing_mode='regular')
        elif mode == 'inverted-roulette':
            mating_pairs = self.selectByFermiDistribution(n_offsprings, mating_pool_size, drawing_mode='inverted')
        else:
            raise ValueError("Invalid selection / matchmaking mode. Options are: 'random', 'roulette', " + \
                             "'inverted-roulette' and 'tournament'")
       
        if verbose:
            print("Generating offsprings ...")
            
        # Create offspring and store in new generation
        children_properties_dict = dict()
        child_polymorphs = {}
        for ind_a, ind_b in mating_pairs:
            partner_a = self.current_generation[ind_a]
            partner_b = self.current_generation[ind_b]
            # Mate partners, only accept if child is valid
            child = None
            while child is None:
                child = partner_a.mateWith(partner_b, validate_child=validate_children, verbose=verbose)
                
            child_polymorphs[child.id] = child
            children_properties_dict[child.id] = child.properties
    
        children_properties = pd.DataFrame.from_dict(children_properties_dict, orient='index', dtype=float)
        
        return child_polymorphs, children_properties
     
     
     #### Mutation and Crossovers ------------------------------------------------------------------------------------ #
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
      
      
    #### Algorithm execution ----------------------------------------------------------------------------------------- #
    def doGenerationStep(self, matchmaking_mode='default', discard_mode='default', verbose=False):
        if discard_mode == 'default':
            discard_mode = self.discard_mode
            
        if matchmaking_mode == 'default':
            matchmaking_mode = self.matchmaking_mode

        #self.current_generation_number += 1
        print("#" + "-" * 60)
        print(f"#  Generation {self.current_generation_number}")
        print("#" + "-" * 60)
        self.mutateAll(verbose=verbose)
        self.attemptCrossovers(verbose=verbose)
        # Evaluate changes
        self.evaluateCurrentGeneration()
        print(f"Adding generation {self.current_generation_number} to population_timeline")
        # Save current generation to timeline
        self.population_timeline.append(self.current_generation.copy())
        self.properties_timeline.append(self.current_properties.copy())
        
        # Start next generation
       
        # TODO: Check for convergence at this point
        # Offsprings:
        print("-" * 60)
        print(f"Creating next generation (# {self.current_generation_number})")
        n_offsprings = self.generation_size - self.n_surviving_parents
        child_polymorphs, child_properties = self.generateOffsprings(n_offsprings, mode=matchmaking_mode, verbose=verbose)
        
        n_to_discard = len(child_polymorphs)
        # Discard (most) parents
        if discard_mode == 'least-fittest':
            self.discardLeastFittest(n_to_discard)
        elif discard_mode == 'random':
            self.discardRandomPolymorphs(n_to_discard)
        elif discard_mode == 'fermi':
            self.discardByFermiDistribution(n_to_discard)
        else:
            ValueError("Unknown discard mode. Valid options are: 'least-fittest', 'random' and 'fermi'")

        print("Updating current_generation and current_properties")
        # Update current_generation with child polymorphs and their properties
        self.current_generation.update(child_polymorphs)
        self.current_properties = self.current_properties.append(child_properties)
        
        
    def doMultipleGenerationSteps(self, n_steps, matchmaking_mode='default', discard_mode='default', verbose=False):
        if discard_mode == 'default':
            discard_mode = self.discard_mode
    
        if matchmaking_mode == 'default':
            matchmaking_mode = self.matchmaking_mode
            
        for k in range(n_steps):
            self.doGenerationStep(matchmaking_mode, discard_mode, verbose=verbose)
            
        self.evaluateCurrentGeneration()

    #### Visualizing / Analyzing a generation ------------------------------------------------------------------------ #

    def renderGeneration(self, generation_number=-1, shader='lambert'):
        """ Displays geometries of all polymorphs in the current generation """
        if generation_number > self.current_generation_number:
            raise ValueError(f"Generation {generation_number} does not exist (yet)")
    
        if generation_number == -1:
            generation = self.current_generation
        else:
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
    
   
    def findBestPolymorphOfGeneration(self, generation_number=-1):
        best_id = self.fitnessRanking(generation_number).index[0]
        return self.population_timeline[generation_number][best_id]
        
    def showBestPolymorphOfGeneration(self, generation_number=-1):
        best_pm = self.findBestPolymorphOfGeneration(generation_number)
        best_pm.visualize()
        
  
    #### Timeline: Analysis and Visualization ------------------------------------------------------------------------ #
    
    def findBestPolymorphOfAll(self):
        if self.fitness_sort_ascending:
            best_value = np.inf
        else:
            best_value = -np.inf
        
        best_id = -1
        best_time = -1
        for gen_number, generation in enumerate(self.population_timeline):
            generation_ranking = self.fitnessRanking(gen_number)
            generation_best = generation_ranking.values[0]
            if self.firstValueIsFitter(generation_best, best_value):
                best_id = generation_ranking.index[0]
                best_time = gen_number

        best_pm =  self.population_timeline[best_time][best_id]
        print(f"Best overall polymorph: {best_pm} in generation {best_time}.")
        print(f"Fitness: {self.fitness_property} = {best_pm.properties[self.fitness_property]}" + \
              f" {Polymorph.DATA_UNITS[self.fitness_property]}")
        return best_pm
    
    def showBestPolymorphOfAll(self):
        best_pm = self.findBestPolymorphOfAll()
        best_pm.visualize()
    
    def findPolymorphInTimeline(self, polymorph_id):
        occurrences = dict()
        for gen_number, generation in enumerate(self.population_timeline):
            if polymorph_id in generation.keys():
                occurrences[gen_number] = generation[polymorph_id]
    
        return occurrences
    
    
    def collectTimelineFor(self, property_key):
        if property_key not in Polymorph.DATA_FIELDS:
            raise ValueError("Invalid data field / property type.")
        
        values_list = list()
        
        sort_ascending = True
        if property_key == self.fitness_property:
            sort_ascending = self.fitness_sort_ascending
        
        for generation_properties in self.properties_timeline:
            values_list.append(generation_properties[property_key].sort_values(ascending=sort_ascending))
            
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
    from Utilities import deleteTempfiles
    molecules_dir = os.path.abspath(join(os.path.dirname(__file__),"../molecules"))
    testing_dir = os.path.abspath(join(os.path.dirname(__file__),"../testing"))
    structure_filepath = join(molecules_dir, "ethane.xyz")
    
    os.chdir(testing_dir)
    mutation_rate = 0.05
    crossover_rate = 0.01
    
    #bond_mutator = MultiplicativeMutator('bond',[0.8,1.2],[0.8,1.2], False, mutation_rate)
    
    factory = PolymorphFactory(structure_filepath, mutation_rate, crossover_rate, bond_mutator=None)
    factory.freezeBonds('all')
    factory.freezeAngles('all')
    #factory.freezeDihedrals('all')
    ga = GeneticAlgorithm(factory, generation_size=15, discard_mode='least-fittest',
                          fitness_property=Polymorph.TOTAL_ENERGY, fitness_goal='minimize',
                          matchmaking_mode='roulette')
    #ga.fillCurrentGeneration()
    
    #ga.doGenerationStep()
    ga.doMultipleGenerationSteps(10, verbose=True)
    ga.plotTimeline()
    ga.analyzeTimeline(yscale='linear')
    #bp = ga.factory.base_polymorph
    
    ga.analyzeGeneticDiversity()
    
    deleteTempfiles(testing_dir)
    