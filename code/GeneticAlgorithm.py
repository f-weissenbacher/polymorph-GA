from Polymorph import Polymorph


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
    
