# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:19:03 2019

@author: Fabian
"""

import ase
from Utilities import checkAtomDistances

#from Mutators import FullRangeMutator
from Polymorph import PolymorphFactory, Polymorph

factory = PolymorphFactory("CF3-CH3.xyz")
factory.freezeBonds('all')
factory.setupDefaultMutators()

polymorphs = list()

print(factory.zmat_base)
print("=" * 60)

structure_valid = checkAtomDistances(factory.zmat_base)

for k in range(10):
    p = factory.generateRandomPolymorph(n_max_restarts=1)
    if p is not None:
        print(p.zmat)
        print("="*60)
        p.visualize()


