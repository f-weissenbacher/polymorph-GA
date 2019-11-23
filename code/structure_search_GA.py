# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:19:03 2019

@author: Fabian
"""

import deap
import pyscf

from Mutators import FullRangeMutator

from Polymorph import PolymorphFactory

factory = PolymorphFactory("CF3-CH3.xyz")
factory.freezeBonds('all')
factory.setupDefaultMutators()

polymorphs = list()

print(factory.zmat_base)
print("=" * 60)

for k in range(3):
    p = factory.generateRandomPolymorph()
    print(p.zmat)
    print("="*60)
    p.saveStructure(f"polymorph-{k:02d}.xyz")


