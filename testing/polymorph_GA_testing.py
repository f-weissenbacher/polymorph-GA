# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:19:03 2019

@author: Fabian
"""

import ase
from Utilities import checkAtomDistances

#from Mutators import FullRangeMutator
from Polymorph import Polymorph
from PolymorphFactory import PolymorphFactory


#factory = PolymorphFactory("CF3-CH3.xyz")
factory = PolymorphFactory("2pyrCN.xyz")
factory.freezeBonds('all')
factory.setupDefaultMutators()

polymorphs = list()

#print(factory.zmat_base)
#print("=" * 60)

pm = factory.base_polymorph
#pm.scf_basis = '6311++g**'
pm.scf_basis = 'sto-3g'
#pm.scf_basis = '6-31G'
#pm.scf_basis = '6-31G**'
#pm.scf_basis = 'sto-6g'

pm.calculateElectronAffinity()
pm.calculateIonizationEnergy()




