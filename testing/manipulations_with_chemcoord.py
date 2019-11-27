import chemcoord as cc

cc.settings['defaults']['viewer'] = 'avogadro'

molecule = cc.Cartesian.read_xyz("CF3-CH3.xyz")

zmat = molecule.to_zmat()
zmat.get_cartesian().view(viewer='ase-gui')

print(zmat)
zmat.safe_loc[5, 'dihedral'] = 0.0
zmat.safe_loc[1, 'angle'] = 85
print("="*60)
print(zmat)
zmat.get_cartesian().view(viewer='ase-gui')
print("="*60)






