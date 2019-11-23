import chemcoord as cc

cc.settings['defaults']['viewer'] = 'avogadro'

molecule = cc.Cartesian.read_xyz("CF3-CH3.xyz")

zmat = molecule.to_zmat()
print(zmat)
#zmat.safe_loc[5, 'dihedral'] = 0.0
#zmat.safe_loc[6, 'angle'] = 90
#zmat.safe_loc[1, 'angle'] = 90
zmat.safe_loc[4, 'dihedral'] = -120.0
#zmat.safe_loc[7, 'dihedral'] =
print("="*60)
print(zmat)
zmat.get_cartesian().view()

molecule.write_xyz()


