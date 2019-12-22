import imolecule

from IPython.display import display, HTML
carbons = ("c1{}c1".format("c" * i) for i in range(3, 5))
renderer = imolecule.draw(next(carbons), size=(200, 150), shader='lambert', display_html=False)
