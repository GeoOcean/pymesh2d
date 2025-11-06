from .mesh_file.loadmsh import loadmsh

def triread(name):
    """
    TRIREAD : lire des données de triangulation 2D à partir d’un fichier.
    
    Parameters
    ----------
    name : str
        Nom du fichier contenant la triangulation.
    
    Returns
    -------
    vert : ndarray (V,2)
        Coordonnées XY des sommets.
    edge : ndarray (E,2)
        Arêtes contraintes.
    tria : ndarray (T,3)
        Triangles (indices des sommets).
    tnum : ndarray (T,1)
        Indices des parties.
    
    Notes
    -----
    - Les données sont renvoyées non vides si elles sont présentes dans le fichier.
    - Cette routine reprend des fonctionnalités du package JIGSAW : 
      github.com/dengwirda/jigsaw-matlab
    """

    import numpy as np

    vert, edge, tria, tnum = None, None, None, None

    # -------------------------------------------- basic checks
    if not isinstance(name, str):
        raise TypeError("triread:incorrectInputClass - Incorrect input class.")

    # ----------------------------------- borrow JIGSAW I/O func!
    mesh = loadmsh(name)   # ⚠️ nécessite une fonction loadmsh() équivalente en Python

    # ----------------------------------- extract data if present
    if "point" in mesh and "coord" in mesh["point"]:
        vert = np.array(mesh["point"]["coord"])[:, :2]

    if "edge2" in mesh and "index" in mesh["edge2"]:
        edge = np.array(mesh["edge2"]["index"])[:, :2]

    if "tria3" in mesh and "index" in mesh["tria3"]:
        arr = np.array(mesh["tria3"]["index"])
        tria = arr[:, :3]
        tnum = arr[:, 3]

    return vert, edge, tria, tnum
