import numpy as np

def loadmsh(name):
    """
    LOADMSH: Load a *.MSH file for JIGSAW.

    Parameters:
        name (str): Path to the .MSH file.

    Returns:
        mesh (dict): Dictionary containing the mesh data.
    """
    mesh = {}

    try:
        with open(name, 'r') as ffid:
            real = float
            kind = 'EUCLIDEAN-MESH'
            nver = 0
            ndim = 0

            while True:
                # Read next line from file
                lstr = ffid.readline()
                if not lstr:  # End of file
                    break

                lstr = lstr.strip()
                if len(lstr) == 0 or lstr[0] == '#':
                    continue

                # Tokenize line about '=' character
                tstr = lstr.lower().split('=')
                if len(tstr) != 2:
                    print(f"Warning: Invalid tag: {lstr}")
                    continue

                key, value = tstr[0].strip(), tstr[1].strip()

                if key == 'mshid':
                    # Read "MSHID" data
                    stag = value.split(';')
                    nver = int(stag[0])
                    if len(stag) >= 2:
                        kind = stag[1].strip().upper()

                elif key == 'ndims':
                    # Read "NDIMS" data
                    ndim = int(value)

                elif key == 'radii':
                    # Read "RADII" data
                    stag = value.split(';')
                    if len(stag) == 3:
                        mesh['radii'] = [float(stag[0]), float(stag[1]), float(stag[2])]
                    else:
                        print(f"Warning: Invalid RADII: {lstr}")

                elif key == 'point':
                    # Read "POINT" data
                    nnum = int(value)
                    data = np.loadtxt(ffid, max_rows=nnum, delimiter=';')
                    mesh['point'] = {'coord': data}

                elif key == 'coord':
                    # Read "COORD" data
                    stag = value.split(';')
                    idim = int(stag[0])
                    cnum = int(stag[1])
                    data = np.loadtxt(ffid, max_rows=cnum, delimiter=';')
                    if 'point' not in mesh:
                        mesh['point'] = {}
                    if 'coord' not in mesh['point']:
                        mesh['point']['coord'] = {}
                    mesh['point']['coord'][idim] = data

                elif key in ['edge2', 'tria3', 'quad4', 'tria4', 'hexa8', 'wedg6', 'pyra5', 'bound']:
                    # Read element data
                    nnum = int(value)
                    data = np.loadtxt(ffid, max_rows=nnum, delimiter=';')
                    mesh[key] = {'index': data}

                elif key in ['value', 'slope', 'power']:
                    # Read "VALUE", "SLOPE", or "POWER" data
                    stag = value.split(';')
                    nnum = int(stag[0])
                    vnum = int(stag[1])
                    numr = nnum * vnum
                    data = np.fromfile(ffid, count=numr, sep=';').reshape(nnum, vnum)
                    
                    mesh[key] = data

            mesh['mshID'] = kind
            mesh['fileV'] = nver

            # Reshape grid data if necessary
            if ndim > 0 and kind in ['EUCLIDEAN-GRID', 'ELLIPSOID-GRID']:
                if 'value' in mesh and 'point' in mesh:
                    if ndim == 2:
                        mesh['value'] = mesh['value'].reshape(
                            len(mesh['point']['coord'][1]),
                            len(mesh['point']['coord'][0]),
                            -1
                        )
                    elif ndim == 3:
                        mesh['value'] = mesh['value'].reshape(
                            len(mesh['point']['coord'][1]),
                            len(mesh['point']['coord'][0]),
                            len(mesh['point']['coord'][2]),
                            -1
                        )
                if 'slope' in mesh and 'point' in mesh:
                    if ndim == 2:
                        mesh['slope'] = mesh['slope'].reshape(
                            len(mesh['point']['coord'][1]),
                            len(mesh['point']['coord'][0]),
                            -1
                        )
                    elif ndim == 3:
                        mesh['slope'] = mesh['slope'].reshape(
                            len(mesh['point']['coord'][1]),
                            len(mesh['point']['coord'][0]),
                            len(mesh['point']['coord'][2]),
                            -1
                        )

    except Exception as err:
        print(f"Error reading file {name}: {err}")
        raise

    return mesh