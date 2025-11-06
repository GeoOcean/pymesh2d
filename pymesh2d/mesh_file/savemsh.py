import numpy as np

def savemsh(name, mesh):
    """
    SAVEMSH: Save a *.MSH file for JIGSAW.

    Parameters:
        name (str): Path to the .MSH file.
        mesh (dict): Dictionary containing the mesh data.
    """
    if not isinstance(name, str):
        raise ValueError("NAME must be a valid file name!")
    if not isinstance(mesh, dict):
        raise ValueError("MESH must be a valid dictionary!")

    # Ensure the file has the correct extension
    if not name.lower().endswith('.msh'):
        name += '.msh'

    # Validate the mesh structure
    certify(mesh)

    try:
        with open(name, 'w') as ffid:
            nver = 3  # Version number

            # Write header
            ffid.write(f"# {name}; created by JIGSAW's Python interface\n")

            mshID = mesh.get('mshID', 'EUCLIDEAN-MESH').upper()

            if mshID in ['EUCLIDEAN-MESH', 'ELLIPSOID-MESH']:
                save_mesh_format(ffid, nver, mesh, mshID)
            elif mshID in ['EUCLIDEAN-GRID', 'ELLIPSOID-GRID']:
                save_grid_format(ffid, nver, mesh, mshID)
            else:
                raise ValueError("Invalid mshID!")

    except Exception as err:
        raise RuntimeError(f"Error writing file {name}: {err}")

def save_mesh_format(ffid, nver, mesh, kind):
    """
    SAVE_MESH_FORMAT: Save mesh data in unstructured-mesh format.

    Parameters:
        ffid (file object): File handle.
        nver (int): Version number.
        mesh (dict): Mesh data.
        kind (str): Mesh kind ('EUCLIDEAN-MESH' or 'ELLIPSOID-MESH').
    """
    ffid.write(f"MSHID={nver};{kind}\n")

    # Write radii if present
    if 'radii' in mesh and mesh['radii'] is not None:
        radii = np.asarray(mesh['radii'])
        if radii.size != 3:
            radii = np.full(3, radii[0])
        ffid.write(f"RADII={radii[0]:.6f};{radii[1]:.6f};{radii[2]:.6f}\n")

    # Write point coordinates
    if 'point' in mesh and 'coord' in mesh['point']:
        coord = np.asarray(mesh['point']['coord'])
        ndim = coord.shape[1] - 1
        ffid.write(f"NDIMS={ndim}\n")
        ffid.write(f"POINT={coord.shape[0]}\n")
        np.savetxt(ffid, coord, fmt="%.16g", delimiter=";")

    # Write other fields (e.g., 'edge2', 'tria3', etc.)
    for field, num_cols in [
        ('edge2', 3), ('tria3', 4), ('quad4', 5),
        ('tria4', 5), ('hexa8', 9), ('wedg6', 7),
        ('pyra5', 6), ('bound', 3)
    ]:
        if field in mesh and 'index' in mesh[field]:
            index = np.asarray(mesh[field]['index'])
            ffid.write(f"{field.upper()}={index.shape[0]}\n")
            np.savetxt(ffid, index, fmt="%d", delimiter=";")

    # Write value data
    if 'value' in mesh:
        value = np.asarray(mesh['value'])
        ffid.write(f"VALUE={value.shape[0]};{value.shape[1]}\n")
        np.savetxt(ffid, value, fmt="%.16g", delimiter=";")

    # Write slope data
    if 'slope' in mesh:
        slope = np.asarray(mesh['slope'])
        ffid.write(f"SLOPE={slope.shape[0]};{slope.shape[1]}\n")
        np.savetxt(ffid, slope, fmt="%.16g", delimiter=";")

def save_grid_format(ffid, nver, mesh, kind):
    """
    SAVE_GRID_FORMAT: Save mesh data in rectilinear-grid format.

    Parameters:
        ffid (file object): File handle.
        nver (int): Version number.
        mesh (dict): Mesh data.
        kind (str): Mesh kind ('EUCLIDEAN-GRID' or 'ELLIPSOID-GRID').
    """
    ffid.write(f"MSHID={nver};{kind}\n")

    # Write radii if present
    if 'radii' in mesh and mesh['radii'] is not None:
        radii = np.asarray(mesh['radii'])
        if radii.size != 3:
            radii = np.full(3, radii[0])
        ffid.write(f"RADII={radii[0]:.6f};{radii[1]:.6f};{radii[2]:.6f}\n")

    # Write grid coordinates
    if 'point' in mesh and 'coord' in mesh['point']:
        coord = mesh['point']['coord']
        ndim = len(coord)
        ffid.write(f"NDIMS={ndim}\n")
        for i, c in enumerate(coord, start=1):
            ffid.write(f"COORD={i};{len(c)}\n")
            np.savetxt(ffid, c, fmt="%.16g", delimiter=";")

    # Write value data
    if 'value' in mesh:
        value = np.asarray(mesh['value'])
        ffid.write(f"VALUE={value.shape[0]};{value.shape[1]}\n")
        np.savetxt(ffid, value, fmt="%.16g", delimiter=";")

    # Write slope data
    if 'slope' in mesh:
        slope = np.asarray(mesh['slope'])
        ffid.write(f"SLOPE={slope.shape[0]};{slope.shape[1]}\n")
        np.savetxt(ffid, slope, fmt="%.16g", delimiter=";")

def certify(mesh):
    """
    CERTIFY: Validate the mesh structure.

    Parameters:
        mesh (dict): Mesh data.

    Raises:
        ValueError: If the mesh is invalid.
    """
    if not isinstance(mesh, dict):
        raise ValueError("MESH must be a valid dictionary!")
    # Additional validation logic can be added here.