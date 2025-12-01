import numpy as np

from .certify import certify


def savemsh(name, mesh):
    """
    Save a `.MSH` file for JIGSAW.

    This function writes a JIGSAW-formatted mesh file (`.MSH`) from a given
    mesh data structure. Entities are written only if present in the `mesh`
    dictionary. Supports Euclidean, ellipsoidal, and structured grid formats.

    Parameters
    ----------
    name : str
        Path or base name of the `.MSH` file to save (without extension).
    mesh : dict
        Dictionary containing the mesh data to be written. The structure
        depends on the mesh type (`mesh["MSHID"]`).

        **If `mesh["MSHID"] == "EUCLIDEAN-MESH"`:**
            - `POINT.COORD` : (NP, ND+1) ndarray
              Point coordinates and unique IDs.
            - `POINT.POWER` : (NP,) ndarray
              Vertex weights for dual “power” tessellations.
            - `EDGE2.INDEX`, `TRIA3.INDEX`, `QUAD4.INDEX`,
              `TRIA4.INDEX`, `HEXA8.INDEX`, `WEDG6.INDEX`, `PYRA5.INDEX` : ndarrays
              Connectivity and element IDs for 1D–3D elements.
            - `BOUND.INDEX` : (NB, 3) ndarray
              Boundary-to-part associations and element type tags.
            - `VALUE` : (NP, NV) ndarray
              Scalar or vector values associated with vertices.
            - `SLOPE` : (NP,) ndarray
              Gradient-limit values `||dh/dx||` used by the Eikonal solver `MARCHE`.

        **If `mesh["MSHID"] == "ELLIPSOID-MESH"`:**
            - `RADII` : (3,) ndarray
              Principal ellipsoid radii.
            - Additional entities as in `"EUCLIDEAN-MESH"` may be included.

        **If `mesh["MSHID"] == "EUCLIDEAN-GRID"` or `"ELLIPSOID-GRID"`:**
            - `POINT.COORD` : list of ND ndarrays
              Coordinate vectors along each spatial axis.
            - `VALUE` : (NM, NV) ndarray
              Values at grid vertices (`NM` = product of grid dimensions).
            - `SLOPE` : (NM,) ndarray
              Gradient-limit values at grid vertices.

    Notes
    -----
    - Only entities present in the input structure are written to file.
    - The `BOUND.INDEX` field defines how mesh elements are associated
      with topological parts, using JIGSAW’s internal `LIBDATA` constants.
    - File format is compatible with JIGSAW and JIGSAW-MATLAB.

    See Also
    --------
    loadmsh : Load a `.MSH` file.
    jigsaw : Run the JIGSAW mesh generator.

    References
    ----------
    Based on the JIGSAW MATLAB implementation by Darren Engwirda.
    Repository: https://github.com/dengwirda/jigsaw-matlab
    """

    if not isinstance(name, str):
        raise ValueError("NAME must be a valid file name!")
    if not isinstance(mesh, dict):
        raise ValueError("MESH must be a valid dictionary!")

    # Ensure the file has the correct extension
    if not name.lower().endswith(".msh"):
        name += ".msh"

    # Validate the mesh structure
    certify(mesh)

    try:
        with open(name, "w") as ffid:
            nver = 3  # Version number

            # Write header
            ffid.write(f"# {name}; created by JIGSAW's Python interface\n")

            mshID = mesh.get("mshID", "EUCLIDEAN-MESH").upper()

            if mshID in ["EUCLIDEAN-MESH", "ELLIPSOID-MESH"]:
                save_mesh_format(ffid, nver, mesh, mshID)
            elif mshID in ["EUCLIDEAN-GRID", "ELLIPSOID-GRID"]:
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

    References
    ----------
    Translation of the MESH2D function `SAVE_MESH_FORMAT`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    ffid.write(f"MSHID={nver};{kind}\n")

    # Write radii if present
    if "radii" in mesh and mesh["radii"] is not None:
        radii = np.asarray(mesh["radii"])
        if radii.size != 3:
            radii = np.full(3, radii[0])
        ffid.write(f"RADII={radii[0]:.6f};{radii[1]:.6f};{radii[2]:.6f}\n")

    # Write point coordinates
    if "point" in mesh and "coord" in mesh["point"]:
        coord = np.asarray(mesh["point"]["coord"])
        ndim = coord.shape[1] - 1
        ffid.write(f"NDIMS={ndim}\n")
        ffid.write(f"POINT={coord.shape[0]}\n")
        np.savetxt(ffid, coord, fmt="%.16g", delimiter=";")

    # Write other fields (e.g., 'edge2', 'tria3', etc.)
    for field, num_cols in [
        ("edge2", 3),
        ("tria3", 4),
        ("quad4", 5),
        ("tria4", 5),
        ("hexa8", 9),
        ("wedg6", 7),
        ("pyra5", 6),
        ("bound", 3),
    ]:
        if field in mesh and "index" in mesh[field]:
            index = np.asarray(mesh[field]["index"])
            ffid.write(f"{field.upper()}={index.shape[0]}\n")
            np.savetxt(ffid, index, fmt="%d", delimiter=";")

    # Write value data
    if "value" in mesh:
        value = np.asarray(mesh["value"])
        ffid.write(f"VALUE={value.shape[0]};{value.shape[1]}\n")
        np.savetxt(ffid, value, fmt="%.16g", delimiter=";")

    # Write slope data
    if "slope" in mesh:
        slope = np.asarray(mesh["slope"])
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
    if "radii" in mesh and mesh["radii"] is not None:
        radii = np.asarray(mesh["radii"])
        if radii.size != 3:
            radii = np.full(3, radii[0])
        ffid.write(f"RADII={radii[0]:.6f};{radii[1]:.6f};{radii[2]:.6f}\n")

    # Write grid coordinates
    if "point" in mesh and "coord" in mesh["point"]:
        coord = mesh["point"]["coord"]
        ndim = len(coord)
        ffid.write(f"NDIMS={ndim}\n")
        for i, c in enumerate(coord, start=1):
            ffid.write(f"COORD={i};{len(c)}\n")
            np.savetxt(ffid, c, fmt="%.16g", delimiter=";")

    # Write value data
    if "value" in mesh:
        value = np.asarray(mesh["value"])
        ffid.write(f"VALUE={value.shape[0]};{value.shape[1]}\n")
        np.savetxt(ffid, value, fmt="%.16g", delimiter=";")

    # Write slope data
    if "slope" in mesh:
        slope = np.asarray(mesh["slope"])
        ffid.write(f"SLOPE={slope.shape[0]};{slope.shape[1]}\n")
        np.savetxt(ffid, slope, fmt="%.16g", delimiter=";")
