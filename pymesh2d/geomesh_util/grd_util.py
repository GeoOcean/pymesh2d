from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def adcirc2DFlowFM(NODE: np.ndarray, EDGE: np.ndarray, netcdf_path: str) -> None:
    """
    Converts ADCIRC grid data to a NetCDF Delft3DFM format.

    Parameters
    ----------
    NODE : np.ndarray
        Array of shape (n_nodes, 3) containing node coordinates (x, y, z).
    EDGE : np.ndarray
        Array of shape (n_elements, 3) containing triangle connectivity (node indices).
    netcdf_path : str
        Path to the output NetCDF file.
    """

    edges = calculate_edges(EDGE) + 1
    EDGE_S = np.sort(EDGE, axis=1)
    EDGE_S = EDGE_S[EDGE_S[:, 2].argsort()]
    EDGE_S = EDGE_S[EDGE_S[:, 1].argsort()]
    face_node = np.array(EDGE_S[EDGE_S[:, 0].argsort()], dtype=np.int32)
    edge_node = np.zeros([len(edges), 2], dtype="i4")
    edge_face = np.zeros([len(edges), 2], dtype=np.double)
    edge_x = np.zeros(len(edges))
    edge_y = np.zeros(len(edges))

    edge_node = np.array(
        edge_node,
        dtype=np.int32,
    )

    face_x = (
        NODE[EDGE[:, 0].astype(int), 0]
        + NODE[EDGE[:, 1].astype(int), 0]
        + NODE[EDGE[:, 2].astype(int), 0]
    ) / 3
    face_y = (
        NODE[EDGE[:, 0].astype(int), 1]
        + NODE[EDGE[:, 1].astype(int), 1]
        + NODE[EDGE[:, 2].astype(int), 1]
    ) / 3

    edge_x = (NODE[edges[:, 0] - 1, 0] + NODE[edges[:, 1] - 1, 0]) / 2
    edge_y = (NODE[edges[:, 0] - 1, 1] + NODE[edges[:, 1] - 1, 1]) / 2

    face_node_dict = {}

    for idx, face in enumerate(face_node):
        for node in face:
            if node not in face_node_dict:
                face_node_dict[node] = []
            face_node_dict[node].append(idx)

    for i, edge in enumerate(edges):
        node1, node2 = map(int, edge)

        edge_node[i, 0] = node1
        edge_node[i, 1] = node2

        faces_node1 = face_node_dict.get(node1 - 1, [])
        faces_node2 = face_node_dict.get(node2 - 1, [])

        faces = list(set(faces_node1) & set(faces_node2))

        if len(faces) < 2:
            edge_face[i, 0] = faces[0] + 1 if faces else 0
            edge_face[i, 1] = 0
        else:
            edge_face[i, 0] = faces[0] + 1
            edge_face[i, 1] = faces[1] + 1

    face_x = np.array(face_x, dtype=np.double)
    face_y = np.array(face_y, dtype=np.double)

    node_x = np.array(NODE[:, 0], dtype=np.double)
    node_y = np.array(NODE[:, 1], dtype=np.double)
    node_z = np.array(NODE[:, 2], dtype=np.double)

    face_x_bnd = np.array(node_x[face_node], dtype=np.double)
    face_y_bnd = np.array(node_y[face_node], dtype=np.double)

    num_nodes = NODE.shape[0]
    num_faces = EDGE.shape[0]
    num_edges = edges.shape[0]

    with Dataset(netcdf_path, "w", format="NETCDF4") as dataset:
        _mesh2d_nNodes = dataset.createDimension("mesh2d_nNodes", num_nodes)
        _mesh2d_nEdges = dataset.createDimension("mesh2d_nEdges", num_edges)
        _mesh2d_nFaces = dataset.createDimension("mesh2d_nFaces", num_faces)
        _mesh2d_nMax_face_nodes = dataset.createDimension("mesh2d_nMax_face_nodes", 3)
        _two_dim = dataset.createDimension("Two", 2)

        mesh2d_node_x = dataset.createVariable(
            "mesh2d_node_x", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_x.standard_name = "projection_x_coordinate"
        mesh2d_node_x.long_name = "x-coordinate of mesh nodes"

        mesh2d_node_y = dataset.createVariable(
            "mesh2d_node_y", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_y.standard_name = "projection_y_coordinate"
        mesh2d_node_y.long_name = "y-coordinate of mesh nodes"

        mesh2d_node_z = dataset.createVariable(
            "mesh2d_node_z", "f8", ("mesh2d_nNodes",)
        )
        mesh2d_node_z.units = "m"
        mesh2d_node_z.standard_name = "altitude"
        mesh2d_node_z.long_name = "z-coordinate of mesh nodes"

        mesh2d_edge_x = dataset.createVariable(
            "mesh2d_edge_x", "f8", ("mesh2d_nEdges",)
        )
        mesh2d_edge_x.standard_name = "projection_x_coordinate"
        mesh2d_edge_x.long_name = (
            "Characteristic x-coordinate of the mesh edge (e.g., midpoint)"
        )

        mesh2d_edge_y = dataset.createVariable(
            "mesh2d_edge_y", "f8", ("mesh2d_nEdges",)
        )
        mesh2d_edge_y.standard_name = "projection_y_coordinate"
        mesh2d_edge_y.long_name = (
            "Characteristic y-coordinate of the mesh edge (e.g., midpoint)"
        )

        mesh2d_edge_nodes = dataset.createVariable(
            "mesh2d_edge_nodes", "i4", ("mesh2d_nEdges", "Two")
        )
        mesh2d_edge_nodes.cf_role = "edge_node_connectivity"
        mesh2d_edge_nodes.long_name = "Start and end nodes of mesh edges"
        mesh2d_edge_nodes.start_index = 1

        mesh2d_edge_faces = dataset.createVariable(
            "mesh2d_edge_faces", "f8", ("mesh2d_nEdges", "Two")
        )
        mesh2d_edge_faces.cf_role = "edge_face_connectivity"
        mesh2d_edge_faces.long_name = "Start and end nodes of mesh edges"
        mesh2d_edge_faces.start_index = 1

        mesh2d_face_nodes = dataset.createVariable(
            "mesh2d_face_nodes", "i4", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_nodes.long_name = "Vertex node of mesh face (counterclockwise)"
        mesh2d_face_nodes.start_index = 1

        mesh2d_face_x = dataset.createVariable(
            "mesh2d_face_x", "f8", ("mesh2d_nFaces",)
        )
        mesh2d_face_x.standard_name = "projection_x_coordinate"
        mesh2d_face_x.long_name = "characteristic x-coordinate of the mesh face"
        mesh2d_face_x.start_index = 1

        mesh2d_face_y = dataset.createVariable(
            "mesh2d_face_y", "f8", ("mesh2d_nFaces",)
        )
        mesh2d_face_y.standard_name = "projection_y_coordinate"
        mesh2d_face_y.long_name = "characteristic y-coordinate of the mesh face"
        mesh2d_face_y.start_index = 1

        mesh2d_face_x_bnd = dataset.createVariable(
            "mesh2d_face_x_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_x_bnd.long_name = (
            "x-coordinate bounds of mesh faces (i.e. corner coordinates)"
        )

        mesh2d_face_y_bnd = dataset.createVariable(
            "mesh2d_face_y_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes")
        )
        mesh2d_face_y_bnd.long_name = (
            "y-coordinate bounds of mesh faces (i.e. corner coordinates)"
        )

        mesh2d_node_x.units = "longitude"
        mesh2d_node_y.units = "latitude"
        mesh2d_edge_x.units = "longitude"
        mesh2d_edge_y.units = "latitude"
        mesh2d_face_x.units = "longitude"
        mesh2d_face_y.units = "latitude"
        mesh2d_face_x_bnd.units = "grados"
        mesh2d_face_y_bnd.units = "grados"
        mesh2d_face_x_bnd.standard_name = "longitude"
        mesh2d_face_y_bnd.standard_name = "latitude"
        mesh2d_face_nodes.coordinates = "mesh2d_node_x mesh2d_node_y"

        wgs84 = dataset.createVariable("wgs84", "int32")
        wgs84.setncatts(
            {
                "name": "WGS 84",
                "epsg": np.int32(4326),
                "grid_mapping_name": "latitude_longitude",
                "longitude_of_prime_meridian": 0.0,
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.314245,
                "inverse_flattening": 298.257223563,
                "EPSG_code": "value is equal to EPSG code",
                "proj4_params": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
                "projection_name": "unknown",
                "wkt": 'GEOGCS["WGS 84",\n    DATUM["WGS_1984",\n        SPHEROID["WGS 84",6378137,298.257223563,\n            AUTHORITY["EPSG","7030"]],\n        AUTHORITY["EPSG","6326"]],\n    PRIMEM["Greenwich",0,\n        AUTHORITY["EPSG","8901"]],\n    UNIT["degree",0.0174532925199433,\n        AUTHORITY["EPSG","9122"]],\n    AXIS["Latitude",NORTH],\n    AXIS["Longitude",EAST],\n    AUTHORITY["EPSG","4326"]]',
            }
        )

        mesh2d_node_x[:] = node_x
        mesh2d_node_y[:] = node_y
        mesh2d_node_z[:] = -node_z

        mesh2d_edge_x[:] = edge_x
        mesh2d_edge_y[:] = edge_y
        mesh2d_edge_nodes[:, :] = edge_node

        mesh2d_edge_faces[:] = edge_face
        mesh2d_face_nodes[:] = face_node + 1
        mesh2d_face_x[:] = face_x
        mesh2d_face_y[:] = face_y

        mesh2d_face_x_bnd[:] = face_x_bnd
        mesh2d_face_y_bnd[:] = face_y_bnd

        dataset.institution = "GeoOcean"
        dataset.references = "https://github.com/GeoOcean/BlueMath_tk"
        dataset.source = f"BlueMath tk {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        dataset.history = "Created with OCSmesh"
        dataset.Conventions = "CF-1.8 UGRID-1.0 Deltares-0.10"

        dataset.createDimension("str_dim", 1)
        mesh2d = dataset.createVariable("mesh2d", "i4", ("str_dim",))
        mesh2d.cf_role = "mesh_topology"
        mesh2d.long_name = "Topology data of 2D mesh"
        mesh2d.topology_dimension = 2
        mesh2d.node_coordinates = "mesh2d_node_x mesh2d_node_y"
        mesh2d.node_dimension = "mesh2d_nNodes"
        mesh2d.edge_node_connectivity = "mesh2d_edge_nodes"
        mesh2d.edge_dimension = "mesh2d_nEdges"
        mesh2d.edge_coordinates = "mesh2d_edge_x mesh2d_edge_y"
        mesh2d.face_node_connectivity = "mesh2d_face_nodes"
        mesh2d.face_dimension = "mesh2d_nFaces"
        mesh2d.face_coordinates = "mesh2d_face_x mesh2d_face_y"
        mesh2d.max_face_nodes_dimension = "mesh2d_nMax_face_nodes"
        mesh2d.edge_face_connectivity = "mesh2d_edge_faces"


def calculate_edges(Elmts: np.ndarray) -> np.ndarray:
    """
    Calculates the unique edges from the given triangle elements.

    Parameters
    ----------
    Elmts : np.ndarray
        A 2D array of shape (nelmts, 3) containing the node indices for each triangle element.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_edges, 2) containing the unique edges,
        each represented by a pair of node indices.
    """

    perc = 0
    Links = np.zeros((len(Elmts) * 3, 2), dtype=int)
    tel = 0
    for ii, elmt in enumerate(Elmts):
        if round(100 * (ii / len(Elmts))) != perc:
            perc = round(100 * (ii / len(Elmts)))
        Links[tel] = [elmt[0], elmt[1]]
        tel += 1
        Links[tel] = [elmt[1], elmt[2]]
        tel += 1
        Links[tel] = [elmt[2], elmt[0]]
        tel += 1

    Links_sorted = np.sort(Links, axis=1)
    Links_unique = np.unique(Links_sorted, axis=0)

    return Links_unique


def build_loops(edges):
    """
    Build closed loops from a list of edges.

    Parameters
    ----------
    edges : (N, 2) array
        List of edges defined by pairs of node indices.

    Returns
    -------
    loops : list of lists
        Each sublist contains node indices forming a closed loop.
    """
    edges = edges.tolist()
    loops = []
    while edges:
        start, end = edges.pop(0)
        loop = [start, end]
        closed = False
        while not closed:
            found = False
            for i, (a, b) in enumerate(edges):
                if a == loop[-1]:
                    loop.append(b)
                    edges.pop(i)
                    found = True
                    break
                elif b == loop[-1]:
                    loop.append(a)
                    edges.pop(i)
                    found = True
                    break
            if not found:
                break
            if loop[-1] == loop[0]:
                closed = True
        loops.append(loop)
    return loops


def export_to_grd(
    filename, vert, tria, z, crs, edge_tag, edge_open=None, edge_land=None
):
    """
    Export mesh to ADCIRC .grd format with boundaries.

    Parameters
    ----------
    filename : str
        Path to output .grd file.
    vert : (N, 2) array
        Node coordinates (x, y).
    tria : (M, 3) array
        Triangle connectivity (node indices).
    z : (N,) array
        Node depth/elevation values.
    crs : str
        Coordinate reference system string.
    edge_tag : (K, 3) array
        Edge tags (node1, node2, tag).
    edge_open : (L, 2) array, optional
        Open boundary edges (node indices). If None, extracted from edge_tag.
    edge_land : (P, 2) array, optional
        Land boundary edges (node indices). If None, extracted from edge_tag.
    """

    # --- 1. Extract edges if not provided
    if edge_open is None:
        edge_open = edge_tag[edge_tag[:, 2] == 1, :2].astype(int)
    if edge_land is None:
        edge_land = edge_tag[edge_tag[:, 2] == 2, :2].astype(int)

    # --- 2. Build loops
    open_loops = build_loops(edge_open) if edge_open.size > 0 else []
    land_loops = build_loops(edge_land) if edge_land.size > 0 else []

    # --- 3. Write to file
    with open(filename, "w") as f:
        # En-tête
        f.write(f"{crs}\n")
        f.write(f"{tria.shape[0]} {vert.shape[0]}\n")

        # --- Nodes
        for i, (x, y, zi) in enumerate(zip(vert[:, 0], vert[:, 1], z), start=1):
            if np.isnan(zi):
                f.write(f"{i} {x:.15f} {y:.15f} NAN\n")
            else:
                f.write(f"{i} {x:.15f} {y:.15f} {zi:.15f}\n")
        # --- Triangles
        for i, tri in enumerate(tria, start=1):
            f.write(f"{i} 3 {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

        # --- Open boundaries
        total_open_nodes = sum(len(loop) for loop in open_loops)
        f.write(f"{len(open_loops)} ! total number of open boundaries\n")
        f.write(f"{total_open_nodes} ! total number of open boundary nodes\n")

        for ib, loop in enumerate(open_loops):
            f.write(f"{len(loop)} ! number of nodes for open_boundary_{ib}\n")
            for nid in loop:
                f.write(f"{nid + 1}\n")

        # ---- Land boundaries
        total_land_nodes = sum(len(loop) for loop in land_loops)
        f.write(f"{len(land_loops)}  ! total number of land boundaries\n")
        f.write(f"{total_land_nodes} ! Total number of land boundary nodes\n")

        for i, loop in enumerate(land_loops):
            f.write(f"{len(loop)} 1 ! boundary 1:{i}\n")
            for nid in loop:
                f.write(f"{nid + 1}\n")


def plot_grd(filename, ax=None, show_boundaries=True):
    """
    draw a preview of an ADCIRC .grd mesh file with optional boundaries.

    Parameters
    ----------
    filename : str
        Path to the .grd file.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show_boundaries : bool
        Whether to plot open and land boundaries.
    """

    if ax is None:
        fig, ax = plt.subplots()

    with open(filename, "r") as f:
        lines = f.readlines()

    # --- search header
    for i, line in enumerate(lines):
        if (
            len(line.split()) == 2
            and line.strip().replace(".", "", 1).replace("-", "", 1).isdigit() is False
        ):
            try:
                nelem, nnode = map(int, line.split())
                header_idx = i
                break
            except Exception:
                continue

    # --- Reading nodes
    node_lines = lines[header_idx + 1 : header_idx + 1 + nnode]
    vert = np.zeros((nnode, 3))
    for i, ln in enumerate(node_lines):
        parts = ln.split()
        vert[i, 0] = float(parts[1])  # lon
        vert[i, 1] = float(parts[2])  # lat
        vert[i, 2] = float(parts[3])  # z

    # --- Reading elements
    elem_lines = lines[header_idx + 1 + nnode : header_idx + 1 + nnode + nelem]
    tria = np.zeros((nelem, 3), dtype=int)
    for i, ln in enumerate(elem_lines):
        parts = ln.split()
        tria[i, :] = np.array(parts[2:5], dtype=int) - 1  # indices 0-based

    # --- Reading boundaries
    open_boundaries = []
    land_boundaries = []

    if show_boundaries:
        idx = header_idx + 1 + nnode + nelem
        for j in range(idx, len(lines)):
            line = lines[j].strip()
            if "! total number of open boundaries" in line:
                n_open = int(line.split()[0])
                j += 1
                n_open_nodes = int(lines[j].split()[0])
                j += 1
                for _ in range(n_open):
                    n_nodes = int(lines[j].split()[0])
                    j += 1
                    ids = []
                    for _ in range(n_nodes):
                        ids.append(int(lines[j].strip()) - 1)
                        j += 1
                    open_boundaries.append(ids)
            if "! total number of land boundaries" in line:
                n_land = int(line.split()[0])
                j += 1
                n_land_nodes = int(lines[j].split()[0])
                j += 1
                for _ in range(n_land):
                    parts = lines[j].split()
                    n_nodes = int(parts[0])
                    j += 1
                    ids = []
                    for _ in range(n_nodes):
                        ids.append(int(lines[j].strip()) - 1)
                        j += 1
                    land_boundaries.append(ids)
            # --- break outer loop
            if j >= len(lines):
                break

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))

    facecolors = np.mean(vert[:, 2][tria], axis=1)
    pm = ax.tripcolor(
        vert[:, 0],
        vert[:, 1],
        tria,
        facecolors=facecolors,
        edgecolors="k",
        lw=0.2,
        cmap="summer",
    )
    plt.colorbar(pm, ax=ax, label="Depth/Elevation")


    for i, b in enumerate(open_boundaries):
        ax.plot(
            vert[b, 0],
            vert[b, 1],
            "r-",
            lw=1.2,
            label="Open boundary" if i == 0 else None,
        )
    for i, b in enumerate(land_boundaries):
        ax.plot(
            vert[b, 0],
            vert[b, 1],
            "k-",
            lw=1.0,
            label="Land boundary" if i == 0 else None,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Mesh preview: {filename}")
    ax.legend(loc="best", frameon=True)
