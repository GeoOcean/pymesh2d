import numpy as np

from .inspect import inspect


def certify(mesh):
    """
    CERTIFY: Error checking for JIGSAW mesh objects.

    Parameters:
        mesh (dict): JIGSAW mesh object.

    Returns:
        flag (int): 1 if the mesh passes all checks, -1 otherwise.

    References
    ----------
    Translation of the MESH2D function `CERTIFY`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    flag = -1

    if not isinstance(mesh, dict):
        raise ValueError("certify: Incorrect input class.")

    np_points = 0

    # Check 'point' field
    if inspect(mesh, "point"):
        if mesh["point"]["coord"] is not None:
            if isinstance(mesh["point"]["coord"], np.ndarray):
                # ----------------------------------------- check MESH coords
                np_points = mesh["point"]["coord"].shape[0]

                if mesh["point"]["coord"].ndim != 2:
                    raise ValueError("certify: Invalid POINT.COORD dimensions.")
                if mesh["point"]["coord"].shape[1] < 3:
                    raise ValueError("certify: Invalid POINT.COORD dimensions.")
                if np.any(np.isinf(mesh["point"]["coord"])):
                    raise ValueError("certify: Invalid POINT.COORD values.")
                if np.any(np.isnan(mesh["point"]["coord"])):
                    raise ValueError("certify: Invalid POINT.COORD values.")

                if "mshID" in mesh:
                    if mesh["mshID"].lower() in ["euclidean-grid", "ellipsoid-grid"]:
                        raise ValueError("certify: Incompatible msh-ID flag.")

            elif isinstance(mesh["point"]["coord"], list):
                # ----------------------------------------- check GRID coords
                if not all(
                    isinstance(coord, np.ndarray) for coord in mesh["point"]["coord"]
                ):
                    raise ValueError("certify: Invalid POINT.COORD dimensions.")
                for coord in mesh["point"]["coord"]:
                    if np.any(np.isinf(coord)) or np.any(np.isnan(coord)):
                        raise ValueError("certify: Invalid POINT.COORD values.")

                if "mshID" in mesh:
                    if mesh["mshID"].lower() in ["euclidean-mesh", "ellipsoid-mesh"]:
                        raise ValueError("certify: Incompatible msh-ID flag.")
            else:
                # ----------------------------------------- wrong POINT class
                raise ValueError("certify: Invalid POINT.COORD type.")

    if inspect(mesh, "radii"):
        # ----------------------------------------- check RADII value
        if mesh["radii"] is not None:
            if not isinstance(mesh["radii"], np.ndarray):
                raise ValueError("certify: Invalid RADII class.")
            if mesh["radii"].ndim != 1 or len(mesh["radii"]) not in [1, 3]:
                raise ValueError("certify: Invalid RADII dimensions.")
            if np.any(np.isinf(mesh["radii"])) or np.any(np.isnan(mesh["radii"])):
                raise ValueError("certify: Invalid RADII entries.")

    if inspect(mesh, "value"):
        # ----------------------------------------- check VALUE value
        if mesh["value"] is not None:
            if isinstance(mesh["point"]["coord"], np.ndarray):
                # ----------------------------------------- for MESH obj kind
                if mesh["value"].ndim != 2:
                    raise ValueError("certify: Invalid VALUE dimensions.")
                if mesh["value"].shape[0] != np_points:
                    raise ValueError("certify: Invalid VALUE dimensions.")
                if np.any(np.isinf(mesh["value"])) or np.any(np.isnan(mesh["value"])):
                    raise ValueError("certify: Invalid VALUE entries.")
            elif isinstance(mesh["point"]["coord"], list):
                # ----------------------------------------- for GRID obj kind
                if len(mesh["point"]["coord"]) != mesh["value"].ndim:
                    raise ValueError("certify: Invalid VALUE dimensions.")
                if len(mesh["point"]["coord"]) == 2:
                    if (
                        np.prod([len(c) for c in mesh["point"]["coord"]])
                        != mesh["value"].size
                    ):
                        raise ValueError("certify: Invalid VALUE dimensions.")
                elif len(mesh["point"]["coord"]) == 3:
                    if (
                        np.prod([len(c) for c in mesh["point"]["coord"]])
                        != mesh["value"].size
                    ):
                        raise ValueError("certify: Invalid VALUE dimensions.")
                if np.any(np.isinf(mesh["value"])) or np.any(np.isnan(mesh["value"])):
                    raise ValueError("certify: Invalid VALUE entries.")
            else:
                # ---------------------------------------- wrong VALUE class
                raise ValueError("certify: Invalid VALUE class.")

    for field, expected_size in [
        ("edge2", 3),
        ("tria3", 4),
        ("quad4", 5),
        ("tria4", 5),
        ("hexa8", 9),
        ("wedg6", 7),
        ("pyra5", 6),
        ("bound", 3),
    ]:
        if inspect(mesh, field):
            if mesh[field]["index"] is not None:
                if not isinstance(mesh[field]["index"], np.ndarray):
                    raise ValueError(f"certify: Invalid {field.upper()}.INDEX type.")
                if (
                    mesh[field]["index"].ndim != 2
                    or mesh[field]["index"].shape[1] != expected_size
                ):
                    raise ValueError(
                        f"certify: Invalid {field.upper()}.INDEX dimensions."
                    )
                if np.any(np.isinf(mesh[field]["index"])) or np.any(
                    np.isnan(mesh[field]["index"])
                ):
                    raise ValueError(
                        f"certify: Invalid {field.upper()}.INDEX indexing."
                    )
                if (
                    np.min(mesh[field]["index"][:, :-1]) < 1
                    or np.max(mesh[field]["index"][:, :-1]) > np_points
                ):
                    raise ValueError(
                        f"certify: Invalid {field.upper()}.INDEX indexing."
                    )

    # ---------------------------------------- ok if we get here
    flag = 1
    return flag
