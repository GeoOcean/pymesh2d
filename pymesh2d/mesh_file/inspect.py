def inspect(mesh, base=None, item=None):
    """
    INSPECT: Helper routine to safely query a MESH structure.

    Parameters:
        mesh (dict): MESH structure to inspect.
        base (str, optional): Base field to check.
        item (str, optional): Subfield to check.

    Returns:
        bool: True if the specified field(s) exist and are not empty, False otherwise.

    References
    ----------
    Translation of the MESH2D function `CERTIFY`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    if not isinstance(mesh, dict):
        raise ValueError("inspect: MESH must be a valid dictionary.")

    if base is not None and not isinstance(base, str):
        raise ValueError("inspect: BASE must be a valid string.")

    if item is not None and not isinstance(item, str):
        raise ValueError("inspect: ITEM must be a valid string.")

    # - default ITEM kinds given BASE types
    if item is None:
        base_to_item = {
            "point": "coord",
            "edge2": "index",
            "tria3": "index",
            "quad4": "index",
            "tria4": "index",
            "hexa8": "index",
            "wedg6": "index",
            "pyra5": "index",
            "bound": "index",
        }
        item = base_to_item.get(base.lower(), None) if base else None

    # -- check whether MESH.BASE exists
    if item is None:
        return base in mesh and mesh[base] is not None

    # -- check whether MESH.BASE.ITEM exists
    return (
        base in mesh
        and isinstance(mesh[base], dict)
        and item in mesh[base]
        and mesh[base][item] is not None
    )
