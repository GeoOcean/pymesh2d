<p align="center">
  <img src="assets/mesh_geocean.webp" alt="pymesh2D logo" width="120">
</p>

## `pymesh2D: Delaunay-based mesh generation in Python`

`pymesh2D` is a `Python`-based unstructured mesh-generator for two-dimensional polygonal geometries, providing a range of relatively simple, yet effective two-dimensional meshing algorithms. `pymesh2D` includes variations on the "classical" Delaunay refinement technique, a new "Frontal"-Delaunay refinement scheme, a non-linear mesh optimisation method, and auxiliary mesh and geometry pre- and post-processing facilities. 

### This code is a translation of <a href="https://github.com/dengwirda/mesh2d">`MESH2D`</a>, a `MATLAB` / `OCTAVE`-based tool developed by Darren Engwirda.

Algorithms implemented in `pymesh2D` are "probably-good" - ensuring convergence, geometrical and topological correctness, and providing guarantees on algorithm termination and worst-case element quality bounds. Support for user-defined "mesh-spacing" functions and "multi-part" geometry definitions is also provided, allowing `pymesh2D` to handle a wide range of complex domain types and user-defined constraints. `pymesh2D` typically generates very high-quality output, appropriate for a variety of finite-volume/element type applications.

`pymesh2D` is a simplified version of my <a href="https://github.com/dengwirda/jigsaw-matlab/">`JIGSAW`</a> mesh-generation algorithm (a `C++` code). `pymesh2D` aims to provide a straightforward `Python` implementation of these Delaunay-based triangulation and mesh optimisation techniques. 

### `Code Structure`

`pymesh2D` is a pure `Python` package, consisting of a core library + associated utilities:

    pymesh2D::
    ├── pymesh2d              -- core pymesh2D library functions. See refine, smooth, tridemo etc.
    ├── pymesh2d/aabb_tree    -- support for fast spatial indexing, via tree-based data-structures.
    ├── pymesh2d/geom_util    -- geometry processing, repair, etc.
    ├── pymesh2d/geomesh_util -- mesh gestion, export interpolation, etc.
    ├── pymesh2d/hfun_util    -- mesh-spacing definitions, limiters, etc.
    ├── pymesh2d/hjac_util    -- solver for Hamilton-Jacobi eqn's.
    ├── pymesh2d/mesh_ball    -- circumscribing balls, orthogonal balls etc.
    ├── pymesh2d/mesh_cost    -- mesh cost/quality functions, etc.
    ├── pymesh2d/mesh_file    -- mesh i/o via ASCII serialisation.
    ├── pymesh2d/mesh_util    -- meshing/triangulation utility functions.
    ├── pymesh2d/poly_data    -- polygon definitions for demo problems, etc.
    └── pymesh2d/poly_test    -- fast inclusion test for polygons.


### `Quickstart`

### Installation

You can install **`pymesh2d`** directly from PyPI:

```bash
pip install pymesh2d
```

If you want to install it in *developer mode* (for local development and contribution):

```bash
pip install -e .
```

### Examples

    python -m pymesh2d.tridemo  0; % a very simple example to get everything started.
    python -m pymesh2d.tridemo  1; % investigate the impact of the "radius-edge" threshold.
    python -m pymesh2d.tridemo  2; % Frontal-Delaunay vs. Delaunay-refinement refinement.
    python -m pymesh2d.tridemo  3; % explore impact of user-defined mesh-size constraints.
    python -m pymesh2d.tridemo  4; % explore impact of "hill-climbing" mesh optimisations.
    python -m pymesh2d.tridemo  5; % assemble triangulations for "multi-part" geometries.
    python -m pymesh2d.tridemo  6; % assemble triangulations with "internal" constraints.
    python -m pymesh2d.tridemo  7; % investigate the use of "quadtree"-type refinement.
    python -m pymesh2d.tridemo  8; % explore use of custom, user-defined mesh-size functions.
    python -m pymesh2d.tridemo  9; % larger-scale problem, mesh refinement + optimisation. 
    python -m pymesh2d.tridemo 10; % medium-scale problem, mesh refinement + optimisation. 

More examples available in [`BlueMath`](https://github.com/GeoOcean/BlueMath/tree/main/toolkit/mesh).

### `References`

If you make use of `pymesh2D` please include a reference to the following! `pymesh2D` is a translation of `MESH2D`, designed to provide a simple and easy-to-understand implementation of Delaunay-based mesh-generation techniques. For a much more advanced and fully three-dimensional mesh-generation library, see the <a href="https://github.com/dengwirda/jigsaw-matlab/">`JIGSAW`</a> package. `MESH2D` makes use of the <a href="https://github.com/dengwirda/aabb-tree">`AABBTREE`</a> and <a href="https://github.com/dengwirda/find-tria">`FINDTRIA`</a> packages to compute efficient spatial queries and intersection tests.

`[1]` - Darren Engwirda, <a href="http://hdl.handle.net/2123/13148">Locally-optimal Delaunay-refinement and optimisation-based mesh generation</a>, Ph.D. Thesis, School of Mathematics and Statistics, The University of Sydney, September 2014.

`[2]` - Darren Engwirda, Unstructured mesh methods for the Navier-Stokes equations, Honours Thesis, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney, November 2005.
