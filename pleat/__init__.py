"""Pleat — geometric tilings, Conway operators, and origami crease patterns.

Importing :mod:`pleat` eagerly loads all public submodules so that
``pleat.<submodule>.<name>`` works without further imports (a convenient
pattern for interactive/notebook use).
"""

# Core data structures and geometry
import pleat.alternating_flagstones
import pleat.base
import pleat.circle_packing
import pleat.classifiers
import pleat.colorization

# Topology and origami pipeline
import pleat.conway
import pleat.cutting
import pleat.example_graphs
import pleat.example_tilesets
import pleat.flat_foldable
import pleat.geometries
import pleat.gjh
import pleat.half
import pleat.instructions
import pleat.intersecting_cylinders

# I/O and rendering
import pleat.io

# Layout, classification, coloring, search
import pleat.layout
import pleat.overlap
import pleat.plotting
import pleat.prototiles
import pleat.search_trees
import pleat.shrink_rotate
import pleat.svg
import pleat.tileset_spec
