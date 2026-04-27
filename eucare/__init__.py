"""Eucare — geometric tilings, Conway operators, and origami crease patterns.

Importing :mod:`eucare` eagerly loads all public submodules so that
``eucare.<submodule>.<name>`` works without further imports (a convenient
pattern for interactive/notebook use).
"""

# Core data structures and geometry
import eucare.base
import eucare.classifiers
import eucare.colorization

# Topology and origami pipeline
import eucare.conway
import eucare.cutting
import eucare.example_graphs
import eucare.example_tilesets
import eucare.geometries
import eucare.half
import eucare.instructions

# I/O and rendering
import eucare.io

# Layout, classification, coloring, search
import eucare.layout
import eucare.overlap
import eucare.plotting

# Tilings
import eucare.prototiles
import eucare.reciprocal_figures
import eucare.search_trees
import eucare.svg
