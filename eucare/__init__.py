"""Eucare — geometric tilings, Conway operators, and origami crease patterns.

Importing :mod:`eucare` eagerly loads all public submodules so that
``eucare.<submodule>.<name>`` works without further imports (a convenient
pattern for interactive/notebook use).
"""

# Core data structures and geometry
import eucare.base
import eucare.half
import eucare.geometries

# Tilings
import eucare.prototiles
import eucare.instructions
import eucare.example_tilesets
import eucare.example_graphs

# Topology and origami pipeline
import eucare.conway
import eucare.reciprocal_figures
import eucare.cutting
import eucare.overlap

# Layout, classification, coloring, search
import eucare.layout
import eucare.classifiers
import eucare.colorization
import eucare.search_trees

# I/O and rendering
import eucare.io
import eucare.svg
import eucare.plotting
