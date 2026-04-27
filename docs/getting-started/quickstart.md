# Quick Start

## Build a tiling

```python
from eucare.example_tilesets import platonic
from eucare.example_graphs import from_tiles

# Create a square tiling with 5 rings of tiles
tiles = platonic(4)
G = from_tiles(tiles, rings=5)

# Visualize
G.show()
```

## Apply a Conway operator

```python
from eucare.conway import ambo_graph

G2 = ambo_graph()(G, delete_on_border=True)
G2.recompute_lengths_and_angles()
G2.show()
```

## Explore different geometries

```python
from eucare.example_tilesets import curved_platonic

# Spherical: icosahedron
tiles = curved_platonic(3, 5)
G = from_tiles(tiles, rings=10)

# Hyperbolic: {7,3} tiling
tiles = curved_platonic(7, 3)
G = from_tiles(tiles, rings=3)
```

## Shrink-rotate origami

```python
from eucare.reciprocal_figures import make_SRG, assign_this_way_by_face_z_order
from eucare.search_trees import face_bfs_tree
from eucare.overlap import fold_complete
import numpy as np

G = from_tiles(platonic(4), rings=3)

# Assign face z-order for crease direction
central = min(G.faces, key=lambda f: np.linalg.norm(f.midpoint()))
central['z_order'] = 0
for orig, dest in face_bfs_tree(central):
    dest['z_order'] = orig['z_order'] + 1
assign_this_way_by_face_z_order(G)

# Build shrink-rotate crease pattern
SRG = make_SRG(G)

# Fold (requires PuLP solver)
results = fold_complete(SRG, overlap_eps=1e-8)
```
