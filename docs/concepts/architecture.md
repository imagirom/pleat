# Architecture

## Core data structure: Half-edge graph

Eucare is built around the **half-edge data structure** (also known as DCEL — doubly-connected edge list). Every undirected edge is stored as a pair of directed half-edges linked via `rev`. Navigation around a face uses `nex`/`pre`; navigation around a vertex uses `v.outgoing_iter()`.

```
HalfEdgeGraph          Topology only (vertices, halfedges, faces as sets)
  └─ InAngleHEG        + interior angles and edge lengths
      └─ GeometricHEG  + pluggable geometry backend
          └─ EuclideanPositionHEG  + 2D vertex positions
```

All graph elements (`HalfEdge`, `Vertex`, `Face`) inherit from `AttributeObject`, providing dict-like attribute storage via `obj['key'] = value`.

## Tiling construction pipeline

1. **ProtoTiles** define tile shapes via angles, edge lengths, and gluing labels
2. **Tile sets** group prototiles into Archimedean or other tilings
3. **Instructions** on border edges describe how to attach new tiles
4. **Growth** iterates over border vertices, executing their instructions

```python
tiles = platonic(6)          # hexagonal prototile set
G = from_tiles(tiles, rings=5)  # grow 5 rings
```

## Conway operators

Topological operators that transform tilings: `dual`, `ambo`, `truncate`, `kis`, `join`, `gyro`, `starify`, and more. Each operator defines a fundamental domain as a small half-edge graph that gets substituted into each face triangle.

```python
G2 = ambo_graph()(G, delete_on_border=True)
```

## Geometry backends

Three pluggable backends in `eucare.geometries`:

| Backend | Model | Use case |
|---------|-------|----------|
| `EuclideanGeometry` | Flat plane | Standard tilings |
| `PoincareDiskModel` | Hyperbolic disk | Hyperbolic tilings ({7,3}, {5,4}, ...) |
| `SphereModel` | Unit sphere (stereographic) | Platonic solids, spherical tilings |

## Origami pipelines

Multiple pipelines for turning tilings into origami tesselations are implemented, see the example notebooks for details.
