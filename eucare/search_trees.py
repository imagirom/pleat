"""Search trees for half-edge graphs."""

def bfs_tree(start, neighbor_iter):
    """breadth first search tree starting from a node or set"""
    boundary = start if isinstance(start, set) else {start}
    parsed = boundary.copy()
    edges = []
    while boundary:
        new_boundary = set()
        for orig in boundary:
            for dest in neighbor_iter(orig):
                if dest not in parsed:
                    parsed.add(dest)
                    new_boundary.add(dest)
                    edges.append((orig, dest))
        boundary = new_boundary
    return edges


def face_bfs_tree(start):
    return bfs_tree(start, lambda f: (f2 for f2 in f.face_iter() if f2 is not None))


def vertex_bfs_tree(start):
    return bfs_tree(start, lambda v: v.vertex_iter())
