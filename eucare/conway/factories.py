"""Factory functions building :class:`GeometricConwayOperator` instances for named Conway operators."""
from __future__ import annotations

import networkx as nx

from ..conversions import EHEG_from_nx
from .operators import GeometricConwayOperator


def dual_graph() -> GeometricConwayOperator:
    """Construct the Conway dual operator."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2, ve], delete=True)
    G.add_nodes_from([v1, v2], delete=True)
    G.add_nodes_from([ve], join=True)
    G.add_edge(ve, vf)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def kis_graph() -> GeometricConwayOperator:
    """Construct the Conway kis (raising) operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def join_graph() -> GeometricConwayOperator:
    """Construct the Conway join operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2])
    G.add_edge(v2, v1, delete=True)
    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def ambo_graph() -> GeometricConwayOperator:
    """Construct the Conway ambo (rectification) operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, 0)
    v1f = (1 / 2, -1 / 2)
    v2f = (1 / 2, 1 / 2)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v12], delete=True)
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_edges_from([[v12, v1f], [v12, v2f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def goldberg2_graph() -> GeometricConwayOperator:
    """Construct the Goldberg-2 subdivision operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, 0)
    v1f = (1 / 2, -1 / 2)
    v2f = (1 / 2, 1 / 2)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v12], delete=True)
    del G.edges[v1, v12]['delete']
    del G.edges[v12, v2]['delete']
    G.add_nodes_from([vf], delete=True)
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_edges_from([[v12, v1f], [v12, v2f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def truncate_graph(t: float = 1 / 2) -> GeometricConwayOperator:
    """Construct the Conway truncate operator with cut depth t."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12t = (0, -1 + t)
    v1ft = (t / 2, -1 + t / 2)
    v21t = (0, 1 - t)
    v2ft = (t / 2, 1 - t / 2)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
    del G.edges[v12t, v21t]['delete']
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1ft, v2ft], join=True)
    G.add_edges_from([[v12t, v1ft], [v21t, v2ft]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def gyro_graph(g: tuple[float, float] = (1 / 4, -1 / 4)) -> GeometricConwayOperator:
    """Construct the Conway gyro operator with snub point position g."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2, ve], delete=True)
    G.add_edges_from([[g, ve], [g, v1], [g, vf]])
    G.add_node(ve, join=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def starify_graph(t: float = 1/3) -> GeometricConwayOperator:
    """Construct the starify operator with parameter t controlling star point depth."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v0 = (0, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v0], delete='True')
    G.add_nodes_from([v0], join=True)
    G.add_edges_from([[v0, v2f], [v2f, v1f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def alternating_flagstone_graph(t: float = 1/3) -> GeometricConwayOperator:
    """Construct the alternating flagstone operator with parameter t."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v0 = (0, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v0], delete=True)
    del G.edges[v1, v1f]['delete']
    del G.edges[v2, v2f]['delete']
    G.add_edge(v2f, v1, color_key=(1, 0, 0))
    G.add_nodes_from([v0], join=True)
    G.add_nodes_from([vf], delete=True)
    G.add_edges_from([[v0, v2f], [v2f, v1f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def shrink_rotate_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the shrink-rotate Conway operator with parameter *t*.

    The operator creates new faces for each original vertex, shrinks the original faces and creates a ring of quadrilateral faces between the twists; the central polygon is later
    rotated and scaled around the reciprocal-figure center to produce a
    flat-foldable crease pattern (see :mod:`eucare.shrink_rotate`).

    Each twist face is marked with the ``'shrink_rotate'`` attribute so
    downstream code can identify it.
    """
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12t = (0, -1 + t)
    v1ft = (t, -1 + t)
    v21t = (0, 1 - t)
    v2ft = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v12t, v21t], join=True)
    G.add_edges_from([[v12t, v1ft], [v21t, v2ft], [v1ft, v2ft]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    v_lookup[vf].get_outgoing_border().rev.face['shrink_rotate'] = True
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def loft_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the loft operator with edge offset parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v1ft = (t, -1 + t)
    v2ft = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, v2ft, v2])
    G.add_nodes_from([vf], delete=True)
    G.add_edges_from([[v1ft, vf], [vf, v2ft]], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def lace_graph(t: float = 1/2, join: bool = False) -> GeometricConwayOperator:
    """Construct the lace operator with offset t. If join is True, merge the v1-v2 edge."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    vc = (t, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)

    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2], delete=True)
    if not join:
        del G.edges[v1, v2]['delete']

    G.add_edges_from([[vc, v1], [vc, v2], [vc, v1f], [vc, v2f]])
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_nodes_from([vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def expand_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the Conway expand operator with offset parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, -1 + t)
    v21 = (0, 1 - t)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)

    G = nx.Graph()
    nx.add_cycle(G, [v2, v21, v12, v1, v1f, vf, v2f], delete=True)

    G.add_edges_from([[v1f, v12], [v2f, v21], [v1f, v2f]])
    G.add_nodes_from([v12, v21, v1f, v2f], join=True)
    G.add_nodes_from([vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def flagstone_pvitelli_graph(t: float = 1/4) -> GeometricConwayOperator:
    """Construct the Pvitelli flagstone operator with parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    vL1 = (0, -1 + t)
    vL2 = (0, 1 - t)
    vL1f = (t/2, -1 + t/2)
    vL2f = (t/2, 1 - t/2)
    vV1 = (t, -1 + t)
    vV2 = (t, 1 - t)
    vN1 = (t/2, -1 + 3/2*t)
    vN2 = (t/2, 1 - 3/2*t)
    vN1e = (0, -1 + 3/2*t)
    vN2e = (0, 1 - 3/2*t)

    G = nx.Graph()
    nx.add_cycle(G, [v2, vL2, vN2e, vN1e, vL1, v1, vL1f, vV1, vf, vV2, vL2f], delete=True)
    nx.add_cycle(G, [vL2, vN2, vN1, vL1, vV1, vV2])
    G.add_edges_from([[vL1f, vL1], [vL2f, vL2], [vN1e, vN1], [vN2e, vN2], [vN1, vV1], [vN2, vV2]])
    G.add_nodes_from([vL1f, vL2f, vN1e, vN2e], join=True)

    # label the points which will be joined that are closer to v1
    G.add_nodes_from([vL1], label='L')
    G.add_nodes_from([vV1], label='V')
    # label both copies of the inner node
    G.add_nodes_from([vN1], label='N1')
    G.add_nodes_from([vN2], label='N2')

    G.add_nodes_from([v1, v2, vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)

    v_lookup[vf].get_outgoing_border().rev.face['is_central_polygon'] = True

    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def chamfer_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the Conway chamfer operator, derived from loft with the v1-v2 edge deleted."""
    assert t < 1
    result = loft_graph(t)
    e = [e for e in result.v1.outgoing_iter() if e.dest is result.v2][0]
    e.rev.attributes['delete'] = True
    e.attributes['delete'] = True
    return result
