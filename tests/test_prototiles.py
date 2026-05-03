"""Tests for prototile construction."""

from __future__ import annotations

import numpy as np
import pytest

from eucare.prototiles import RegularEuclideanTile, RegularProtoTile


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_regular_euclidean_tile_make_graph(n):
    tile = RegularEuclideanTile(n)
    G, edge_dict = tile.make_graph(add_positions=True)
    G.check_consistency()
    assert len(G.faces) == 1
    f = next(iter(G.faces))
    assert f.order() == n
    assert len(edge_dict) == n
    # vertices have positions
    for v in G.vertices:
        assert "pos" in v.attributes
        assert np.asarray(v["pos"]).shape == (2,)


def test_regular_euclidean_tile_attach_instruction_returns_callable():
    tile = RegularEuclideanTile(4)
    inst = tile.attach_instruction()
    assert callable(inst)


def test_regular_euclidean_tile_invalid_label_rejected():
    tile = RegularEuclideanTile(4)
    with pytest.raises(AssertionError):
        tile.attach_instruction(label="not-an-edge-label")


def test_regular_proto_tile_chooses_geometry_by_curvature():
    # interior angle of regular triangle: pi/3 -> Euclidean fails, sphere wins
    sph = RegularProtoTile(3, np.pi / 3 + 0.1, 1.0)
    eucl = RegularProtoTile(3, np.pi / 3, 1.0)
    hyp = RegularProtoTile(3, np.pi / 3 - 0.1, 1.0)
    # Just check that geometry attribute is set distinctly:
    assert sph.geometry is not eucl.geometry
    assert hyp.geometry is not eucl.geometry
