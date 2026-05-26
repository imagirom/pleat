from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

pytest.importorskip("mahotas", reason="image_to_graph requires the [image] extra")
from eucare.image_to_graph import image_to_graph


def test_image_to_graph_sample_image_merges_branch_points_without_type_error(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    image_path = Path(__file__).resolve().parents[1] / "docs" / "notebooks" / "images" / "test_graph_image.jpg"

    graph = image_to_graph(
        str(image_path),
        threshold=75,
        closing_iterations=3,
        edge_length_cutoff=40,
    )

    assert len(graph.vertices) > 0
    assert len(graph.halfedges_representing_edges()) > 0
