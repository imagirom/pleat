import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from skimage import io, color
from skimage.feature import canny
from skimage.transform import downscale_local_mean
from scipy import ndimage as ndi
from fancy.plotting import plot_image
from fancy.colors import colorize_segmentation
from skimage.morphology import skeletonize
import mahotas as mh
from mahotas import polygon
import os
import eucare as ec
from .overlap import group_closeby



# partially adapted from
# https://gist.githubusercontent.com/jeanpat/5712699/raw/9b6139432a1715c31b36bd740841bbb8599b18d8/SkeletonToGraph.ipynb

def branchedPoints(skel):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mh.morph.hitmiss(skel,branch1)
    br2=mh.morph.hitmiss(skel,branch2)
    br3=mh.morph.hitmiss(skel,branch3)
    br4=mh.morph.hitmiss(skel,branch4)
    br5=mh.morph.hitmiss(skel,branch5)
    br6=mh.morph.hitmiss(skel,branch6)
    br7=mh.morph.hitmiss(skel,branch7)
    br8=mh.morph.hitmiss(skel,branch8)
    br9=mh.morph.hitmiss(skel,branch9)
    return br1+br2+br3+br4+br5+br6+br7+br8+br9

def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])

    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep

def pruning(skeleton, size=None):
    '''remove iteratively end points "size"
       times from the skeleton
    '''
    i = 0
    while True:
        i += 1
        if size is not None and i > size:
            break
        endpoints = endPoints(skeleton)
        if not endpoints.sum():
            break
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton


def image_to_graph(image, threshold=None, closing_iterations=3, edge_length_cutoff=None):
    if isinstance(image, str):
        rgb = io.imread(image)
    else:
        assert isinstance(image, np.ndarray)
        assert image.shape[-1] == 3
        rbg = image

    # downsample the image until one of the sides is smaller than 1000px
    while min(rgb.shape[:2]) >= 1000:
        rgb = downscale_local_mean(rgb, (2, 2, 1))

    plot_image(rgb/255, figheight=5, title='original image')
    plt.show()

    grayscale = color.rgb2gray(rgb)

    plot_image(grayscale, cmap='gray', figheight=5, colorbar=True, title='grayscale')
    plt.show()
    plt.hist(grayscale.flatten(), bins=100);
    plt.title('grayscale histogram. use this to select cutoff')
    plt.show()
    if threshold is None:
        raise NotImplementedError('Automatic thresholding is not implemented.')

    fg = grayscale < threshold

    plot_image(fg, figheight=5, title='foreground')

    # close boundaries
    closed = ndi.binary_closing(fg, iterations=closing_iterations)
    plot_image(closed, figheight=5, title='closed')

    skeleton = mh.thin(closed)
    skeleton = pruning(skeleton)

    branching_points = branchedPoints(skeleton) > 0

    branch_point_labels, n_branch_points = ndi.label(branching_points)
    edge_labels, n_edges = ndi.label(skeleton ^ branching_points)

    plot_image(colorize_segmentation(ndi.grey_dilation(edge_labels, size=3).astype(np.int32), ignore_label=0),
               figheight=5, title='edges')
    plt.show()

    # construct the graph: for every branching point, find the edges connected to it
    edge_dict = {}
    for bp in range(1, n_branch_points+1):
        adjacent_edges = np.unique(edge_labels[ndi.binary_dilation(branch_point_labels == bp)])
        adjacent_edges = adjacent_edges[adjacent_edges > 0]
        edge_dict[bp] = set(adjacent_edges)

    edges = [(bp1, bp2)
             for (bp1, edge_ids1) in edge_dict.items()
             for (bp2, edge_ids2) in edge_dict.items()
             if bp2 > bp1 and edge_ids1.intersection(edge_ids2)
             ]

    # compute branching point positions
    xy = np.moveaxis(np.mgrid[:skeleton.shape[0], :skeleton.shape[1]], 0, -1)[:, :, ::-1]
    pos_dict = {}
    for bp in range(1, n_branch_points+1):
        pos_dict[bp] = xy[branch_point_labels == bp].mean(0)

    lengths = [np.linalg.norm(pos_dict[i] - pos_dict[j]) for (i, j) in edges]
    plt.hist(lengths, bins=20)
    plt.title('edge lengths')
    plt.show()

    if edge_length_cutoff is None:
        raise NotImplementedError('Automatic edge length cutoff not implemented.')

    bp_positions = np.stack(list(pos_dict.values()))

    new_bp_labels = group_closeby(bp_positions, edge_length_cutoff/2)
    new_pos_dict = {int(i): np.mean([pos_dict[int(j)] for j in np.argwhere(new_bp_labels==i)+1], axis=0)
                    for i in np.unique(new_bp_labels)}
    new_edges = set((new_bp_labels[i-1], new_bp_labels[j-1])
                    for i, j in edges if new_bp_labels[i-1] != new_bp_labels[j-1])

    graph = nx.Graph()
    graph.add_edges_from(new_edges)

    # convert to eucare graph
    G = ec.conversions.EHEG_from_nx(graph, new_pos_dict)

    G.recompute_lengths_and_angles()
    G.show()
    return G
