import numpy as np

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["knotty_centrality"]


@not_implemented_for("directed", "multigraph")
def knotty_centrality(G, compact=True):
    """Return the sub-graph with the highest value of knotty centrality

    Attempts to find the sub-graph of G with the highest value for
    knotty-centrality. Carries out a series of exhaustive searches on
    subsets of the nodes ranked by "indirect" betweenness centrality, then
    carries out a phase of hill-climbing to see whether the sub-graph can
    be improved by adding further nodes.

    Written by Erik Ziegler, adapted from original MATLAB code
    by Murray Shanahan and Mark Wildie

    Parameters
    ----------
    G : NetworkX graph
    compact : bool (optional)
       Calculates the compact knotty centrality (default=True)

    Returns
    -------
    nodes: list
        nodes in the sub-graph with the highest knotty centrality
    kc : float
        the highest value of knotty centrality

    Examples
    --------
    >>> G = nx.Graph([(0,1),(0,2),(1,2),(1,3),(1,4),(4,5)])
    >>> nodes, kc = nx.knotty_centrality(G,compact=False)

    Notes
    ------
    The knotty centrality definition and algorithm are found in [1]_.

    References
    ----------
    .. [1] Shanahan M, Wildie M (2012) Knotty-Centrality:
    Finding the Connective Core of a Complex Network.
    PLoS ONE 7(5): e36579. doi:10.1371/journal.pone.0036579

    """
    return _find_knotty_centre(G, compact)


def _find_knotty_centre(G, compact):
    # nodes = the sub-graph found
    # kc = its knotty-centredness

    N = G.number_of_nodes()

    # binarise matrix - all non-zero weights become 1s
    CIJ = nx.to_numpy_array(G, weight=None)

    # Exhastive search phase
    Exh = min(5, N)  # number of nodes for exhaustive search (2^Exh combinations)

    BC_list = list(nx.betweenness_centrality(G).values())

    BC = np.array(BC_list) / sum(BC_list)  # normalise wrt total betweenness centrality

    # Calculate indirect betweenness centrality
    BC2 = BC + np.sum(CIJ * BC, axis=0) + np.sum(CIJ, axis=1).T * BC

    # Ranking the nodes in descending order
    IxBC = np.argsort(BC2)[::-1].tolist()

    nodes = []
    improving = True

    while improving:
        L = len(nodes)
        # nodes_left = IxBC
        nodes_left = [x for x in IxBC if x not in set(nodes)]
        # choices = [nodes_left[i] for i in range(0, min([Exh, len(nodes_left)]))]
        choices = nodes_left[: min(Exh, len(nodes_left))]
        nodes, kc = _best_perm(nodes, choices, G, CIJ, compact, BC)
        improving = len(nodes) > L

    # Hill climbing phase
    nodes_left = [x for x in range(N) if x not in set(nodes)]
    improving = True

    while improving and nodes_left:
        best_kc = 0
        for node in nodes_left:
            nodes2 = np.hstack((nodes, node))
            kc2 = _compute_knotty_centrality(G, CIJ, nodes2, compact, BC)
            if kc2 > best_kc:
                best_kc = kc2
                best_node = node

        if best_kc > kc:
            kc = best_kc
            nodes = np.hstack((nodes, best_node))
            nodes_left.remove(best_node)
        else:
            improving = 0
    return nodes, kc


def _best_perm(given, choices, G, CIJ, compact, BC):
    # Carries out exhaustive search to find a permutation of nodes in
    # "choices" that when added to the nodes in "given" yields the highest
    # value of knotty-centrality

    if choices:
        choices2 = [choices[i] for i in range(0, len(choices))]
        new = choices[0]
        nodes1, kc1 = _best_perm(np.hstack((given, new)), choices2, G, CIJ, compact, BC)
        nodes2, kc2 = _best_perm(given, choices2, G, CIJ, compact, BC)
        if kc1 > kc2:
            nodes = nodes1
            kc = kc1
        else:
            nodes = nodes2
            kc = kc2
    else:
        nodes = given
        kc = _compute_knotty_centrality(G, CIJ, nodes, compact, BC)
    return nodes, kc


def _compute_knotty_centrality(G, CIJ, nodes, compact, BC):
    # Returns knotty-centrality of the subgraph of CIJ comprising only
    # "nodes" and the associated connections

    if len(nodes) < 3:
        kc = 0
    else:
        CIJ = CIJ != 0  # binarise matrix
        CIJ = CIJ.astype(int)  # set to 1/0 instead of True/False
        N = len(CIJ)  # nodes in overall graph
        M = len(nodes)  # nodes in subgraph

        BC_list = [BC[i] for i in nodes]
        BCtot = sum(BC_list)

        p = (float(N) - M) / N  # proportion of nodes not in subgraph

        g = G.subgraph(nodes.tolist())
        RC = nx.density(g)

        if compact:
            kc = p * BCtot * RC
            # compact knotty-centrality
        else:
            kc = BCtot * RC
            # knotty-centrality
    return kc
