#!/usr/bin/env python
from nose.tools import *
import networkx as nx
from networkx.algorithms import isomorphism as iso


class TestIsomorph:

    @classmethod
    def setup_class(cls):
        cls.G1 = nx.Graph()
        cls.G2 = nx.Graph()
        cls.G3 = nx.Graph()
        cls.G4 = nx.Graph()
        cls.G1.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 3]])
        cls.G2.add_edges_from([[10, 20], [20, 30], [10, 30], [10, 50]])
        cls.G3.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 5]])
        cls.G4.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 4]])

    def test_could_be_isomorphic(self):
        assert iso.could_be_isomorphic(self.G1, self.G2)
        assert iso.could_be_isomorphic(self.G1, self.G3)
        assert not iso.could_be_isomorphic(self.G1, self.G4)
        assert iso.could_be_isomorphic(self.G3, self.G2)

    def test_fast_could_be_isomorphic(self):
        assert iso.fast_could_be_isomorphic(self.G3, self.G2)

    def test_faster_could_be_isomorphic(self):
        assert iso.faster_could_be_isomorphic(self.G3, self.G2)

    def test_is_isomorphic(self):
        assert iso.is_isomorphic(self.G1, self.G2)
        assert not iso.is_isomorphic(self.G1, self.G4)
