# -*- coding: utf-8 -*-\
'''
    IO functions for Gmsh .msh files
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.

    Copyright (C) 2013-2020 Andre Antunes, Guilherme B Saturnino, Kristoffer H Madsen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import division
from __future__ import print_function
import os
import struct
import copy
import warnings
import gc
import hashlib


import numpy as np
import scipy.spatial
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import scipy.interpolate
import nibabel


from . import cython_msh



class InvalidMeshError(ValueError):
    pass


__all__ = [
    'read_msh',
    'write_msh',
    'Msh',
    'Nodes',
    'Elements',
    'ElementData',
    'NodeData'
]
# =============================================================================
# CLASSES
# =============================================================================

class Nodes:
    """class to handle the node information:

    Parameters
    -----------------------
    node_coord (optional): (Nx3) ndarray
        Coordinates of the nodes

    Attributes
    ----------------------
    node_coord: (Nx3) ndarray
        Coordinates of the nodes

    nr: property
        Number of nodes

    Examples
    -----------------------------------
     >>> nodes = Nodes(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
     array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
     >>> nodes.node_number
     array([1, 2, 3])
     >>> nodes[1]
     array([1, 0, 0])
     >>> nodes[array(True, False, True)]
     array([[1, 0, 0], [0, 0, 1]])
    """

    def __init__(self, node_coord=None):
        # gmsh fields
        self.node_coord = np.array([], dtype='float64')
        if node_coord is not None:
            self.node_coord = node_coord

    @property
    def nr(self):
        ''' Number of nodes '''
        return self.node_coord.shape[0]

    @property
    def node_number(self):
        ''' Node numbers (1, ..., nr) '''
        return np.array(range(1, self.nr + 1), dtype='int32')

    def find_closest_node(self, querry_points, return_index=False):
        """ Finds the closest node to each point in p

        Parameters
        --------------------------------
        querry_points: (Nx3) ndarray
            List of points (x,y,z) to which the closes node in the mesh should be found

        return_index: (optional) bool
        Whether to return the index of the closes nodes, default: False

        Returns
        -------------------------------
        coords: Nx3 array of floats
            coordinates of the closest points

        indexes: Nx1 array of ints
            Indices of the nodes in the mesh

        --------------------------------------
        The indices are in the mesh listing, that starts at one!
       """
        if len(self.node_coord) == 0:
            raise InvalidMeshError('Mesh has no nodes defined')

        kd_tree = scipy.spatial.cKDTree(self.node_coord)
        _, indexes = kd_tree.query(querry_points)
        coords = self.node_coord[indexes, :]

        if return_index:
            return (coords, indexes + 1)

        else:
            return coords

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __getitem__(self, index):
        return _getitem_one_indexed(self.node_coord, index)

    def __str__(self):
        return str(self.node_coord)


class Elements:
    """ Mesh elements.

    Can only handle triangles and tetrahedra!

    Parameters
    --------------------------
    triangles (optional): (Nx3) ndarray
        List of nodes composing each triangle
    tetrahedra(optional): (Nx3) ndarray
        List of nodes composing each tetrahedra


    Attributes
    ----------------------------------
    elm_number: (Nx1) ndarray
          element ID (u from 1 till nr)
    elm_type: (Nx1) ndarray
        elm-type (2=triangle, 4=tetrahedron, etc)
    tag1: (Nx1) ndarray
        first tag for each element
    tag2: (Nx1) ndarray
        second tag for each elenent
    node_number_list: (Nx4) ndarray
        4xnumber_of_element matrix of the nodes that constitute the element.
        For the triangles, the fourth element = -1
    nr: int
        Number or elemets


    Notes
    -------------------------
    Node and element count starts at 1!

    """

    def __init__(self, triangles=None, tetrahedra=None, points=None, lines=None):
        # gmsh fields
        self.elm_type = np.zeros(0, 'int8')
        self.tag1 = np.zeros(0, dtype='int16')
        self.tag2 = np.zeros(0, dtype='int16')
        self.node_number_list = np.zeros((0, 4), dtype='int32')

        if points is not None:
            assert len(points.shape) == 1
            assert np.all(points > 0), "Node count should start at 1"
            self.node_number_list = np.zeros(
                (points.shape[0], 4), dtype='int32')
            self.node_number_list[:, 0] = points.astype('int32')
            self.node_number_list[:, 1:] = -1
            self.elm_type = np.ones((self.nr,), dtype='int32') * 15

        if lines is not None:
            assert lines.shape[1] == 2
            assert np.all(lines > 0), "Node count should start at 1"
            self.node_number_list = np.zeros(
                (lines.shape[0], 4), dtype='int32')
            self.node_number_list[:, :2] = lines.astype('int32')
            self.node_number_list[:, 2:] = -1
            self.elm_type = np.ones((self.nr,), dtype='int32') * 1

        if triangles is not None:
            assert triangles.shape[1] == 3
            assert np.all(triangles > 0), "Node count should start at 1"
            self.node_number_list = np.zeros(
                (triangles.shape[0], 4), dtype='int32')
            self.node_number_list[:, :3] = triangles.astype('int32')
            self.node_number_list[:, 3] = -1
            self.elm_type = np.ones((self.nr,), dtype='int32') * 2

        if tetrahedra is not None:
            assert tetrahedra.shape[1] == 4
            assert np.all(tetrahedra > 0), "Node count should start at 1"
            if len(self.node_number_list) == 0:
                self.node_number_list = tetrahedra.astype('int32')
                self.elm_type = np.ones((self.nr,), dtype='int32') * 4
            else:
                self.node_number_list = np.vstack(
                    (self.node_number_list, tetrahedra.astype('int32')))
                self.elm_type = np.append(
                    self.elm_type, np.ones(len(tetrahedra), dtype='int32') * 4)

        if len(self.node_number_list) > 0:
            self.tag1 = np.ones((self.nr,), dtype='int32')
            self.tag2 = np.ones((self.nr,), dtype='int32')

    @property
    def nr(self):
        ''' Number of elements '''
        return self.node_number_list.shape[0]

    @property
    def triangles(self):
        ''' Triangle element numbers '''
        return self.elm_number[self.elm_type == 2]

    @property
    def tetrahedra(self):
        ''' Tetrahedra element numbers '''
        return self.elm_number[self.elm_type == 4]

    @property
    def elm_number(self):
        ''' Element numbers (1, ..., nr) '''
        return np.arange(1, self.nr + 1, dtype='int32')

    def find_all_elements_with_node(self, node_nr):
        """ Finds all elements that have a given node

        Parameters
        -----------------
        node_nr: int
            number of node

        Returns
        ---------------
        elm_nr: np.ndarray
            array with indices of element numbers

        """
        elm_with_node = np.any(
            np.isin(self.node_number_list, node_nr),
            axis=1)

        return self.elm_number[elm_with_node]

    def get_faces(self, tetrahedra_indexes=None):
        ''' Creates a list of nodes in each face and a list of faces in each tetrahedra

        Parameters
        ----------------
        tetrahedra_indexes: np.ndarray
            Indices of the tetrehedra where the faces are to be determined (default: all
            tetrahedra)

        Returns
        -------------
        faces: np.ndarray
            List of nodes in faces, in arbitrary order
        th_faces: np.ndarray
            List of faces in each tetrahedra, starts at 0, order=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        face_adjacency_list: np.ndarray
            List of tetrahedron adjacent to each face, filled with -1 if a face is in a
            single tetrahedron. Not in the normal element ordering, but only in the order
            the tetrahedra are presented
        '''
        if tetrahedra_indexes is None:
            tetrahedra_indexes = self.tetrahedra
        th = self[tetrahedra_indexes]
        faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        faces = faces.reshape(-1, 3)
        unique, idx, inv, count = np.unique(
            np.sort(faces, axis=1),
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)

        if np.any(count > 2):
            raise InvalidMeshError(
                'Found a face with more than 2 adjacent tetrahedra!')
        face_adjacency_list = np.argsort(inv)
        # I extend the "inv" and link all the outside faces with an artificial tetrahedra
        out_faces = np.where(count == 1)[0]
        inv_extended = np.hstack((inv, out_faces))
        # The argsort operation will give me the pairs I want
        face_adjacency_list = np.argsort(inv_extended).reshape(-1, 2) // 4
        # Do a sorting here just for organizing the larger indexes to the right
        face_adjacency_list = np.sort(face_adjacency_list, axis=1)
        # Finally, remove the outside faces
        face_adjacency_list[face_adjacency_list > len(th)-1] = -1
        # TODO: handling of cases wheren count > 2?
        # I can't return unique because order matters
        return faces[idx], inv.reshape(-1, 4), face_adjacency_list


    def get_outside_faces(self, tetrahedra_indexes=None):
        ''' Creates a list of nodes in each face that are in the outer volume

        Parameters
        ----------------
        tetrahedra_indexes: np.ndarray (optional)
            Indices of the tetrehedra where the outer volume is to be determined (1-based
            element indices, default: all tetrahedra)
        Returns
        -------------
        faces: np.ndarray
            Outside faces
        '''
        if tetrahedra_indexes is None:
            tetrahedra_indexes = self.tetrahedra
        th = self[tetrahedra_indexes]
        faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        faces = faces.reshape(-1, 3)

        unique, idx, count = np.unique(
            np.sort(faces, axis=1),
            return_index=True,
            return_counts=True,
            axis=0)

        if np.any(count > 2):
            warnings.warn(
                'Found a face with more than 2 adjacent tetrahedra!')

        outside_faces = faces[idx[count == 1]]
        return outside_faces


    def get_surface_outline(self, triangle_indices=None):
        """ returns the outline of a non-closed surface

        Parameters
        ----------------
        triangle_indices: np.ndarray (optional)
            Indices of the triangles for which the outline should be determined
            (1-based element indices, default: all triangles)

        Returns
        -------------
        edges: np.ndarray
            Nx2 array of node indices

        """
        if triangle_indices is None:
            triangle_indices = self.triangles
        tr = self[triangle_indices][:,0:3]

        edges = tr[:, [[0, 1], [1, 2], [2, 0]]]
        edges = edges.reshape(-1, 2)

        _, idx, count = np.unique(
                np.sort(edges, axis=1),
                return_index=True,
                return_counts=True,
                axis=0)

        if np.any(count > 2):
            warnings.warn('Found an edge with more than 2 adjacent triangles!')

        return edges[idx[count == 1]]


    def nodes_with_tag(self, tags):
        ''' Gets all nodes indexes that are part of at least one element with the given
        tags

        Parameters
        -----------
        tags: list
            Integer tags to search

        Returns
        -------------
        nodes: ndarray of integer
            Indexes of nodes with given tag
        '''
        nodes = np.unique(self[np.isin(self.tag1, tags)].reshape(-1))
        nodes = nodes[nodes > 0]
        return nodes


    def __getitem__(self, index):
        return _getitem_one_indexed(self.node_number_list, index)

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __str__(self):
        s = ''
        s += 'Nr elements: {0}\n'.format(self.nr)
        s += 'elm types: {0}\n'.format(self.elm_type)
        s += 'tags: {0}\n'.format(self.tag1)
        s += 'node list: {0}'.format(self.node_number_list)
        return s

class Msh:
    """class to handle the meshes.
    Gathers Nodes, Elements and Data

    Parameters
    -------------------------
    nodes: (optional) simnibs.msh.Nodes
        Nodes structure

    elements: (optional) simnibs.msh.Elements()
        Elements structure

    fn: str (optional)
        Name of ".msh" file to be read.
        Overides nodes and elements

    Attributes
    -------------------------
    nodes: simnibs.msh.Nodes
        a Nodes field
    elm: simnibs.msh.Elements
        A Elements field
    nodedata: simnibs.msh.NodeData
        list of NodeData filds
    elmdata: simnibs.msh.ElementData
        list of ElementData fields
    fn: str
        name of file
    binary: bool
        wheather or not the mesh was in binary format
   """

    def __init__(self, nodes=None, elements=None, fn=None):
        self.nodes = Nodes()
        self.elm = Elements()
        self.nodedata = []
        self.elmdata = []
        self.fn = ''  # file name to save msh

        if nodes is not None:
            self.nodes = nodes
        if elements is not None:
            self.elm = elements

        if fn is not None:
            self = read_msh(fn, m=self)

    @property
    def field(self):
        '''Dictionary of fields indexed by their name'''
        return dict(
            [(data.field_name, data) for data in self.nodedata + self.elmdata])

    def write(self, out_fn):
        ''' Writes out the mesh as a ".msh" file

        Parameters
        ---------------
        out_fn: str
            Name of output file
        '''
        write_msh(self, out_fn)

    def crop_mesh(self, tags=None, elm_type=None, nodes=None, elements=None):
        """ Crops the specified tags from the mesh
        Generates a new mesh, with only the specified tags
        The nodes are also reordered

        Parameters
        ---------------------
        tags:(optinal) int or list
            list of tags to be cropped, default: all

        elm_type: (optional) list of int
            list of element types to be croped (2 for triangles, 4 for tetrahedra), default: all

        nodes: (optional) list of ints
            List of nodes to be cropped, returns the minimal mesh containing elements
            wih at least one of the given nodes

        elements: (optional) list of ints
            List of elements to be cropped

        Returns
        ---------------------
        simnibs.msh.Msh
            Mesh with only the specified tags/elements/nodes

        Raises
        -----------------------
            ValueError, if the tag and elm_type combination is not foud

        Notes
        -----------
        If more than one (tags, elm_type, nodes, elements) is selected, they are joined by an OR
        operation
        """
        if tags is None and elm_type is None and nodes is None and elements is None:
            raise ValueError("At least one type of crop must be specified")

        elm_keep = np.zeros((self.elm.nr, ), dtype=bool)

        if tags is not None:
            elm_keep += np.in1d(self.elm.tag1, tags)

        if elm_type is not None:
            elm_keep += np.in1d(self.elm.elm_type, elm_type)

        if nodes is not None:
            elm_keep += np.any(np.in1d(self.elm.node_number_list, nodes).reshape(-1, 4), axis=1)

        if elements is not None:
            elm_keep += np.in1d(self.elm.elm_number, elements)

        if not np.any(elm_keep):
            raise ValueError("Could not find any element to crop!")

        idx = np.where(elm_keep)[0]
        nr_elements = len(idx)
        unique_nodes = np.unique(self.elm.node_number_list[idx, :].reshape(-1))
        unique_nodes = unique_nodes[unique_nodes != -1]
        if unique_nodes[0] == 0:
            unique_nodes = np.delete(unique_nodes, 0)
        nr_unique = np.size(unique_nodes)

        # creates a dictionary
        nodes_dict = np.zeros(self.nodes.nr + 1, dtype='int')
        nodes_dict[unique_nodes] = np.arange(1, 1 + nr_unique, dtype='int32')

        # Gets the new node numbers
        node_number_list = nodes_dict[self.elm.node_number_list[idx, :]]

        # and the positions in appropriate order
        node_coord = self.nodes.node_coord[unique_nodes - 1]

        # gerenates new mesh
        cropped = Msh()

        cropped.elm.tag1 = np.copy(self.elm.tag1[idx])
        cropped.elm.tag2 = np.copy(self.elm.tag2[idx])
        cropped.elm.elm_type = np.copy(self.elm.elm_type[idx])
        cropped.elm.node_number_list = np.copy(node_number_list)
        cropped.elm.node_number_list[cropped.elm.elm_type == 2, 3] = -1

        cropped.nodes.node_coord = np.copy(node_coord)

        cropped.nodedata = copy.deepcopy(self.nodedata)

        for nd in cropped.nodedata:
            nd.mesh = cropped
            if nd.nr_comp == 1:
                nd.value = np.copy(nd.value[unique_nodes - 1])
            else:
                nd.value = np.copy(nd.value[unique_nodes - 1, :])

        for ed in self.elmdata:
            cropped.elmdata.append(
                ElementData(ed.value[idx],
                            ed.field_name,
                            mesh=cropped))

        return cropped

        """ Join the current mesh with another

        Parameters
        -----------
        other: simnibs.msh.Msh
            Mesh to be joined

        Returns
        --------
        joined: simnibs.msh.Msh
            Mesh with joined nodes and elements
        """
        if self.nodes.nr == 0:
            return copy.deepcopy(other)
        elif other.nodes.nr == 0:
            return copy.deepcopy(self)

        joined = copy.deepcopy(self)
        joined.elmdata = []
        joined.nodedata = []
        other = copy.deepcopy(other)

        joined.nodes.node_coord = np.vstack([joined.nodes.node_coord,
                                             other.nodes.node_coord])
        other_node_number_list = other.elm.node_number_list + self.nodes.nr
        joined.elm.node_number_list = np.vstack([joined.elm.node_number_list,
                                                 other_node_number_list])
        joined.elm.tag1 = np.hstack([joined.elm.tag1, other.elm.tag1])
        joined.elm.tag2 = np.hstack([joined.elm.tag2, other.elm.tag2])
        joined.elm.elm_type = np.hstack([joined.elm.elm_type, other.elm.elm_type])
        lines = np.where(joined.elm.elm_type == 1)[0]
        points = np.where(joined.elm.elm_type == 15)[0]
        triangles = np.where(joined.elm.elm_type == 2)[0]
        tetrahedra = np.where(joined.elm.elm_type == 4)[0]
        new_elm_order = np.hstack([points, lines, triangles, tetrahedra])
        joined.elm.node_number_list = joined.elm.node_number_list[new_elm_order]
        joined.elm.tag1 = joined.elm.tag1[new_elm_order]
        joined.elm.tag2 = joined.elm.tag2[new_elm_order]
        joined.elm.elm_type = joined.elm.elm_type[new_elm_order]
        joined.elm.node_number_list[joined.elm.elm_type == 2, 3] = -1
        joined.elm.node_number_list[joined.elm.elm_type == 1, 2:] = -1
        joined.elm.node_number_list[joined.elm.elm_type == 15, 1:] = -1

        for nd in self.nodedata:
            assert len(nd.value) == self.nodes.nr
            pad_length = [(0, other.nodes.nr)] + [(0, 0)] * (nd.value.ndim - 1)
            new_values = np.pad(nd.value.astype(float), pad_length, 'constant', constant_values=np.nan)
            joined.nodedata.append(NodeData(new_values, nd.field_name, mesh=joined))

        for ed in self.elmdata:
            assert len(ed.value) == self.elm.nr
            pad_length = [(0, other.elm.nr)] + [(0, 0)] * (ed.value.ndim - 1)
            new_values = np.pad(ed.value.astype(float), pad_length, 'constant', constant_values=np.nan)
            joined.elmdata.append(ElementData(new_values[new_elm_order], ed.field_name, mesh=joined))

        for nd in other.nodedata:
            assert len(nd.value) == other.nodes.nr
            pad_length = [(self.nodes.nr, 0)] + [(0, 0)] * (nd.value.ndim - 1)
            new_values = np.pad(nd.value.astype(float), pad_length, 'constant', constant_values=np.nan)
            joined.nodedata.append(NodeData(new_values, nd.field_name, mesh=joined))

        for ed in other.elmdata:
            assert len(ed.value) == other.elm.nr
            pad_length = [(self.elm.nr, 0)] + [(0, 0)] * (ed.value.ndim - 1)
            new_values = np.pad(ed.value.astype(float), pad_length, 'constant', constant_values=np.nan)
            joined.elmdata.append(ElementData(new_values[new_elm_order], ed.field_name, mesh=joined))

        return joined
    def elements_volumes_and_areas(self):
        """ Calculates the volumes of tetrahedra and areas of triangles

        Returns
        ----------
        v: simnibs.Msh.ElementData
            Volume/areas of tetrahedra/triangles

        Note
        ------
            In the mesh's unit (normally mm)
        """
        vol = ElementData(np.zeros(self.elm.nr, dtype=float),
                          'volumes_and_areas')

        tr_indexes = self.elm.triangles
        node_tr = self.nodes[self.elm[tr_indexes, :3]]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        n = np.cross(sideA, sideB)
        vol[tr_indexes] = np.linalg.norm(n, axis=1) * 0.5

        th_indexes = self.elm.tetrahedra
        node_th = self.nodes[self.elm[th_indexes]]
        M = node_th[:, 1:] - node_th[:, 0, None]
        vol[th_indexes] = np.abs(np.linalg.det(M)) / 6.

        return vol

    def find_closest_element(self, querry_points, return_index=False,
                             elements_of_interest=None,
                             k=1, return_distance=False):
        """ Finds the closest element to each point in p

        Parameters
        --------------------------------
        querry_points: (Nx3) ndarray
            List of points (x,y,z) to which the closest element in the mesh should be found

        return_index: (optional) bool
            Whether to return the index of the closes nodes, default=False

        elements_of_interest: (opional) list
            list of element indices that are of interest

        k: (optional) int
            number of nearest neighbourt to return

        Returns
        -------------------------------
        coords: Nx3 array of floats
            coordinates of the baricenter of the closest element

        indexes: Nx1 array of ints
            Indice of the closest elements

        Notes
        --------------------------------------
        The indices are in the mesh listing, that starts at one!

        """
        if len(self.elm.node_number_list) == 0:
            raise ValueError('Mesh has no elements defined')

        baricenters = self.elements_baricenters()
        if elements_of_interest is not None:
            bar = baricenters[elements_of_interest]
        else:
            elements_of_interest = baricenters.elm_number
            bar = baricenters.value

        kd_tree = scipy.spatial.cKDTree(bar)
        d, indexes = kd_tree.query(querry_points, k=k)
        indexes = elements_of_interest[indexes]
        coords = baricenters[indexes]

        if return_distance and return_index:
            return coords, indexes, d

        elif return_index:
            return coords, indexes

        elif return_distance:
            return coords, d

        else:
            return coords

    def elm_node_coords(self, elm_nr=None, tag=None, elm_type=None):
        """ Returns the position of each of the element's nodes

        Arguments
        -----------------------------
        elm_nr: (optional) array of ints
            Elements to return, default: Return all elements
        tag: (optional) array of ints
            Only return elements with specified tag. default: all tags
        elm_type: (optional) array of ints
            Only return elements of specified type. default: all

        Returns
        -----------------------------
        Nx4x3 ndarray
            Array with node position of every element
            For triangles, the fourth coordinates are 0,0,0
        """
        elements_to_return = np.ones((self.elm.nr, ), dtype=bool)

        if elm_nr is not None:
            elements_to_return[elm_nr] = True

        if elm_type is not None:
            elements_to_return = np.logical_and(
                elements_to_return,
                np.in1d(self.elm.elm_type, elm_type))

        if tag is not None:
            elements_to_return = np.logical_and(
                elements_to_return,
                np.in1d(self.elm.tag1, tag))

        tmp_node_coord = np.vstack((self.nodes.node_coord, [0, 0, 0]))

        elm_node_coords = \
            tmp_node_coord[self.elm.node_number_list[elements_to_return, :] - 1]

        return elm_node_coords
    

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def triangle_normals(self, smooth=False):
        """ Calculates the normals of triangles

        Parameters
        ------------
        smooth (optional): int
            Number of smoothing steps to perform. Default: 0
        Returns
        --------
        normals: ElementData
            normals of triangles, zero at the tetrahedra

        """
        normals = ElementData(
            np.zeros((self.elm.nr, 3), dtype=float),
            'normals'
        )
        tr_indexes = self.elm.triangles
        node_tr = self.nodes[self.elm[tr_indexes, :3]]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        n = np.cross(sideA, sideB)
        normals[tr_indexes] = n / np.linalg.norm(n, axis=1)[:, None]
        if smooth == 0:
            node_tr = self.nodes[self.elm[tr_indexes, :3]]
            sideA = node_tr[:, 1] - node_tr[:, 0]

            sideB = node_tr[:, 2] - node_tr[:, 0]
            n = np.cross(sideA, sideB)
            normals[tr_indexes] = n / np.linalg.norm(n, axis=1)[:, None]
        elif smooth > 0:
            normals_nodes = self.nodes_normals(smooth=smooth)
            tr = self.elm[tr_indexes, :3]
            n = np.mean(normals_nodes[tr], axis=1)
            normals[tr_indexes] = n / np.linalg.norm(n, axis=1)[:, None]
        else:
            raise ValueError('smooth parameter must be >= 0')
        return normals

    def nodes_volumes_or_areas(self):
        ''' Return the volume (volume mesh) if area (surface mesh) of all nodes
        Only works for ordered values of mesh and node indices

        Returns
        -------------
        nd: NodeData
            NodeData structure with the volume or area of each node
        '''
        nd = np.zeros(self.nodes.nr)
        if len(self.elm.tetrahedra) > 0:
            name = 'volumes'
            volumes = self.elements_volumes_and_areas()[self.elm.tetrahedra]
            th_nodes = self.elm[self.elm.tetrahedra] - 1
            for i in range(4):
                nd[:np.max(th_nodes[:, i]) + 1] += \
                    np.bincount(th_nodes[:, i], volumes / 4.)

        elif len(self.elm.triangles) > 0:
            name = 'areas'
            areas = self.elements_volumes_and_areas()[self.elm.triangles]
            tr_nodes = self.elm[self.elm.triangles] - 1
            for i in range(3):
                nd[:np.max(tr_nodes[:, i]) + 1] += \
                    np.bincount(tr_nodes[:, i], areas / 3.)

        return NodeData(nd, name)

    def nodes_areas(self):
        ''' Areas for all nodes in a surface

        Returns
        ---------
        nd: NodeData
            NodeData structure with normals for each node

        '''
        areas = self.elements_volumes_and_areas()[self.elm.triangles]
        triangle_nodes = self.elm[self.elm.triangles, :3] - 1
        nd = np.bincount(
            triangle_nodes.reshape(-1),
            np.repeat(areas/3., 3), self.nodes.nr
        )

        return NodeData(nd, 'areas')


    def nodes_normals(self, triangles=None, smooth=0):
        ''' Normals for all nodes in a surface

        Parameters
        ------------
        triangles: list of ints
            List of triangles to be taken into consideration for calculating the normals

        smooth: int (optional)
            Number of smoothing cycles to perform. Default: 0

        Returns
        ---------
        nd: NodeData
            NodeData structure with normals for each surface node

        '''
        if triangles is None:
            elements = self.elm.triangles - 1
        else:
            elements = triangles - 1
        triangle_nodes = self.elm.node_number_list[elements, :3] - 1

        node_tr = self.nodes.node_coord[triangle_nodes]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        normals = np.cross(sideA, sideB)

        nd = np.zeros((self.nodes.nr, 3))
        for s in range(smooth + 1):
            for i in range(3):
                nd[:, i] = \
                    np.bincount(triangle_nodes.reshape(-1),
                                np.repeat(normals[:, i], 3),
                                self.nodes.nr)

            normals = np.sum(nd[triangle_nodes], axis=1)
            normals /= np.linalg.norm(normals, axis=1)[:, None]

        nodes = np.unique(triangle_nodes)
        nd[nodes] = nd[nodes] / np.linalg.norm(nd[nodes], axis=1)[:, None]

        return NodeData(nd, 'normals')

    def compact_ordering(self, node_number):
        ''' Changes the node and element ordering so that it goes from 1 to nr_nodes

        Parameters
        --------------
        node_number: N_nodes x 1 ndarray of inte
            Node numbering in the original mesh
        '''
        rel = int(-9999) * np.ones((np.max(node_number) + 1), dtype=int)
        rel[node_number] = np.arange(1, self.nodes.nr + 1, dtype=int)
        self.elm.node_number_list = rel[self.elm.node_number_list]
        self.elm.node_number_list[self.elm.elm_type == 2, 3] = -1

    def prepare_surface_tags(self):
        triangles = self.elm.elm_type == 2
        surf_tags = np.unique(self.elm.tag1[triangles])
        for s in surf_tags:
            if s < 1000:
                self.elm.tag1[triangles *
                              (self.elm.tag1 == s)] += 1000

    def find_corresponding_tetrahedra(self):
        ''' Finds the tetrahedra corresponding to each triangle

        Returns
        ---------
        corresponding_th_indices: ndarray of ints
           List of the element indices of the tetrahedra corresponding to each triangle.
           Note: This is in mesh ordering (starts at 1), -1 if there's no corresponding

        '''
        # Look into the cache
        node_nr_list_hash = hashlib.sha1(
            np.hstack((self.elm.tag1[:, None], self.elm.node_number_list))).hexdigest()
        try:
            if self._correspondance_node_nr_list_hash == node_nr_list_hash:
                return self._corresponding_tetrahedra
            else:
                raise AttributeError

        except AttributeError:
            pass

        # If could not find correspondence in cache
        tr_tags = np.unique(self.elm.tag1[self.elm.elm_type == 2])
        corresponding_th_indices = -np.ones(len(self.elm.triangles), dtype=int)
        for t in tr_tags:
            # look into tetrahedra with tags t, 1000-t
            to_crop = [t]
            to_crop.append(t - 1000)
            if t >= 1100:  # Electrodes
                to_crop.append(t - 600)
            if t >= 2000:
                to_crop.append(t - 2000)
                to_crop.append(t - 1600)
            # Select triangles and tetrahedra with tags
            th_of_interest = np.where((self.elm.elm_type == 4) *
                                      np.in1d(self.elm.tag1, to_crop))[0]
            if len(th_of_interest) == 0:
                continue
            tr_of_interest = np.where((self.elm.elm_type == 2) *
                                      (self.elm.tag1 == t))[0]

            th = self.elm.node_number_list[th_of_interest]
            faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]].reshape(-1, 3)
            faces = faces.reshape(-1, 3)
            faces_hash_array = _hash_rows(faces)

            tr = self.elm.node_number_list[tr_of_interest, :3]
            tr_hash_array = _hash_rows(tr)
            # This will check if all triangles have a corresponding face
            has_tetra = np.in1d(tr_hash_array, faces_hash_array)
            faces_argsort = faces_hash_array.argsort()
            faces_hash_array = faces_hash_array[faces_argsort]
            tr_search = np.searchsorted(faces_hash_array, tr_hash_array[has_tetra])
            # Find the indice
            index = faces_argsort[tr_search] // 4

            # Put the values in corresponding_th_indices
            position = np.searchsorted(self.elm.triangles,
                                       self.elm.elm_number[tr_of_interest[has_tetra]])
            corresponding_th_indices[position] = \
                self.elm.elm_number[th_of_interest[index]]

        if np.any(corresponding_th_indices==-1):
            # add triangles at the outer boundary, irrespective of tag
            # get all tet faces, except those of "air" tetrahedra (i.e., tag1 = -1)
            idx_th_all = np.where((self.elm.elm_type == 4)*(self.elm.tag1 != -1))[0]
            th = self.elm.node_number_list[idx_th_all]
            faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
            faces = faces.reshape(-1, 3)
            faces_hash_array = _hash_rows(faces)
            # keep only tet faces that occur once (i.e. at the outer boundary)
            [faces_hash_array, idx_fc, counts] = np.unique(faces_hash_array,
                                                  return_index = True,
                                                  return_counts = True)
            faces_hash_array = faces_hash_array[counts == 1]
            idx_fc = idx_fc[counts == 1]

            tr_of_interest = self.elm.triangles[corresponding_th_indices==-1]-1
            tr = self.elm.node_number_list[tr_of_interest, :3]
            tr_hash_array = _hash_rows(tr)
            _, idx_tr, idx_th = np.intersect1d(tr_hash_array, faces_hash_array,
                                               return_indices = True)
            # recover indices
            idx_tr = tr_of_interest[idx_tr]
            idx_th = idx_th_all[ idx_fc[idx_th]//4 ] + 1
            corresponding_th_indices[idx_tr] = idx_th

        self._correspondance_node_nr_list_hash = node_nr_list_hash
        self._corresponding_tetrahedra = corresponding_th_indices
        gc.collect()
        return corresponding_th_indices

    def add_element_field(self, field, field_name):
        ''' Adds field defined in the elements

        Parameters
        ----------
        field: np.ndarray or simnibs.ElementData
            Value of field in the mesh elements. Should have the shape
            - (n_elm,) or (n_elm, 1) for scalar fields
            - (n_elm, 3) for vector fields
            - (n_elm, 9) for tensors

        field_name: str
            Name for the field.

        Returns
        --------
        ed: simnibs.ElementData
            ElementData class with the input field

        '''
        if isinstance(field, ElementData):
            if field.nr != self.elm.nr:
                raise ValueError('Number of data points in the field '
                                 'and number of mesh elements do not match')
            field.field_name = field_name
            field.mesh = self
            self.elmdata.append(field)

            return field

        else:
            if self.elm.nr != field.shape[0]:
                raise ValueError('Number of data points in the field '
                                 'and number of mesh elements do not match')
            ed = ElementData(field, field_name, self)
            self.elmdata.append(ed)
            return ed

    def fields_summary(self, roi=None, fields=None,
                       percentiles=(99.9, 99, 95),
                       focality_cutoffs=(75, 50)):
        ''' Creates a text summary of the field

        Parameters
        ------------
        roi: list (optional)
            Regions of interest, in tissue tags. Default: Whole mesh
        fields: list (optional)
            Fields for which to calculate the summary
        percentiles: ndarray (optinal)
            Field percentiles to be printed. Default: (99.9, 99, 95)
        focality_cutoffs: ndarray (optional)
            Cuttofs for focality calculations. Default: (75, 50)
        '''
        if roi is None:
            mesh = self
        else:
            try:
                mesh = self.crop_mesh(roi)
            except ValueError:
                warnings.warn(f"Could not find any element with tags {roi}")
                return f"Could not find any element with tags {roi}"

        if fields is None:
            fields = self.field.keys()

        units = []
        for f in fields:
            if f in ['E', 'magnE', 'D', 'g']:
                units.append(' V/m')
            elif f in ['J', 'magnJ']:
                units.append(' A/m²')
            elif f == 'v':
                units.append(' V')
            else:
                units.append('')

        if 2 in mesh.elm.elm_type:
            units_mesh = ' mm²'
        elif 4 in mesh.elm.elm_type:
            units_mesh = ' mm³'
        if np.all(np.isin([2, 4], mesh.elm.elm_type)):
            warnings.warn("Can't report Field summary in meshes with volumes and surfaces")
            return ''

        percentiles_table = [['Field'] + [f'{p:.1f}%' for p in percentiles]]
        focality_table = [['Field'] + [f'{f:.1f}%' for f in focality_cutoffs]]
        for fn, u in zip(fields, units):
            f = mesh.field[fn]
            prc = f.get_percentiles(percentiles)
            percentiles_table.append([fn] + [f'{p:.2e}{u}' for p in prc])
            focality = f.get_focality(focality_cutoffs, 99.9)
            focality_table.append([fn] + [f'{fv:.2e}{units_mesh}' for fv in focality])

        def format_table(table):
            entry_sizes = np.array([[len(e) for e in row] for row in table])
            col_sizes = np.max(entry_sizes, axis=0)
            align_string = ['{:<' + str(cs) + '}' for cs in col_sizes]
            t = ''
            for i, row in enumerate(table):
                t += '|'
                t += ' |'.join(a_s.format(col) for a_s, col in zip(align_string, row))
                t += ' |\n'
                if i == 0:
                    t += '|' + '|'.join((cs+1) * '-' for cs in col_sizes) + '|\n'
            return t

        string = ''
        string += 'Field Percentiles\n'
        string += '-----------------\n'
        string += 'Top percentiles of the field (or field magnitude for vector fields)\n'
        string += format_table(percentiles_table)
        string += '\n'
        string += 'Field Focality\n'
        string += '---------------\n'
        string += 'Mesh volume or area with a field >= X% of the 99.9th percentile\n'
        string += format_table(focality_table)

        return string

    def smooth_surfaces(self, n_steps, step_size=.3, tags=None, max_gamma=5):
        ''' In-place smoothing of the mesh surfaces using Taubin smoothing,
            ensures that the tetrahedra quality does not fall below
            a minimal level. Can still decrease overall tetrahedra quality, though.

        Parameters
        ------------
        msh: simnibs.msh.Msh
            Mesh structure, must have inner triangles
        n_steps: int
            number of smoothing steps to perform
        step_size (optional): float 0<step_size<1
            Size of each smoothing step. Default: 0.3
        tags: (optional) int or list
            list of tags to be smoothed. Default: all
        max_gamma: (optional) float
            Maximum gamma tetrahedron quality metric. Surface nodes are
            excluded from smoothing when gamma of neighboring tetrahedra
            would exceed max_gamma otherwise. Default: 5
            (for gamma metric see Parthasarathy et al., Finite Elements in
             Analysis and Design, 1994)
        '''
        assert step_size > 0 and step_size < 1
        # Surface nodes and surface node mask
        idx = self.elm.elm_type == 2
        if tags is not None:
            idx *= np.in1d(self.elm.tag1, tags)
        surf_nodes = np.unique(self.elm.node_number_list[idx,:3]) - 1
        nodes_mask = np.zeros(self.nodes.nr, dtype=bool)
        nodes_mask[surf_nodes] = True

        # Triangle neighbourhood information
        adj_tr = scipy.sparse.csr_matrix((self.nodes.nr, self.nodes.nr), dtype=bool)
        tr = self.elm[self.elm.triangles, :3] - 1
        ones = np.ones(len(tr), dtype=bool)
        for i in range(3):
            for j in range(3):
                adj_tr += scipy.sparse.csr_matrix(
                    (ones, (tr[:, i], tr[:, j])),
                    shape=adj_tr.shape
                )
        adj_tr -= scipy.sparse.dia_matrix((np.ones(adj_tr.shape[0]), 0), shape=adj_tr.shape)

        # Tetrahedron neighbourhood information
        th = self.elm[self.elm.tetrahedra] - 1
        adj_th = scipy.sparse.csr_matrix((self.nodes.nr, len(th)), dtype=bool)
        th_indices = np.arange(len(th))
        ones = np.ones(len(th), dtype=bool)
        for i in range(4):
            adj_th += scipy.sparse.csr_matrix(
                (ones, (th[:, i], th_indices)),
                shape=adj_th.shape
            )
        # keep only tets connected to surface nodes
        idx = np.sum(adj_th[nodes_mask], axis=0) > 0
        idx = np.asarray(idx).reshape(-1)
        th = th[idx,:]
        adj_th = adj_th[:,idx]
        adj_th = adj_th.tocsc()

        def calc_gamma(nodes_coords,th):
            ''' gamma of tetrahedra '''
            node_th = nodes_coords[th]
            # tet volumes
            M = node_th[:, 1:] - node_th[:, 0, None]
            vol = np.linalg.det(M) / 6.
            # edge lengths
            edge_rms = np.zeros(len(th))
            for i in range(4):
                for j in range(i+1, 4):
                    edge_rms += np.sum(
                        (node_th[:,i,:] - node_th[:,j,:])**2,
                        axis=1
                    )
            edge_rms = edge_rms/6.
            # gamma
            gamma = edge_rms**1.5/vol
            gamma /= 8.479670
            gamma[vol<0] = -1
            return gamma

        nodes_coords = np.ascontiguousarray(self.nodes.node_coord, float)
        gamma = calc_gamma(nodes_coords,th)
        n_badgamma = np.sum((gamma < 0) + (gamma > max_gamma))
        for i in range(n_steps):
            nc_before = nodes_coords.copy()
            cython_msh.gauss_smooth_simple(
                surf_nodes.astype(np.uint),
                nodes_coords,
                np.ascontiguousarray(adj_tr.indices, np.uint),
                np.ascontiguousarray(adj_tr.indptr, np.uint),
                float(step_size)
            )
            # Taubin step
            cython_msh.gauss_smooth_simple(
                surf_nodes.astype(np.uint),
                nodes_coords,
                np.ascontiguousarray(adj_tr.indices, np.uint),
                np.ascontiguousarray(adj_tr.indptr, np.uint),
                -1.05 * float(step_size)
            )
            # revert where gamma exceeded max_gamma
            gamma = calc_gamma(nodes_coords,th)
            idx_badtet = (gamma < 0) + (gamma > max_gamma)
            for k in range(4): # mostly < 4 iterations required, limit to ensure stability
                if np.sum(idx_badtet) <= n_badgamma: break

                idx_badnodes = np.sum(adj_th[:,idx_badtet], axis=1) > 0
                idx_badnodes = np.asarray(idx_badnodes).reshape(-1)
                nodes_coords[idx_badnodes] = nc_before[idx_badnodes]

                gamma = calc_gamma(nodes_coords,th)
                idx_badtet = (gamma < 0) + (gamma > max_gamma)

            n_badgamma = np.sum(idx_badtet)

        self.nodes.node_coord = nodes_coords


    def smooth_surfaces_simple(self, n_steps, step_size=.3, tags=None, nodes_mask=None):
        ''' In-place smoothing of the mesh surfaces using Taubin smoothing,
            no control of tetrahedral quality

        Parameters
        ------------
        msh: simnibs.msh.Msh
            Mesh structure, must have inner triangles
        n_steps: int
            number of smoothing steps to perform
        step_size (optional): float 0<step_size<1
            Size of each smoothing step. Default: 0.3
        tags: (optional) int or list
            list of tags to be smoothed. Default: all
        nodes_mask: (optional) bool
            mask of nodes to be smoothed. Default: all
        '''
        assert step_size > 0 and step_size < 1
        # Surface nodes and surface node mask
        idx = self.elm.elm_type == 2
        if tags is not None:
            idx *= np.in1d(self.elm.tag1, tags)
        surf_nodes = np.unique(self.elm.node_number_list[idx,:3]) - 1

        if nodes_mask is not None:
            assert len(nodes_mask) == self.nodes.nr
            mask = np.zeros(self.nodes.nr, dtype=bool)
            mask[surf_nodes] = True
            mask *= nodes_mask
            surf_nodes = np.where(mask)[0]

        # Triangle neighbourhood information
        adj_tr = scipy.sparse.csr_matrix((self.nodes.nr, self.nodes.nr), dtype=bool)
        tr = self.elm[self.elm.triangles, :3] - 1
        ones = np.ones(len(tr), dtype=bool)
        for i in range(3):
            for j in range(3):
                adj_tr += scipy.sparse.csr_matrix(
                    (ones, (tr[:, i], tr[:, j])),
                    shape=adj_tr.shape
                )
        adj_tr -= scipy.sparse.dia_matrix((np.ones(adj_tr.shape[0]), 0), shape=adj_tr.shape)

        nodes_coords = np.ascontiguousarray(self.nodes.node_coord, float)
        for i in range(n_steps):
            cython_msh.gauss_smooth_simple(
                surf_nodes.astype(np.uint),
                nodes_coords,
                np.ascontiguousarray(adj_tr.indices, np.uint),
                np.ascontiguousarray(adj_tr.indptr, np.uint),
                float(step_size)
            )
            # Taubin step
            cython_msh.gauss_smooth_simple(
                surf_nodes.astype(np.uint),
                nodes_coords,
                np.ascontiguousarray(adj_tr.indices, np.uint),
                np.ascontiguousarray(adj_tr.indptr, np.uint),
                -1.05 * float(step_size)
            )

        self.nodes.node_coord = nodes_coords


    def gamma_metric(self):
        """ calculates the (normalized) Gamma quality metric for tetrahedra

        Returns
        ----------
        gamma: ElementData
            Gamma metric from Parthasarathy et al., 1994
        """
        vol = self.elements_volumes_and_areas()
        th = self.elm.elm_type == 4
        edge_rms = np.zeros(self.elm.nr)
        for i in range(4):
            for j in range(i+1, 4):
                edge_rms[th] += np.sum(
                    (self.nodes[self.elm[th, i]] - self.nodes[self.elm[th, j]])**2,
                    axis=1
                )
        edge_rms = np.sqrt(edge_rms/6.)
        gamma = np.zeros(self.elm.nr)
        gamma[th] = edge_rms[th]**3/vol[th]
        gamma /= 8.479670 # dividing by value for equilateral tetrahedra -> normalized value as metric
        return ElementData(gamma, 'gamma', self)


    def surface_EC(self):
        """ return euler characteristic of surfaces """
        idx_tr = self.elm.elm_type == 2

        nr_tr = np.sum(idx_tr)
        nr_node_tr = np.unique(self.elm.node_number_list[idx_tr,0:3].flatten()).shape[0]

        M = np.sort(self.elm.node_number_list[idx_tr,0:3], axis=1)
        nr_edges = np.unique(np.vstack( (M[:,[0,1]], M[:,[1,2]], M[:,[0,2]]) ), axis=0).shape[0]

        EC = nr_node_tr + nr_tr - nr_edges
        return EC


    def split_tets_along_line(self, idx_n1, idx_n2, do_checks = True,
                              return_tetindices = False):
        """
        Adds a new node in the middle between the two given nodes
        and splits all tetrahedra connected to the line between the
        two given nodes. Works in-place.

        Parameters
        ----------
        idx_n1 : int
            index of first node
        idx_n2 : int
            index of second node
        do_checks : bool, optional
            The mesh must only contain tets and no data. The corresponding
            checks can be disabled to gain a bit of speed. The default is True.
        return_tetindices : bool, optional
            The indices of the new tetrahedra are returned if set to True.
            The default is False.

        Returns
        -------
        tets1, tets2: lists of ints (optinal)
            Returns the indices of the new tetrahedra connected to nodes n1 and n2.
            (when return_tetindices = True). The new node is added at the end, i.e.
            its index equals the number of nodes in the mesh. Add indices are 1-based.
        """
        if do_checks:
            if not np.all(self.elm.elm_type == 4):
                raise TypeError("The mesh must only contain tetrahedra")
            if len(self.elmdata) > 0 or len(self.nodedata) > 0:
                raise TypeError("The mesh must not contain data")

        # get tets connected to the two nodes
        idx_orgtets = np.where( np.any(self.elm.node_number_list == idx_n1,axis=1) *
                                np.any(self.elm.node_number_list == idx_n2,axis=1) )[0]
        if len(idx_orgtets) == 0:
            raise ValueError("The two nodes are not connected!")

        # add new node
        pos_newnode = np.mean( self.nodes.node_coord[[idx_n1-1,idx_n2-1],:],
                               axis=0 )
        self.nodes.node_coord = np.vstack((self.nodes.node_coord, pos_newnode))
        idx_newnode = self.nodes.nr

        # add new tets - connect them to the new node and node n2
        idx_newtets = np.arange(self.elm.nr, self.elm.nr+len(idx_orgtets))
        self.elm.node_number_list = np.vstack((self.elm.node_number_list,
                                               self.elm.node_number_list[idx_orgtets]))
        self.elm.tag1 = np.hstack((self.elm.tag1, self.elm.tag1[idx_orgtets]))
        self.elm.tag2 = np.hstack((self.elm.tag2, self.elm.tag2[idx_orgtets]))
        self.elm.elm_type = np.hstack((self.elm.elm_type,
                                       4*np.ones((len(idx_orgtets)), np.int32)))

        idx = np.where(self.elm.node_number_list[idx_newtets] == idx_n1)[1]
        self.elm.node_number_list[idx_newtets,idx] = idx_newnode

        # connect old tets to to the new node and node n1
        idx = np.where(self.elm.node_number_list[idx_orgtets] == idx_n2)[1]
        self.elm.node_number_list[idx_orgtets,idx] = idx_newnode

        if return_tetindices:
            return idx_orgtets+1, idx_newtets+1


class Data(object):
    """Store data in elements or nodes

    Parameters
    -----------------------
    value: np.ndarray
        Value of field in nodes

    field_name: str (optional)
        name of field. Default: empty

    mesh: simnibs.msh.Msh (optional)
        Mesh where the field is define. Required for several methods

    Attributes
    --------------
    value: ndarray
        Value of field in nodes
    field_name: str
        name of field
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)

    """

    def __init__(self, value, name='', mesh=None):
        self.field_name = name
        self.value = value
        self.mesh = mesh

        if value.ndim > 2:
            raise ValueError('Can only handle 1 and 2 dimensional fields '
                             'Tensors should be given as a Nx9 array')

        if self.nr_comp > self.nr:
            warnings.warn('Second axis larger than the first '
                          'Field is probably transposed')

    @property
    def type(self):
        '''NodeData of ElementData'''
        return self.__class__.__name__

    @property
    def nr(self):
        '''Number of data entries'''
        return self.value.shape[0]

    @property
    def nr_comp(self):
        '''Number of field components'''
        try:
            return self.value.shape[1]
        except IndexError:
            return 1

    def to_nifti(self, n_voxels, affine, fn=None, units='mm', qform=None,
                 method='linear', continuous=False):
        ''' Transforms the data in a nifti file

        Parameters
        -----------
        n_voxels: list of ints
            Number of vexels in each dimension
        affine: 4x4 ndarray
            Transformation of voxel space into xyz. This sets the sform
        fn: str (optional)
            String with file name to be used, if the result is to be saved
        units: str (optional)
            Units to be set in the NifTI header. Default: mm
        qform: 4x4 ndarray (optional)
            Header qform. Default: set the same as the affine
        method: {'assign' or 'linear'} (Optional)
            If 'assign', gives to each voxel the value of the element that contains
            it. If linear, first assign fields to nodes, and then perform
            baricentric interpolatiom. Only for ElementData input. Default: linear
        continuous: bool
            Wether fields is continuous across tissue boundaries. Changes the
            behaviour of the function only if method == 'linear'. Default: False

        Returns
        ---------
        img: nibabel.Nifti1Pair
            Image object with the field interpolated in the voxels
        '''
        data = self.interpolate_to_grid(n_voxels, affine, method=method,
                                        continuous=continuous)
        if data.dtype == np.bool_ or data.dtype == bool:
            data = data.astype(np.uint8)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        img = nibabel.Nifti1Pair(data, affine)
        img.header.set_xyzt_units(units)
        if qform is not None:
            img.set_qform(qform)
        else:
            img.set_qform(affine)
        img.update_header()
        del data
        if fn is not None:
            nibabel.save(img, fn)
        else:
            return img


    def _norm(self):
        ''' simple norm of the field '''
        if self.nr_comp == 1:
            return np.abs(self.value).reshape(-1)
        else:
            return np.linalg.norm(self.value, axis=1)

    def _weights(self, roi=slice(None)):
        ''' Area / volume of each nodes or element '''
        if isinstance(self, NodeData):
            return self.mesh.nodes_volumes_or_areas()[roi]

        elif isinstance(self, ElementData):
            return self.mesh.elements_volumes_and_areas()[roi]

        else:
            raise NotImplementedError


    def mean_field_norm(self):
        ''' Calculates V*w/sum(w)
        Where V is the magnitude of the field, and w is the volume or area of the mesh where
        the field is defined. This can be used as a focality metric. It should give out
        small values when the field is focal.

        Returns
        ----------
        eff_area: float
            Area or volume of mesh, weighted by the field
        '''
        self._test_msh()
        if np.all(np.isin([2, 4], self.mesh.elm.elm_type)):
            warnings.warn('Calculating effective volume/area of fields in meshes with'
                          ' triangles and tetrahedra can give misleading results')

        norm = self._norm()
        weights = self._weights()

        return np.sum(norm * weights) / np.sum(weights)

    def get_percentiles(self, percentile=[99.9], roi=None):
        ''' Get percentiles of field (or field magnitude, if a vector field)

        Parameters
        ------------
        percentile: ndarray (optional)
            Percentiles of interest, between 0 and 100. Defaut: 99.9

        roi: ndarray (optinal)
            Region of interest in terms of element/node indices. Default: the whole mesh

        Returnts
        ----------
        f_p: ndarray
            Field at the given percentiles
        '''
        self._test_msh()
        if roi is None:
            roi = slice(None)

        if self.nr_comp > 1:
            v = np.linalg.norm(self[roi], axis=1)
        else:
            v = np.squeeze(self[roi])
        s = np.argsort(v)
        v = v[s]
        weights = self._weights(roi)[s]
        weights = np.cumsum(weights)
        weights /= weights[-1]
        perc = np.array(percentile, dtype=float) / 100
        perc = perc.reshape(-1)
        closest = np.zeros(perc.shape, dtype=int)
        for i, p in enumerate(perc):
            closest[i] = np.argmin(np.abs(weights - p))

        return v[closest]

    def get_focality(self, cuttofs=[50, 70], peak_percentile=99.9):
        ''' Caluclates field focality as the area/volume of the mesh experiencing a field
        magnitude of above (cut_off% of the field peak). peak_percentile gives what is the
        field peak

        Parameters
        ------------
        cuttofs: ndarray (optional)
            Percentage of the peak value for the cut_off, between 0 and 100. Default: [50, 70]

        peak_percentile: float (optional)
            Percentile to be used to calculate peak value. Default: 99.9

        Returns
        ---------
        focality: ndarray
            Area/volume exceeding the cuttof of the peak value
        '''
        self._test_msh()
        if np.all(np.isin([2, 4], self.mesh.elm.elm_type)):
            warnings.warn('Calculating focality of fields in meshes with'
                          ' triangles and tetrahedra can give misleading results')

        norm = self._norm()
        s = np.argsort(norm)
        norm = norm[s]
        weights = self._weights()[s]
        weights_norm = np.cumsum(weights)
        weights_norm = weights_norm * 100 / weights_norm[-1]
        peak_value = norm[np.argmin(np.abs(weights_norm - peak_percentile))]

        co = np.array(cuttofs, dtype=float).reshape(-1) / 100
        focality = np.zeros(co.shape, dtype=int)
        for i, c in enumerate(co):
            focality[i] = np.sum(weights[norm > c * peak_value])

        return focality

    def summary(self, percentiles=(99.9, 99, 95), focality_cutoffs=(75, 50), units=None):
        ''' Creates a text summary of the field

        Parameters
        ------------
        percentiles: ndarray (optinal)
            Field percentiles to be printed. Default: (99.9, 99, 95)
        focality_cutoffs: ndarray (optional)
            Cuttofs for focality calculations. Default: (75, 50)
        units: str or None
            Name of field units or automatically determine from name
        '''
        if units is None:
            if self.field_name in ['E', 'magnE', 'D', 'g']:
                units = 'V/m'
            elif self.field_name in ['J', 'magnJ']:
                units = 'A/m²'
            elif self.field_name == 'v':
                units = 'V'
            else:
                units = ''
        if units:
            units = ' ' + units

        if 2 in self.mesh.elm.elm_type:
            units_mesh = ' mm²'
        elif 4 in self.mesh.elm.elm_type:
            units_mesh = ' mm³'
        if np.all(np.isin([2, 4], self.mesh.elm.elm_type)):
            warnings.warn('Field summary in meshes with'
                          ' triangles and tetrahedra can give misleading results')
        norm = self._norm()
        weights = self._weights()
        mean_norm = np.sum(norm * weights)/np.sum(weights)
        prc = self.get_percentiles(percentiles)
        focality = self.get_focality(focality_cutoffs, percentiles[-1])
        string = f'Field: {self.field_name}\n'
        string += 'Peak Values:\n'
        n_spaces = len(f'{percentiles[0]:.2e}{units} ') - 5
        string += (n_spaces*' ' + '|').join(f'{p:.1f}%' for p in percentiles) + '\n'
        string += ' |'.join(f'{p:.2e}{units}' for p in prc) + '\n'
        string += 'Focality:\n'
        n_spaces = len(f'{focality[0]:.2e}{units_mesh} ') - 5
        string += (n_spaces*' ' + '|').join(f'{fc:.1f}%' for fc in focality_cutoffs) + '\n'
        string += ' |'.join(f'{fv:.2e}{units_mesh}' for fv in focality) + '\n'

        string += f'Mean Field:\n{mean_norm:.2f}{units}'
        return string

    @property
    def indexing_nr(self):
        ''' Nodes or element numbers '''
        raise Exception('indexing_nr is not defined')

    def __eq__(self, other):
        try:
            return self.__dict__ == other.__dict__
        except AttributeError:
            return False

    def __getitem__(self, index):
        return _getitem_one_indexed(self.value, index)

    def __setitem__(self, index, item):
        index = _fix_indexing_one(index)
        self.value[index] = item

    def __mul__(self, other):
        cp = copy.copy(self)
        cp.value = self.value.__mul__(other)
        return cp

    def __neg__(self):
        cp = copy.copy(self)
        cp.value = self.value.__neg__()
        return cp

    def __sub__(self, other):
        cp = copy.copy(self)
        cp.value = self.value.__sub__(other)
        return cp

    def __add__(self, other):
        cp = copy.copy(self)
        cp.value = self.value.__add__(other)
        return cp

    def __str__(self):
        return self.field_name + '\n' + self.value.__str__()

    def __truediv__(self, other):
        cp = copy.copy(self)
        cp.value = self.value.__truediv__(other)
        return cp

    def __div__(self, other):
        cp = copy.copy(self)
        cp.value = self.value.__div__(other)
        return cp

    def __pow__(self, p):
        cp = copy.copy(self)
        cp.value = self.value.__pow__(p)
        return cp

    def _test_msh(self):
        if self.mesh is None:
            raise ValueError('Cannot evaluate function if .mesh property is not '
                             'assigned')


class ElementData(Data):
    """
    Data (scalar, vector or tensor) defined in mesh elements.

    Parameters
    -----------------------
    value: ndarray
        Value of field in elements. Should have the shape
            - (n_elm,) or (n_elm, 1) for scalar fields
            - (n_elm, 3) for vector fields
            - (n_elm, 9) for tensors

    field_name: str (optional)
        name of field. Default: ''

    mesh: simnibs.msh.Msh (optional)
        Mesh structure where the field is defined. Required for many methods


    Attributes
    --------------------------
    value: ndarray
        Value of field in elements
    field_name: str
        name of field
    elm_number: ndarray
        index of elements
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)
    """

    def __init__(self, value, name='', mesh=None):
        Data.__init__(self, value=value, name=name, mesh=mesh)

    @property
    def elm_number(self):
        '''Element numbers (1, ..., nr)'''
        return np.arange(1, self.nr + 1, dtype='int32')

    @property
    def indexing_nr(self):
        '''Same as elm_number'''
        return self.elm_number

    def elm_data2node_data(self):
        """Transforms an ElementData field into a NodeData field using Superconvergent
        patch recovery
        For volumetric data. Will not work well for discontinuous fields (like E, if
        several tissues are used)

        Returns
        ----------------------
        simnibs.NodeData
            Structure with NodeData

        References
        -------------------
            Zienkiewicz, Olgierd Cecil, and Jian
            Zhong Zhu. "The superconvergent patch recovery and a posteriori error
            estimates. Part 1: The recovery technique." International Journal for
            Numerical Methods in Engineering 33.7 (1992): 1331-1364.

        """
        self._test_msh()
        msh = self.mesh
        if self.nr != msh.elm.nr:
            raise ValueError("The number of data points in the data "
                             "structure should be equal to the number of elements in the mesh")
        nd = np.zeros((msh.nodes.nr, self.nr_comp))

        if len(msh.elm.tetrahedra) == 0:
            raise ValueError("Can only transform volume data")

        value = np.atleast_2d(self[msh.elm.tetrahedra])
        if value.shape[0] < value.shape[1]:
            value = value.T

        # get all nodes used in tetrahedra, creates the NodeData structure
        uq = np.unique(msh.elm[msh.elm.tetrahedra])
        nd = NodeData(np.zeros((len(uq), self.nr_comp)), self.field_name, mesh=msh)

        # Get the point in the outside surface
        points_outside = np.unique(msh.elm.get_outside_faces())
        outside_points_mask = np.in1d(msh.elm[msh.elm.tetrahedra],
                                      points_outside).reshape(-1, 4)
        masked_th_nodes = np.copy(msh.elm[msh.elm.tetrahedra])
        masked_th_nodes[outside_points_mask] = -1

        # Calculates the quantities needed for the superconvergent patch recovery
        uq_in, th_nodes = np.unique(masked_th_nodes, return_inverse=True)
        baricenters = msh.elements_baricenters()[msh.elm.tetrahedra]
        volumes = msh.elements_volumes_and_areas()[msh.elm.tetrahedra]
        baricenters = np.hstack([np.ones((baricenters.shape[0], 1)), baricenters])

        A = np.empty((len(uq_in), 4, 4))
        b = np.empty((len(uq_in), 4, self.nr_comp), self.value.dtype)
        for i in range(4):
            for j in range(i, 4):
                A[:, i, j] = np.bincount(th_nodes.reshape(-1),
                                         np.repeat(baricenters[:, i], 4) *
                                         np.repeat(baricenters[:, j], 4))
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 3, 0] = A[:, 0, 3]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 3, 1] = A[:, 1, 3]
        A[:, 3, 2] = A[:, 2, 3]

        for j in range(self.nr_comp):
            for i in range(4):
                b[:, i, j] = np.bincount(th_nodes.reshape(-1),
                                         np.repeat(baricenters[:, i], 4) *
                                         np.repeat(value[:, j], 4))

        try:
            a = np.linalg.solve(A[uq_in != -1], b[uq_in != -1])
        except np.linalg.LinAlgError:
            # The mesh probably contains "duplicate" nodes
            # TODO fix the mesh instead - then this shouldn't be necessary
            used_nodes = uq_in[uq_in != -1]-1
            the_nodes = msh.nodes.node_coord[used_nodes]

            S = np.linalg.svd(A[uq_in != -1], compute_uv=False)
            to_interp = S[:, 1:].sum(1) < 1e-6
            to_compute = ~to_interp
            tree = scipy.spatial.cKDTree(the_nodes[to_compute])
            di, ix = tree.query(the_nodes[to_interp])

            warnings.warn(
                ("NumPy raised a `LinAlgError` interpolating to certain nodes "
                 f"(mean coordinate {the_nodes[to_interp].mean(0)}, "
                 f"standard deviation {the_nodes[to_interp].std(0)}). "
                 f"Using nearest neighbor interpolation at {to_interp.sum()} "
                 f"nodes (maximum distance is {di.max():.5f})."
                )
            )

            a = np.zeros(b[uq_in != -1].shape)
            a[to_compute] = np.linalg.solve(A[uq_in != -1][to_compute], b[uq_in != -1][to_compute])
            a[to_interp] = a[to_compute][ix]

        p = np.hstack([np.ones((np.sum(uq_in != -1), 1)), msh.nodes[uq_in[uq_in != -1]]])
        f = np.einsum('ij, ijk -> ik', p, a)
        nd[uq_in[uq_in != -1]] = f

        # Assigns the average value to the points in the outside surface
        masked_th_nodes = np.copy(msh.elm[msh.elm.tetrahedra])
        masked_th_nodes[~outside_points_mask] = -1
        uq_out, th_nodes_out = np.unique(masked_th_nodes,
                                         return_inverse=True)
        sum_vals = np.empty((len(uq_out), self.nr_comp), self.value.dtype)
        for j in range(self.nr_comp):
            sum_vals[:, j] = np.bincount(th_nodes_out.reshape(-1),
                                         np.repeat(value[:, j], 4) *
                                         np.repeat(volumes, 4))

        sum_vols = np.bincount(th_nodes_out.reshape(-1), np.repeat(volumes, 4))

        nd[uq_out[uq_out != -1]] = (sum_vals/sum_vols[:, None])[uq_out != -1]

        nd.value = np.squeeze(nd.value)
        return nd

    def as_nodedata(self):
        ''' Converts the current ElementData instance to NodaData
        For more information see the elm_data2node_data method
        '''
        return self.elm_data2node_data()

    def interpolate_to_grid(self, n_voxels, affine, method='linear', continuous=False):
        ''' Interpolates the ElementData into a grid.
            finds which tetrahedra contais the given voxel and
            assign the value of the tetrahedra to the voxel.

        Parameters
        -----------
        n_voxels: list or tuple
            number of voxels in x, y, and z directions
        affine: ndarray
            A 4x4 matrix specifying the transformation from voxels to xyz
        method: {'assign' or 'linear'} (Optional)
            If 'assign', gives to each voxel the value of the element that contains
            it. If linear, first assign fields to nodes, and then perform
            baricentric interpolatiom. Default: linear
        continuous: bool
            Wether fields is continuous across tissue boundaries. Changes the
            behaviour of the function only if method == 'linear'. Default: False

        Returns
        --------
        image: ndarray
            An (n_voxels[0], n_voxels[1], n_voxels[2], nr_comp) matrix with
            interpolated values. If nr_comp == 1, the last dimension is squeezed out
        '''

        msh = self.mesh
        self._test_msh()
        if self.nr != msh.elm.nr:
            raise ValueError('Invalid Mesh! Mesh should have the same number of elements'
                             'as the number of data points')
        if len(n_voxels) != 3:
            raise ValueError('n_voxels should have length = 3')
        if affine.shape != (4, 4):
            raise ValueError('Affine should be a 4x4 matrix')
        if len(msh.elm.tetrahedra) == 0:
            raise InvalidMeshError('Mesh has no volume elements')

        msh_th = msh.crop_mesh(elm_type=4)
        msh_th.elmdata = []
        msh_th.nodedata = []
        v = np.atleast_2d(self.value)
        if v.shape[0] < v.shape[1]:
            v = v.T
        v = v[msh.elm.elm_type == 4]

        if method == 'assign':
            nd = np.hstack([msh_th.nodes.node_coord, np.ones((msh_th.nodes.nr, 1))])
            inv_affine = np.linalg.inv(affine)
            nd = inv_affine.dot(nd.T).T[:, :3]

            # initialize image
            image = np.zeros([n_voxels[0], n_voxels[1], n_voxels[2], self.nr_comp], dtype=float)
            field = v.astype(float)
            image = cython_msh.interp_grid(
                np.array(n_voxels, dtype=int), field, nd.astype(float),
                (msh_th.elm.node_number_list - 1).astype(int))
            image = image.astype(self.value.dtype)
            if self.nr_comp == 1:
                image = np.squeeze(image, axis=3)

        elif method == 'linear':
            if continuous:
                nd = self.elm_data2node_data()
                image = nd.interpolate_to_grid(n_voxels, affine)

            else:
                if self.nr_comp != 1:
                    image = np.zeros(list(n_voxels) + [self.nr_comp], dtype=float)
                else:
                    image = np.zeros(list(n_voxels), dtype=float)
                # Interpolate each tag separetelly
                tags = np.unique(msh_th.elm.tag1)
                msh_th.elmdata = [ElementData(v, mesh=msh_th)]
                for t in tags:
                    msh_tag = msh_th.crop_mesh(tags=t)
                    nd = msh_tag.elmdata[0].elm_data2node_data()
                    image += nd.interpolate_to_grid(n_voxels, affine)
                    del msh_tag
                    del nd
                    gc.collect()
        else:
            raise ValueError('Invalid interpolation method!')

        del msh_th
        del v

        gc.collect()
        return image


    def norm(self, ord=2):
        ''' Calculate the norm (magnitude) of the field

        Parameters
        ------------
        ord: float
            Order of norm. Default: 2 (euclidian norm, i.e. magnitude)

        Returns
        -----------
        norm: NodeData
            NodeData field with the norm the field
        '''
        if len(self.value.shape) == 1:
            ed = ElementData(np.abs(self.value),
                             name='magn' + self.field_name,
                             mesh=self.mesh)
        else:
            ed = ElementData(np.linalg.norm(self.value, axis=1, ord=ord),
                             name='magn' + self.field_name,
                             mesh=self.mesh)
        return ed

  
    def append_to_mesh(self, fn, mode='binary'):
        """Appends this ElementData fields to a file

        Parameters
        ----------
            fn: str
                file name
            mode: binary or ascii
                mode in which to write
        """
        with open(fn, 'ab') as f:
            f.write(b'$ElementData\n')
            # string tags
            f.write((str(1) + '\n').encode('ascii'))
            f.write(('"' + self.field_name + '"\n').encode('ascii'))

            f.write((str(1) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            f.write((str(4) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))
            f.write((str(self.nr_comp) + '\n').encode('ascii'))
            f.write((str(self.nr) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            if mode == 'ascii':
                for ii in range(self.nr):
                    f.write((str(self.elm_number[ii]) + ' ' +
                             str(self.value[ii]).translate(None, '[](),') +
                             '\n').encode('ascii'))

            elif mode == 'binary':

                elm_number = self.elm_number.astype('int32')
                value = self.value.astype('float64')
                try:
                    value.shape[1]
                except IndexError:
                    value = value[:, np.newaxis]
                m = elm_number[:, np.newaxis]
                for i in range(self.nr_comp):
                    m = np.concatenate((m,
                                        value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                       axis=1)
                f.write(m.tobytes())

            else:
                raise IOError("invalid mode:", mode)

            f.write(b'$EndElementData\n')

    def write(self, fn):
        """Writes this ElementData fields to a file with field information only
        This file needs to be merged with a mesh for visualization

        Parameters
        ---------
            fn: str
                file name
            mode: binary or ascii
                mode in which to write
        """
        with open(fn, 'wb') as f:
            f.write(b'$MeshFormat\n2.2 1 8\n')
            f.write(struct.pack('i', 1))
            f.write(b'\n$EndMeshFormat\n')
            f.write(b'$ElementData\n')
            # string tags
            f.write((str(1) + '\n').encode('ascii'))
            f.write(('"' + self.field_name + '"\n').encode('ascii'))

            f.write((str(1) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            f.write((str(4) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))
            f.write((str(self.nr_comp) + '\n').encode('ascii'))
            f.write((str(self.nr) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            elm_number = self.elm_number.astype('int32')
            value = self.value.astype('float64')
            try:
                value.shape[1]
            except IndexError:
                value = value[:, np.newaxis]
            m = elm_number[:, np.newaxis]
            for i in range(self.nr_comp):
                m = np.concatenate((m,
                                    value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                   axis=1)
            f.write(m.tobytes())

            f.write(b'$EndElementData\n')


class NodeData(Data):
    """
    Data (scalar, vector or tensor) defined in mesh nodes.

    Parameters
    ----------
    value: ndarray
        Value of field in nodes. Should have the shape
            - (n_nodes,) or (n_nodes, 1) for scalar fields
            - (n_nodes, 3) for vector fields
            - (n_nodes, 9) for tensors

    field_name: str (optional)
        name of field. Default: ''

    mesh: simnibs.msh.Msh (optinal)
        Mesh where the field is defined. Required for many methods

    Attributes
    --------------------------
    value: ndarray
        Value of field in elements
    field_name: str
        name of field
    node_number: ndarray
        index of elements
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)
    """

    def __init__(self, value, name='', mesh=None):
        Data.__init__(self, value=value, name=name, mesh=mesh)

    @property
    def node_number(self):
        ''' Node numbers (1, ..., nr)'''
        return np.arange(1, self.nr + 1, dtype='int32')

    @property
    def indexing_nr(self):
        ''' Same as node_numbers'''
        return self.node_number

    def as_nodedata(self):
        return self

    def node_data2elm_data(self):
        """Transforms an ElementData field into a NodeData field
        the value in the element is the average of the value in the nodes

        Returns
        --------
        simnibs.ElementData
            structure with field value interpolated at element centers

        """
        self._test_msh()
        msh = self.mesh
        if (self.nr != msh.nodes.nr):
            raise ValueError(
                "The number of data points in the data structure should be"
                "equal to the number of elements in the mesh")

        triangles = np.where(msh.elm.elm_type == 2)[0]
        tetrahedra = np.where(msh.elm.elm_type == 4)[0]

        if self.nr_comp == 1:
            elm_data = np.zeros((msh.elm.nr,), dtype=float)
        else:
            elm_data = np.zeros((msh.elm.nr, self.nr_comp), dtype=float)

        if len(triangles) > 0:
            elm_data[triangles] = \
                np.average(self.value[msh.elm.node_number_list[
                           triangles, :3] - 1], axis=1)
        if len(tetrahedra) > 0:
            elm_data[tetrahedra] = \
                np.average(self.value[msh.elm.node_number_list[
                           tetrahedra, :4] - 1], axis=1)

        return ElementData(elm_data, self.field_name, mesh=msh)

    def gradient(self):
        ''' Calculates the gradient of a field in the middle of the tetrahedra

        Parameters
        ------
        mesh: simnibs.Msh
            A mesh with the geometrical information

        Returns
        -----
        grad: simnibs.ElementData
            An ElementData field with gradient in the middle of each tetrahedra
        '''
        self._test_msh()
        mesh = self.mesh
        if self.nr_comp != 1:
            raise ValueError('can only take gradient of scalar fields')
        if mesh.nodes.nr != self.nr:
            raise ValueError('mesh must have the same number of nodes as the NodeData')

        # Tetrahedra gradients
        elm_node_coords = mesh.nodes[mesh.elm[mesh.elm.tetrahedra]]

        tetra_matrices = elm_node_coords[:, 1:4, :] - \
            elm_node_coords[:, 0, :][:, None]

        dif_between_tetra_nodes = self[mesh.elm[mesh.elm.tetrahedra][:, 1:4]] - \
            self[mesh.elm[mesh.elm.tetrahedra][:, 0]][:, None]

        th_grad = np.linalg.solve(tetra_matrices, dif_between_tetra_nodes)

        gradient = np.zeros((mesh.elm.nr, 3), dtype=float)
        gradient[mesh.elm.elm_type == 4] = th_grad
        gradient[mesh.elm.elm_type == 2] = 0.

        return ElementData(gradient, 'grad_' + self.field_name, mesh)


    def interpolate_to_grid(self, n_voxels, affine, **kwargs):
        ''' Interpolates the NodeData into a grid.
            finds which tetrahedra contais the given voxel and
            performs linear interpolation inside the voxel

        The kwargs is ony to have the same interface as the ElementData version

        Parameters
        ------
            n_voxels: list or tuple
                number of voxels in x, y, and z directions
            affine: ndarray
                A 4x4 matrix specifying the transformation from voxels to xyz
        Returns
        ----
            image: ndarray
                An (n_voxels[0], n_voxels[1], n_voxels[2], nr_comp) matrix with
                interpolated values. If nr_comp == 1, the last dimension is squeezed out
        '''

        msh = self.mesh
        self._test_msh()
        if len(n_voxels) != 3:
            raise ValueError('n_voxels should have length = 3')
        if affine.shape != (4, 4):
            raise ValueError('Affine should be a 4x4 matrix')
        if len(msh.elm.tetrahedra) == 0:
            raise InvalidMeshError('Mesh has no volume elements')

        v = np.atleast_2d(self.value)
        if v.shape[0] < v.shape[1]:
            v = v.T
        msh_th = msh.crop_mesh(elm_type=4)
        inv_affine = np.linalg.inv(affine)
        nd = np.hstack([msh_th.nodes.node_coord, np.ones((msh_th.nodes.nr, 1))])
        nd = inv_affine.dot(nd.T).T[:, :3]

        # initialize image
        image = np.zeros([n_voxels[0], n_voxels[1], n_voxels[2], self.nr_comp], dtype=float)
        field = v.astype(float)
        if v.shape[0] != msh_th.nodes.nr:
            raise ValueError('Number of data points in the structure does not match '
                             'the number of nodes present in the volume-only mesh')
        image = cython_msh.interp_grid(
            np.array(n_voxels, dtype=int), field, nd,
            msh_th.elm.node_number_list - 1)
        image = image.astype(self.value.dtype)
        if self.nr_comp == 1:
            image = np.squeeze(image, axis=3)
        del nd
        del msh_th
        del field
        gc.collect()
        return image

    def norm(self, ord=2):
        ''' Calculate the norm (magnitude) of the field

        Parameters
        ------------
        ord: float
            Order of norm. Default: 2 (euclidian norm, i.e. magnitude)

        Returns
        ---------
        norm: NodeData
            NodeData field with the norm the field
        '''
        if len(self.value.shape) == 1:
            nd = NodeData(np.abs(self.value),
                          name='magn' + self.field_name,
                          mesh=self.mesh)
        else:
            nd = NodeData(np.linalg.norm(self.value, axis=1, ord=ord),
                          name='magn' + self.field_name,
                          mesh=self.mesh)
        return nd

    def normal(self, fill=np.nan):
        ''' Calculate the normal component of the field in the mesh surfaces

        Parameters
        -----------
        fill: float (optional)
            Value to be used when node is not in surface (Default: NaN)

        Returns
        ---------
        normal: NodeData
            NodeData field with the normal the field where a surface is defined and the
            fill value where it's not
        '''
        self._test_msh()
        assert self.nr_comp == 3, 'Normals are only defined for vector fields'
        normal = NodeData(fill * np.ones(self.nr, dtype=self.value.dtype),
                          name='normal' + self.field_name,
                          mesh=self.mesh)
        nodes_in_surface = self.mesh.elm[self.mesh.elm.triangles, :3]
        nodes_in_surface = np.unique(nodes_in_surface)
        nodes_normals = self.mesh.nodes_normals()
        normal[nodes_in_surface] = np.sum(self[nodes_in_surface] * nodes_normals[nodes_in_surface], axis=1)
        return normal

    def angle(self, fill=np.nan):
        ''' Calculate the angle between the field and the surface normal

        Parameters
        -------------
        fill: float (optional)
            Value to be used when node is not in surface

        Returns
        -----
        angle: NodeData
            NodeData field with the angles the field where a surface is defined and the
            fill value where it's not
        '''
        self._test_msh()
        assert self.nr_comp == 3, 'angles are only defined for vector fields'
        angle = NodeData(fill * np.ones(self.nr, dtype=self.value.dtype),
                          name='angle' + self.field_name,
                          mesh=self.mesh)

        nodes_in_surface = self.mesh.elm[self.mesh.elm.triangles, :3]
        nodes_in_surface = np.unique(nodes_in_surface)
        nodes_normals = self.mesh.nodes_normals()
        normal = np.sum(self[nodes_in_surface] * nodes_normals[nodes_in_surface], axis=1)
        norm = np.linalg.norm(self[nodes_in_surface], axis=1)
        tan = np.sqrt(norm ** 2 - normal ** 2)
        #angle[nodes_in_surface] = np.arccos(normal/norm)
        angle[nodes_in_surface] = np.arctan2(tan, normal)
        return angle


    def tangent(self, fill=np.nan):
        ''' Calculate the tangent component of the field in the surfaces

        Parameters
        -----------
        fill: float (optional)
            Value to be used when node is not in surface

        Returns
        -----
        tangent: NodeData
            NodeData field with the tangent component of the field where a surface is defined and the
            fill value where it's not
        '''
        self._test_msh()
        assert self.nr_comp == 3, 'angles are only defined for vector fields'
        tangent = NodeData(fill * np.ones(self.nr, dtype=self.value.dtype),
                          name='tangent' + self.field_name,
                          mesh=self.mesh)
        nodes_in_surface = self.mesh.elm[self.mesh.elm.triangles, :3]
        nodes_in_surface = np.unique(nodes_in_surface)
        nodes_normals = self.mesh.nodes_normals()
        normal = np.sum(self[nodes_in_surface] * nodes_normals[nodes_in_surface], axis=1)
        norm = np.linalg.norm(self[nodes_in_surface], axis=1)
        tangent[nodes_in_surface] = np.sqrt(norm ** 2 - normal ** 2)
        return tangent



    def append_to_mesh(self, fn, mode='binary', mmg_fix=False):
        """Appends this NodeData fields to a file

        Parameters
        ------------
            fn: str
                file name
            mode: binary or ascii
                mode in which to write
        """
        with open(fn, 'ab') as f:
            f.write(b'$NodeData\n')
            # string tags
            f.write((str(1) + '\n').encode('ascii'))
            f.write(('"' + self.field_name + '"\n').encode('ascii'))

            f.write((str(1) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            if mmg_fix:
                f.write((str(3) + '\n').encode('ascii'))
                f.write((str(0) + '\n').encode('ascii'))
                f.write((str(self.nr_comp) + '\n').encode('ascii'))
                f.write((str(self.nr) + '\n').encode('ascii'))
            else:
                f.write((str(4) + '\n').encode('ascii'))
                f.write((str(0) + '\n').encode('ascii'))
                f.write((str(self.nr_comp) + '\n').encode('ascii'))
                f.write((str(self.nr) + '\n').encode('ascii'))
                f.write((str(0) + '\n').encode('ascii'))

            if mode == 'ascii':
                for ii in range(self.nr):
                    f.write((
                        str(self.node_number[ii]) + ' ' +
                        str(self.value[ii]).translate(None, '[](),') +
                        '\n').encode('ascii'))

            elif mode == 'binary':
                value = self.value.astype('float64')
                try:
                    value.shape[1]
                except IndexError:
                    value = value[:, np.newaxis]

                m = self.node_number[:, np.newaxis].astype('int32')
                for i in range(self.nr_comp):
                    m = np.concatenate((m,
                                        value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                       axis=1)

                f.write(m.tobytes())
            else:
                raise IOError("invalid mode:", mode)

            f.write(b'$EndNodeData\n')


    def write(self, fn):
        """Writes this NodeData field to a file with field information only
        This file needs to be merged with a mesh for visualization

        Parameters
        -----------
            fn: str
                file name
        """
        with open(fn, 'wb') as f:
            f.write(b'$MeshFormat\n2.2 1 8\n')
            f.write(struct.pack('i', 1))
            f.write(b'\n$EndMeshFormat\n')
            f.write(b'$NodeData\n')
            f.write((str(1) + '\n').encode('ascii'))
            f.write(('"' + self.field_name + '"\n').encode('ascii'))

            f.write((str(1) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            f.write((str(4) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))
            f.write((str(self.nr_comp) + '\n').encode('ascii'))
            f.write((str(self.nr) + '\n').encode('ascii'))
            f.write((str(0) + '\n').encode('ascii'))

            value = self.value.astype('float64')
            try:
                value.shape[1]
            except IndexError:
                value = value[:, np.newaxis]

            m = self.node_number[:, np.newaxis].astype('int32')
            for i in range(self.nr_comp):
                m = np.concatenate((m,
                                    value[:, i].astype('float64').view('int32').reshape(-1, 2)),
                                   axis=1)

            f.write(m.tobytes())

            f.write(b'$EndNodeData\n')


def read_msh(fn, m=None, skip_data=False):
    ''' Reads a gmsh '.msh' file

    Parameters
    ------------
    fn: str
        File name
    m: simnibs.msh.Msh (optional)
        Mesh structure to be overwritten. If unset, will create a new structure
    skip_data: bool (optional)
        If True, reading of NodeData and ElementData will be skipped (Default: False)

    Returns
    --------
    msh: simnibs.msh.Msh
        Mesh structure
    '''
    if m is None:
        m = Msh()

    fn = os.path.expanduser(fn)

    if not os.path.isfile(fn):
        raise IOError(fn + ' not found')

    version_number = _find_mesh_version(fn)
    if version_number == 2:
        m = _read_msh_2(fn, m, skip_data)

    elif version_number == 4:
        m = _read_msh_4(fn, m, skip_data)

    else:
        raise IOError('Unrecgnized Mesh file version : {}'.format(version_number))

    return m


def _find_mesh_version(fn):
    if not os.path.isfile(fn):
        raise IOError(fn + ' not found')

    # file open
    with open(fn, 'rb') as f:
        # check 1st line
        first_line = f.readline().decode()
        if first_line != '$MeshFormat\n':
            raise IOError(fn, "must start with $MeshFormat")

        # parse 2nd line
        version_number, file_type, data_size = f.readline().decode().split()
        version_number = int(version_number[0])
        file_type = int(file_type)
        data_size = int(data_size)
    return version_number


def _read_msh_2(fn, m, skip_data=False):
    m.fn = fn

    # file open
    with open(fn, 'rb') as f:
        # check 1st line
        first_line = f.readline()
        if first_line != b'$MeshFormat\n':
            raise IOError(fn, "must start with $MeshFormat")

        # parse 2nd line
        version_number, file_type, data_size = f.readline().decode().strip().split()
        version_number = int(version_number[0])
        file_type = int(file_type)
        data_size = int(data_size)

        if version_number != 2:
            raise IOError("Can only handle v2 meshes")

        if file_type == 1:
            binary = True
        elif file_type == 0:
            binary = False
        else:
            raise IOError("File_type not recognized: {0}".format(file_type))

        if data_size != 8:
            raise IOError(
                "data_size should be double (8), i'm reading: {0}".format(data_size))

        # read next byte, if binary, to check for endianness
        if binary:
            endianness = struct.unpack('i', f.readline()[:4])[0]
        else:
            endianness = 1

        if endianness != 1:
            raise IOError("endianness is not 1, is the endian order wrong?")

        # read 3rd line
        if f.readline() != b'$EndMeshFormat\n':
            raise IOError(fn + " expected $EndMeshFormat")

        # read 4th line
        if f.readline() != b'$Nodes\n':
            raise IOError(fn + " expected $Nodes")

        # read 5th line with number of nodes
        try:
            node_nr = int(f.readline().decode().strip())
        except:
            raise IOError(fn + " something wrong with Line 5 - should be a number")

        # read all nodes
        if binary:
            # 0.02s to read binary.msh
            dt = np.dtype([
                ('id', np.int32),
                ('coord', np.float64, 3)])

            temp = np.fromfile(f, dtype=dt, count=node_nr)
            node_number = np.copy(temp['id'])
            node_coord = np.copy(temp['coord'])

            # sometimes there's a line feed here, sometimes there is not...
            LF_byte = f.read(1)  # read \n
            if not ord(LF_byte) == 10:
                # if there was not a LF, go back 1 byte from the current file
                # position
                f.seek(-1, 1)

        else:
            # nodes has 4 entries: [node_ID x y z]
            node_number = np.empty(node_nr, dtype='int32')
            # array Nx3 for (x,y,z) coordinates of the nodes
            node_coord = np.empty(3 * node_nr, dtype='float64')
            for ii in range(node_nr):
                line = f.readline().decode().strip().split()
                node_number[ii] = line[0]
                # it's faster to use a linear array and than reshape
                node_coord[3 * ii] = line[1]
                node_coord[3 * ii + 1] = line[2]
                node_coord[3 * ii + 2] = line[3]
            node_coord = node_coord.reshape((node_nr, 3))

        if not np.all(node_number == np.arange(1, node_nr + 1)):
            warnings.warn("Mesh file with discontinuos nodes, things can fail"
                          " unexpectedly")
        m.nodes.node_coord = node_coord

        if f.readline() != b'$EndNodes\n':
            raise IOError(fn + " expected $EndNodes after reading " +
                          str(node_nr) + " nodes")

        # read all elements
        if f.readline() != b'$Elements\n':
            raise IOError(fn, "expected line with $Elements")

        try:
            elm_nr = int(f.readline().decode().strip())
        except:
            raise IOError(
                fn + " something wrong when reading number of elements (line after $Elements)"
                "- should be a number")

        if binary:
            current_element = 0

            elm_number = np.empty(elm_nr, dtype='int32')
            m.elm.elm_type = np.empty(elm_nr, dtype='int32')
            m.elm.tag1 = np.empty(elm_nr, dtype='int32')
            m.elm.tag2 = np.empty(elm_nr, dtype='int32')
            m.elm.node_number_list = -np.ones((elm_nr, 4), dtype='int32')
            read = np.ones(elm_nr, dtype=bool)

            nr_nodes_elm = [None, 2, 3, 4, 4, 8, 6, 5, 3, 6, 9,
                            10, 27, 18, 14, 1, 8, 20, 15, 13]
            while current_element < elm_nr:
                elm_type, nr, _ = np.fromfile(f, 'int32', 3)
                if elm_type == 1:
                    tmp = np.fromfile(f, 'int32', nr * 5).reshape(-1, 5)

                    m.elm.elm_type[current_element:current_element+nr] = \
                        1 * np.ones(nr, 'int32')
                    elm_number[current_element:current_element+nr] = tmp[:, 0]
                    m.elm.tag1[current_element:current_element+nr] = tmp[:, 1]
                    m.elm.tag2[current_element:current_element+nr] = tmp[:, 2]
                    m.elm.node_number_list[current_element:current_element+nr, :2] = tmp[:, 3:]
                    read[current_element:current_element+nr] = 1

                elif elm_type == 2:
                    tmp = np.fromfile(f, 'int32', nr * 6).reshape(-1, 6)

                    m.elm.elm_type[current_element:current_element+nr] = \
                        2 * np.ones(nr, 'int32')
                    elm_number[current_element:current_element+nr] = tmp[:, 0]
                    m.elm.tag1[current_element:current_element+nr] = tmp[:, 1]
                    m.elm.tag2[current_element:current_element+nr] = tmp[:, 2]
                    m.elm.node_number_list[current_element:current_element+nr, :3] = tmp[:, 3:]
                    read[current_element:current_element+nr] = 1

                elif elm_type == 4:
                    tmp = np.fromfile(f, 'int32', nr * 7).reshape(-1, 7)

                    m.elm.elm_type[current_element:current_element+nr] = \
                        4 * np.ones(nr, 'int32')
                    elm_number[current_element:current_element+nr] = tmp[:, 0]
                    m.elm.tag1[current_element:current_element+nr] = tmp[:, 1]
                    m.elm.tag2[current_element:current_element+nr] = tmp[:, 2]
                    m.elm.node_number_list[current_element:current_element+nr] = tmp[:, 3:]
                    read[current_element:current_element+nr] = 1

                elif elm_type == 15:
                    tmp = np.fromfile(f, 'int32', nr * 4).reshape(-1, 4)

                    m.elm.elm_type[current_element:current_element+nr] = \
                        15 * np.ones(nr, 'int32')
                    elm_number[current_element:current_element+nr] = tmp[:, 0]
                    m.elm.tag1[current_element:current_element+nr] = tmp[:, 1]
                    m.elm.tag2[current_element:current_element+nr] = tmp[:, 2]
                    m.elm.node_number_list[current_element:current_element+nr, :1] = tmp[:, 3:]
                    read[current_element:current_element+nr] = 1
                else:
                    warnings.warn('element of type {0} '
                                  'cannot be read, ignoring it'.format(elm_type))
                    np.fromfile(f, 'int32', nr * (3 + nr_nodes_elm[elm_type]))
                    read[current_element:current_element+nr] = 0
                current_element += nr

            elm_number = elm_number[read]
            m.elm.elm_type = m.elm.elm_type[read]
            m.elm.tag1 = m.elm.tag1[read]
            m.elm.tag2 = m.elm.tag2[read]
            m.elm.node_number_list = m.elm.node_number_list[read]

        else:

            elm_number = np.empty(elm_nr, dtype='int32')
            m.elm.elm_type = np.empty(elm_nr, dtype='int32')
            m.elm.tag1 = np.empty(elm_nr, dtype='int32')
            m.elm.tag2 = np.empty(elm_nr, dtype='int32')
            m.elm.node_number_list = -np.ones((elm_nr, 4), dtype='int32')
            read = np.ones(elm_nr, dtype=bool)

            for ii in range(elm_nr):
                line = f.readline().decode().strip().split()
                if line[1] == '1':
                    elm_number[ii] = line[0]
                    m.elm.elm_type[ii] = line[1]
                    m.elm.tag1[ii] = line[3]
                    m.elm.tag2[ii] = line[4]
                    m.elm.node_number_list[ii, :2] = [int(i) for i in line[5:]]
                elif line[1] == '2':
                    elm_number[ii] = line[0]
                    m.elm.elm_type[ii] = line[1]
                    m.elm.tag1[ii] = line[3]
                    m.elm.tag2[ii] = line[4]
                    m.elm.node_number_list[ii, :3] = [int(i) for i in line[5:]]
                elif line[1] == '4':
                    elm_number[ii] = line[0]
                    m.elm.elm_type[ii] = line[1]
                    m.elm.tag1[ii] = line[3]
                    m.elm.tag2[ii] = line[4]
                    m.elm.node_number_list[ii] = [int(i) for i in line[5:]]
                elif line[1] == '15':
                    elm_number[ii] = line[0]
                    m.elm.elm_type[ii] = line[1]
                    m.elm.tag1[ii] = line[3]
                    m.elm.tag2[ii] = line[4]
                    m.elm.node_number_list[ii, :1] = [int(i) for i in line[5:]]
                else:
                    read[ii] = 0
                    warnings.warn('element of type {0} '
                                  'cannot be read, ignoring it'.format(line[1]))

            elm_number = elm_number[read]
            m.elm.elm_type = m.elm.elm_type[read]
            m.elm.tag1 = m.elm.tag1[read]
            m.elm.tag2 = m.elm.tag2[read]
            m.elm.node_number_list = m.elm.node_number_list[read]

        elm_nr_changed = False
        if not np.all(elm_number == np.arange(1, m.elm.nr + 1)):
            warnings.warn('Changing element numbering')
            elm_nr_changed = True
            elm_number = np.arange(1, m.elm.nr + 1)

        line = f.readline()
        if b'$EndElements' not in line:
            line = f.readline()
            if b'$EndElements' not in line:
                raise IOError(fn + " expected $EndElements after reading " +
                              str(m.elm.nr) + " elements. Read " + line)

        # read the header in the beginning of a data section
        def parse_Data():
            section = f.readline()
            if section == b'':
                return 'EOF', '', 0, 0
            # string tags
            number_of_string_tags = int(f.readline().decode('ascii'))
            assert number_of_string_tags == 1, "Invalid Mesh File: invalid number of string tags"
            name = f.readline().decode('ascii').strip().strip('"')
            # real tags
            number_of_real_tags = int(f.readline().decode('ascii'))
            assert number_of_real_tags == 1, "Invalid Mesh File: invalid number of real tags"
            f.readline()
            # integer tags
            number_of_integer_tags = int(f.readline().decode().strip())  # usually 3 or 4
            integer_tags = [int(f.readline().decode().strip())
                            for i in range(number_of_integer_tags)]
            nr = integer_tags[2]
            nr_comp = integer_tags[1]
            return section.strip(), name, nr, nr_comp

        def read_NodeData(t, name, nr, nr_comp, m):
            data = NodeData(np.empty((nr, nr_comp)), name=name, mesh=m)
            if binary:
                if nr_comp == 1:
                    value_dt = ('values', np.float64)
                else:
                    value_dt = ('values', np.float64, nr_comp)
                dt = np.dtype([('id', np.int32), value_dt])
                temp = np.fromfile(f, dtype=dt, count=nr)
                node_number = np.copy(temp['id'])
                data.value = np.copy(temp['values'])
            else:
                node_number = np.empty(nr, dtype='int32')
                data.value = np.empty((nr, nr_comp), dtype='float64')
                for ii in range(nr):
                    line = f.readline().decode().strip().split()
                    node_number[ii] = int(line[0])
                    data.value[ii, :] = [float(v) for v in line[1:]]

            if f.readline() != b'$EndNodeData\n':
                raise IOError(fn + " expected $EndNodeData after reading " +
                              str(nr) + " lines in $NodeData")

            if np.any(node_number != m.nodes.node_number):
                raise IOError("Can't read NodeData field: "
                              "it does not have one data point per node")

            return data

        def read_ElementData(t, name, nr, nr_comp, m):
            if elm_nr_changed or not np.all(read):
                raise IOError('Could not read ElementData: '
                              'Element ordering not compact or invalid element type')
            data = ElementData(np.empty((nr, nr_comp)), name=name, mesh=m)
            if binary:
                if nr_comp == 1:
                    value_dt = ('values', np.float64)
                else:
                    value_dt = ('values', np.float64, nr_comp)
                dt = np.dtype([('id', np.int32), value_dt])
                temp = np.fromfile(f, dtype=dt, count=nr)
                elm_number = np.copy(temp['id'])
                data.value = np.copy(temp['values'])

            else:
                elm_number = np.empty(nr, dtype='int32')
                data.value = np.empty([nr, nr_comp], dtype='float64')

                for ii in range(nr):
                    line = f.readline().decode().strip().split()
                    elm_number[ii] = int(line[0])
                    data.value[ii, :] = [float(jj) for jj in line[1:]]

            if f.readline() != b'$EndElementData\n':
                raise IOError(fn + " expected $EndElementData after reading " +
                              str(nr) + " lines in $ElementData")

            if np.any(elm_number != m.elm.elm_number):
                raise IOError("Can't read ElementData field: "
                              "it does not have one data point per element")

            return data

        # read sections recursively
        def read_next_section():
            t, name, nr, nr_comp = parse_Data()
            if t == 'EOF':
                return
            elif t == b'$NodeData':
                m.nodedata.append(read_NodeData(t, name, nr, nr_comp, m))
            elif t == b'$ElementData':
                m.elmdata.append(read_ElementData(t, name, nr, nr_comp, m))
            else:
                raise IOError("Can't recognize section name:" + t)

            read_next_section()
            return

        if not skip_data:
            read_next_section()
    m.compact_ordering(node_number)
    return m


def _read_msh_4(fn, m, skip_data=False):
    m.fn = fn
    # file open
    with open(fn, 'rb') as f:
        # check 1st line
        first_line = f.readline()
        if first_line != b'$MeshFormat\n':
            raise IOError(fn, "must start with $MeshFormat")

        # parse 2nd line
        version_number, file_type, data_size = f.readline().decode().strip().split()
        version_number = int(version_number[0])
        file_type = int(file_type)
        data_size = int(data_size)

        if version_number != 4:
            raise IOError("Can only handle v4 meshes")

        if file_type == 1:
            binary = True
        elif file_type == 0:
            binary = False
        else:
            raise IOError("File_type not recognized: {0}".format(file_type))

        if data_size != 8:
            raise IOError(
                "data_size should be double (8), i'm reading: {0}".format(data_size))

        # read next byte, if binary, to check for endianness
        if binary:
            endianness = struct.unpack('i', f.readline()[:4])[0]
        else:
            endianness = 1

        if endianness != 1:
            raise IOError("endianness is not 1, is the endian order wrong?")

        # read 3rd line
        if f.readline() != b'$EndMeshFormat\n':
            raise IOError(fn + " expected $EndMeshFormat")

        # Skip  everyting untill nodes
        while f.readline() != b'$Nodes\n':
            continue

        # Number of nodes and number of blocks
        if binary:
            entity_blocks, node_nr = struct.unpack('LL', f.read(struct.calcsize('LL')))
        else:
            line = f.readline().strip()
            entity_blocks, node_nr = line.decode().split()
            entity_blocks = int(entity_blocks)
            node_nr = int(node_nr)
        n_read = 0
        node_number = np.zeros(node_nr, dtype=np.int32)
        node_coord = np.zeros((node_nr, 3), dtype=float)
        for block in range(entity_blocks):
            if binary:
                _, _, parametric, n_in_block = struct.unpack(
                    'iiii', f.read(struct.calcsize('iiii')))
                # We need to read 4 extra bytes here
                f.read(4)
            else:
                _, _, parametric, n_in_block = f.readline().decode().strip().split()
                parametric = int(parametric)
                n_in_block = int(n_in_block)
            if parametric:
                raise IOError("Can't read parametric entity!")
            if binary:
                dt = np.dtype([
                    ('id', np.int32, 1),
                    ('coord', np.float64, 3)])
                temp = np.fromfile(f, dtype=dt, count=n_in_block)
                node_nbr_block = temp['id']
                node_coord_block = temp['coord']
            else:
                node_nbr_block = np.zeros(n_in_block, dtype=int)
                node_coord_block = np.zeros(3 * n_in_block, dtype=float)
                for i in range(n_in_block):
                    line = f.readline().decode().strip().split()
                    node_nbr_block[i] = line[0]
                    node_coord_block[3*i] = line[1]
                    node_coord_block[3*i + 1] = line[2]
                    node_coord_block[3*i + 2] = line[3]
                node_coord_block = node_coord_block.reshape(-1, 3)

            node_number[n_read:n_read+n_in_block] = node_nbr_block
            node_coord[n_read:n_read+n_in_block, :] = node_coord_block
            n_read += n_in_block

        m.nodes.node_coord = node_coord

        line = f.readline()
        if b'$EndNodes' not in line:
            line = f.readline()
            if b'$EndNodes' not in line:
                raise IOError(fn + " expected $EndNodes after reading " +
                              str(m.noedes.nr) + " nodes. Read " + line)

        # Read Elements
        if f.readline() != b'$Elements\n':
            raise IOError(fn, "expected line with $Elements")
        if binary:
            entity_blocks, elm_nr = struct.unpack('LL', f.read(struct.calcsize('LL')))
        else:
            line = f.readline().strip()
            entity_blocks, elm_nr = line.decode().split()
            entity_blocks = int(entity_blocks)
            elm_nr = int(elm_nr)
        n_read = 0
        elm_number = np.zeros(elm_nr, dtype=np.int32)
        m.elm.elm_type = np.zeros(elm_nr, dtype=np.int32)
        m.elm.tag1 = np.zeros(elm_nr, dtype=np.int32)
        m.elm.node_number_list = -np.ones((elm_nr, 4), dtype=np.int32)
        read = np.ones(elm_nr, dtype=bool)
        for block in range(entity_blocks):
            if binary:
                tag, _, elm_type, n_in_block = struct.unpack(
                    'iiii', f.read(struct.calcsize('iiii')))
                f.read(4)
            else:
                tag, _, elm_type, n_in_block = f.readline().decode().strip().split()
                tag = int(tag)
                elm_type = int(elm_type)
                n_in_block = int(n_in_block)

            if elm_type == 2:
                nr_nodes_elm = 3
            elif elm_type == 4:
                nr_nodes_elm = 4
            else:
                warnings.warn(
                    "Can't read element type: {}. Ignoring it".format(elm_type))
                continue

            if binary:
                dt = np.dtype([
                    ('id', np.int32, 1),
                    ('nodes', np.int32, nr_nodes_elm)])
                temp = np.fromfile(f, dtype=dt, count=n_in_block)
                elm_nbr_block = temp['id']
                elm_node_block = temp['nodes']

            else:
                elm_nbr_block = np.zeros(n_in_block, dtype=np.int32)
                elm_node_block = -np.ones(nr_nodes_elm * n_in_block, dtype=np.int32)
                for i in range(n_in_block):
                    line = f.readline().decode().strip().split()
                    elm_nbr_block[i] = int(line[0])
                    elm_node_block[nr_nodes_elm*i:nr_nodes_elm*(i+1)] = [int(l) for l in line[1:]]
                elm_node_block = elm_node_block.reshape(-1, nr_nodes_elm)

            elm_number[n_read:n_read+n_in_block] = elm_nbr_block
            m.elm.node_number_list[n_read:n_read+n_in_block, :nr_nodes_elm] = elm_node_block
            m.elm.tag1[n_read:n_read+n_in_block] = tag
            m.elm.elm_type[n_read:n_read+n_in_block] = elm_type
            read[n_read:n_read+n_in_block] = True
            n_read += n_in_block

        elm_number = elm_number[read]
        m.elm.node_number_list = m.elm.node_number_list[read]
        m.elm.tag1 = m.elm.tag1[read]
        m.elm.elm_type = m.elm.elm_type[read]

        order = np.argsort(elm_number)
        elm_number = elm_number[order]
        m.elm.node_number_list = m.elm.node_number_list[order]
        m.elm.tag1 = m.elm.tag1[order]
        m.elm.elm_type = m.elm.elm_type[order]
        m.elm.tag2 = m.elm.tag1

        line = f.readline()
        if b'$EndElements' not in line:
            line = f.readline()
            if b'$EndElements' not in line:
                raise IOError(fn + " expected $EndElements after reading " +
                              str(m.elm.nr) + " elements. Read " + line)

        # read the header in the beginning of a data section
        def parse_Data():
            section = f.readline()
            if section == b'':
                return 'EOF', '', 0, 0
            # string tags
            number_of_string_tags = int(f.readline().decode('ascii'))
            assert number_of_string_tags == 1, "Invalid Mesh File: invalid number of string tags"
            name = f.readline().decode('ascii').strip().strip('"')
            # real tags
            number_of_real_tags = int(f.readline().decode('ascii'))
            assert number_of_real_tags == 1, "Invalid Mesh File: invalid number of real tags"
            f.readline()
            # integer tags
            number_of_integer_tags = int(f.readline().decode('ascii'))  # usually 3 or 4
            integer_tags = [int(f.readline().decode('ascii'))
                            for i in range(number_of_integer_tags)]
            nr = integer_tags[2]
            nr_comp = integer_tags[1]
            return section.strip(), name, nr, nr_comp

        def read_NodeData(t, name, nr, nr_comp, m):
            data = NodeData(np.empty((nr, nr_comp)), name=name, mesh=m)
            if binary:
                dt = np.dtype([
                    ('id', np.int32, 1),
                    ('values', np.float64, nr_comp)])

                temp = np.fromfile(f, dtype=dt, count=nr)
                node_number = np.copy(temp['id'])
                data.value = np.copy(temp['values'])
            else:
                node_number = np.empty(nr, dtype='int32')
                data.value = np.empty((nr, nr_comp), dtype='float64')
                for ii in range(nr):
                    line = f.readline().decode('ascii').split()
                    node_number[ii] = int(line[0])
                    data.value[ii, :] = [float(v) for v in line[1:]]

            if f.readline() != b'$EndNodeData\n':
                raise IOError(fn + " expected $EndNodeData after reading " +
                              str(nr) + " lines in $NodeData")

            if np.any(node_number != m.nodes.elm_number):
                raise IOError("Can't read NodeData field: "
                              "it does not have one data point per node")

            return data

        def read_ElementData(t, name, nr, nr_comp, m):
            if not np.all(read):
                raise IOError('Could not read ElementData: '
                              'Element ordering not compact or invalid element type')
            data = ElementData(np.empty((nr, nr_comp)), name=name, mesh=m)
            if binary:
                dt = np.dtype([
                    ('id', np.int32, 1),
                    ('values', np.float64, nr_comp)])

                temp = np.fromfile(f, dtype=dt, count=nr)
                elm_number = np.copy(temp['id'])
                data.value = np.copy(temp['values'])

            else:
                elm_number = np.empty(nr, dtype='int32')
                data.value = np.empty([nr, nr_comp], dtype='float64')

                for ii in range(nr):
                    line = f.readline().decode('ascii').split()
                    elm_number[ii] = int(line[0])
                    data.value[ii, :] = [float(jj) for jj in line[1:]]

            if f.readline() != b'$EndElementData\n':
                raise IOError(fn + " expected $EndElementData after reading " +
                              str(nr) + " lines in $ElementData")

            if np.any(elm_number != m.elm.elm_number):
                raise IOError("Can't read ElementData field: "
                              "it does not have one data point per element")

            return data

        # read sections recursively
        def read_next_section():
            t, name, nr, nr_comp = parse_Data()
            if t == 'EOF':
                return
            elif t == b'$NodeData':
                m.nodedata.append(read_NodeData(t, name, nr, nr_comp, m))
            elif t == b'$ElementData':
                m.elmdata.append(read_ElementData(t, name, nr, nr_comp, m))
            else:
                raise IOError("Can't recognize section name:" + t)

            read_next_section()
            return

        if not skip_data:
            read_next_section()
    m.compact_ordering(node_number)
    return m


# write msh to mesh file
def write_msh(msh, file_name=None, mode='binary', mmg_fix=False):
    """ Writes a gmsh 'msh' file

    Parameters
    ------------
    msh: simnibs.msh.Msh
        Mesh structure
    file_name: str (optional)
        Name of file to be writte. Default: msh.fn
    mode: 'binary' or 'ascii':
        The mode in which the file should be read
    """
    if file_name is not None:
        msh.fn = file_name

    fn = msh.fn

    if fn[0] == '~':
        fn = os.path.expanduser(fn)

    if mode not in ['ascii', 'binary']:
        raise ValueError("Only 'ascii' and 'binary' are allowed")

    with open(fn, 'wb') as f:
        if mode == 'ascii':
            f.write(b'$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')

        elif mode == 'binary':
            f.write(b'$MeshFormat\n2.2 1 8\n')
            f.write(struct.pack('i', 1))
            f.write(b'\n$EndMeshFormat\n')

        # write nodes
        f.write(b'$Nodes\n')
        f.write('{0}\n'.format(msh.nodes.nr).encode('ascii'))

        if mode == 'ascii':
            for ii in range(msh.nodes.nr):
                f.write((str(msh.nodes.node_number[ii]) + ' ' +
                         str(msh.nodes.node_coord[ii][0]) + ' ' +
                         str(msh.nodes.node_coord[ii][1]) + ' ' +
                         str(msh.nodes.node_coord[ii][2]) + '\n').encode('ascii'))

        elif mode == 'binary':
            node_number = msh.nodes.node_number.astype('int32')
            node_coord = msh.nodes.node_coord.astype('float64')
            m = node_number[:, np.newaxis]
            for i in range(3):
                nc = node_coord[:, i].astype('float64')
                m = np.concatenate((m,
                                    nc.view('int32').reshape(-1, 2)), axis=1)
            f.write(m.tobytes())
        f.write(b'$EndNodes\n')
        # write elements
        f.write(b'$Elements\n')
        f.write((str(msh.elm.nr) + '\n').encode('ascii'))

        if mode == 'ascii':
            for ii in range(msh.elm.nr):
                line = str(msh.elm.elm_number[ii]) + ' ' + \
                    str(msh.elm.elm_type[ii]) + ' ' + str(2) + ' ' +\
                    str(msh.elm.tag1[ii]) + ' ' + str(msh.elm.tag2[ii]) + ' '

                if msh.elm.elm_type[ii] == 2:
                    line += str(msh.elm.node_number_list[ii, :3]
                                ).translate(None, '[](),') + '\n'
                elif msh.elm.elm_type[ii] == 4:
                    line += str(msh.elm.node_number_list[ii, :]
                                ).translate(None, '[](),') + '\n'
                elif msh.elm.elm_type[ii] == 15:
                    line += str(msh.elm.node_number_list[ii, :1]
                                ).translate(None, '[](),') + '\n'
                elif msh.elm.elm_type[ii] == 1:
                    line += str(msh.elm.node_number_list[ii, :2]
                                ).translate(None, '[](),') + '\n'
                else:
                    raise IOError(
                        "ERROR: cant write meshes with elements of type",
                        msh.elm.elm_type[ii])

                f.write(line.encode('ascii'))

        elif mode == 'binary':
            points = np.where(msh.elm.elm_type == 15)[0]
            if len(points > 0):
                points_header = np.array((15, len(points), 2), 'int32')
                points_number = msh.elm.elm_number[points].astype('int32')
                points_tag1 = msh.elm.tag1[points].astype('int32')
                points_tag2 = msh.elm.tag2[points].astype('int32')
                points_node_list = msh.elm.node_number_list[
                    points, :1].astype('int32')
                f.write(points_header.tobytes())
                f.write(np.concatenate((points_number[:, np.newaxis],
                                        points_tag1[:, np.newaxis],
                                        points_tag2[:, np.newaxis],
                                        points_node_list), axis=1).tobytes())
            lines = np.where(msh.elm.elm_type == 1)[0]
            if len(lines > 0):
                lines_header = np.array((1, len(lines), 2), 'int32')
                lines_number = msh.elm.elm_number[lines].astype('int32')
                lines_tag1 = msh.elm.tag1[lines].astype('int32')
                lines_tag2 = msh.elm.tag2[lines].astype('int32')
                lines_node_list = msh.elm.node_number_list[
                    lines, :2].astype('int32')
                f.write(lines_header.tobytes())
                f.write(np.concatenate((lines_number[:, np.newaxis],
                                        lines_tag1[:, np.newaxis],
                                        lines_tag2[:, np.newaxis],
                                        lines_node_list), axis=1).tobytes())
            triangles = np.where(msh.elm.elm_type == 2)[0]
            if len(triangles > 0):
                triangles_header = np.array((2, len(triangles), 2), 'int32')
                triangles_number = msh.elm.elm_number[triangles].astype('int32')
                triangles_tag1 = msh.elm.tag1[triangles].astype('int32')
                triangles_tag2 = msh.elm.tag2[triangles].astype('int32')
                triangles_node_list = msh.elm.node_number_list[
                    triangles, :3].astype('int32')
                f.write(triangles_header.tobytes())
                f.write(np.concatenate((triangles_number[:, np.newaxis],
                                        triangles_tag1[:, np.newaxis],
                                        triangles_tag2[:, np.newaxis],
                                        triangles_node_list), axis=1).tobytes())

            tetra = np.where(msh.elm.elm_type == 4)[0]
            if len(tetra > 0):
                tetra_header = np.array((4, len(tetra), 2), 'int32')
                tetra_number = msh.elm.elm_number[tetra].astype('int32')
                tetra_tag1 = msh.elm.tag1[tetra].astype('int32')
                tetra_tag2 = msh.elm.tag2[tetra].astype('int32')
                tetra_node_list = msh.elm.node_number_list[tetra].astype('int32')

                f.write(tetra_header.tobytes())
                f.write(np.concatenate((tetra_number[:, np.newaxis],
                                        tetra_tag1[:, np.newaxis],
                                        tetra_tag2[:, np.newaxis],
                                        tetra_node_list), axis=1).tobytes())

        f.write(b'$EndElements\n')

    # write nodeData, if existent
    for nd in msh.nodedata:
        nd.append_to_mesh(fn, mode, mmg_fix)

    for eD in msh.elmdata:
        eD.append_to_mesh(fn, mode)


'''
# Adds 1000 to the label of triangles, if less than 100
def create_surface_labels(msh):
    triangles = np.where(msh.elm.elm_type == 2)[0]
    triangles = np.where(msh.elm.tag1[triangles] < 1000)[0]
    msh.elm.tag1[triangles] += 1000
    msh.elm.tag2[triangles] += 1000
    return msh
'''


def _fix_indexing_one(index):
    '''Fix indexing to allow getting and setting items with one-idexed arrays'''
    def fix_slice(slc):
        start = slc.start
        stop = slc.stop
        step = slc.step
        if start is not None:
            if start > 0:
                start -= 1
            elif start == 0:
                raise IndexError('Cant get item 0 in one-indexed array')
            else:
                raise IndexError('Cant get negative slices in one-indexed array')

        if stop is not None:
            if stop > 0:
                stop -= 1
            elif stop == 0:
                raise IndexError('Cant get item 0 in one-indexed array')
            else:
                raise IndexError('Cant get negative slices in one-indexed array')

        if step is not None and step < 0:
            raise IndexError('Cant get negative slices in one-indexed array')

        return slice(start, stop, step)

    def fix_index_array(idx_array):
        idx_array = np.array(idx_array)
        if idx_array.dtype == bool:
            return idx_array
        else:
            if np.any(idx_array == 0):
                raise IndexError('Cant get item 0 in one-indexed array')
            idx_array[idx_array > 0] -= 1
            return idx_array

    def is_integer(index):
        answer = isinstance(index, int)
        answer += isinstance(index, int)
        answer += isinstance(index, np.intc)
        answer += isinstance(index, np.intp)
        answer += isinstance(index, np.int8)
        answer += isinstance(index, np.int16)
        answer += isinstance(index, np.int32)
        answer += isinstance(index, np.int64)
        answer += isinstance(index, np.uint8)
        answer += isinstance(index, np.uint16)
        answer += isinstance(index, np.uint32)
        answer += isinstance(index, np.uint64)
        return answer

    if is_integer(index):
        if index > 0:
            return index - 1
        if index < 0:
            return index
        if index == 0:
            raise IndexError('Cant get item 0 in one-indexed array')

    elif isinstance(index, slice):
        return fix_slice(index)

    elif isinstance(index, list) or isinstance(index, np.ndarray):
        return fix_index_array(index)

    elif isinstance(index, tuple):
        index = list(index)
        if isinstance(index[0], list) or isinstance(index[0], np.ndarray):
            index[0] = fix_index_array(index[0])
        if isinstance(index[0], slice):
            index[0] = fix_slice(index[0])
        return tuple(index)

    else:
        return index

def _getitem_one_indexed(array, index):
    index = _fix_indexing_one(index)
    return array.__getitem__(index)
