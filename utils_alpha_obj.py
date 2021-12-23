import numpy as np
# from operator import xor
from graph_generator import get_all_oriented_edges
from itertools import repeat
from utils_basic_functions import from_matrix_to_dict, from_dict_to_matrix


class Alpha_obj:
    """
    {'alpha': alpha, 
    'alpha_c': alpha_c, 'alpha_d':alpha_d, 
    'dict_alpha_impaired': dict_alpha_impaired,
    'K_nodes': K_nodes, 'K_edges': K_edges,
    'alpha_matrix': alpha_matrix,
    'K_nodes_vector': K_nodes_vector, 'K_edges_matrix': K_edges_matrix
    }
    """
    
    def __init__(self, d):
        assert type(d) == dict
        self.d = d
#         for key, val in d.items():
#             setattr(self, key, val)
    
    def __str__(self):
        """
        print function
        """
        d = self.to_dict()
        d = {key: val for key, val in d.items() if val is not None} #remove the None values
        return d.__str__()
    
    def get(self, s, default_val=None):
        """
        Useful for CIbeliefs for s='dict_alpha_impaired' (because get_alpha_dict doesn't apply to this algorithm, as create_alpha_dict does not)
        """
#         print("self.to_dict() = {}".format(self.to_dict()))
#         print("self.to_dict().get(s, default_val) = {}".format(self.to_dict().get(s, default_val)))
        return self.to_dict().get(s, default_val)
    
    def __eq__(self, other): 
        if not isinstance(other, Alpha_obj):
            # don't attempt to compare against unrelated types
            return NotImplemented
        else:
            return self.to_dict() == other.to_dict()
            
    def check(self):
        """
        Checks that one and only one attribute is not None:
        """
        check_alpha_obj(**self.to_dict())
        
    def get_alpha_dict(self, graph):
        """
        Transforms the object into dict_alpha (by forming it from the attributes, using create_alpha_dict):
        create_alpha_dict(graph, 
                          alpha=alpha, alpha_c=alpha_c, alpha_d=alpha_d, dict_alpha_impaired=dict_alpha_impaired)
        """
#         print("get_alpha_dict")
        return create_alpha_dict(graph, **self.to_dict())
    
    def to_dict(self):
        """
        Returns a dict with the form {attribute: values , for all attributes of the object}
        """
        return self.d #if the object has attributes .alpha, .alpha_c, ...etc, then use "return var(self)" instead
    
    def to_matrix(self, graph=None):
        """
        In fact graph is needed only in the non-parallel case (unless alpha_ij = alpha i.e. uniform alpha)
        """
        if self.get('alpha_matrix') is not None: #(self.to_dict().get('alpha_matrix') is not None):
            return self.to_dict().get('alpha_matrix')
        elif (self.get('K_nodes_vector') is not None) or (self.get('K_edges_matrix') is not None): #(self.to_dict().get('K_nodes_vector') is not None) or (self.to_dict().get('K_edges_matrix') is not None):
#             K_nodes_vector = self.to_dict().get('K_nodes_vector', 1)
#             K_edges_matrix = self.to_dict().get('K_edges_matrix', 1)
#             if K_nodes_vector is not 1: #!= 1
#                 K_nodes_vector = K_nodes_vector[..., np.newaxis]
#             return K_nodes_vector / K_edges_matrix #alpha_ij = K_i / K_ij
            K_nodes_vector = self.to_dict().get('K_nodes_vector')
            K_edges_matrix = self.to_dict().get('K_edges_matrix')
            if (K_nodes_vector is not None) and (K_edges_matrix is not None):
                return K_nodes_vector[..., np.newaxis] / K_edges_matrix 
            elif K_nodes_vector is None:
                return 1 / K_edges_matrix
            elif K_edges_matrix is None:
                return K_nodes_vector[..., np.newaxis] #To do the operation alpha_ij * M_ji = K_i * M_ji, then do K_nodes_vector[..., np.newaxis] * M.T (= M.T * K_nodes_vector[..., np.newaxis]) ---> = alpha_matrix * M.T (where alpha_matrix is what's returned by this function to_matrix)
        #get dict
#         assert graph is not None #I removed it because it could be that 'alpha' is given (with parallel_CI = True)
        alpha_dict = self.get_alpha_dict(graph)
        #transform dict into matrix
        alpha_matrix = from_dict_to_matrix(alpha_dict, list(graph.nodes), default_value=1)
        return alpha_matrix
    
    def is_temporal(self, graph):
        """
        Works only for models != 'CIbeliefs'
        
        #False if (isinstance(alpha_c, int) + isinstance(alpha_c, float)) else True #does not deal with the cases where only one out of (alpha_c, alpha_d) is constant and the other one varies with time
        """
        if self.get('alpha_matrix') is not None:
            return len(self.get('alpha_matrix').shape) > 2
        elif self.get('K_nodes_vector') is not None:
            return len(self.get('K_nodes_vector').shape) > 1
        elif self.get('K_edges_matrix') is not None:
            return len(self.get('K_edges_matrix').shape) > 2
        else:
#             print("is_temporal")
            alpha_dict = self.get_alpha_dict(graph)
        return np.sum([not(isinstance(val, int) or isinstance(val, np.int64)
                           or isinstance(val, float) or isinstance(val, np.float32)) 
                       for val in alpha_dict.values()]) != 0
        
        
def generalized_xor(*args):
    """
    Returns True if and if only there is exactly one argument equal to True (and all the others = False)
    It helps to check that one of the options (and only one) is selected
    For 2 arguments, it is the normal XOR = operator.xor function
    """
    return np.sum([int(arg) for arg in args]) == 1


def one_among(*args):
    """
    Returns True if and if only there is exactly one argument different from None (and all the others = None)
    It helps to check that one of the options (and only one) is selected
    
    Example:
    one_among(None, None, 1, None) ---> True
    one_among(None, None, 1, 3)    ---> False
    """
    return np.sum([int(arg is not None) for arg in args]) == 1

    
def check_alpha_obj(alpha=None, alpha_c=None, alpha_d=None, K_nodes=None, K_edges=None, dict_alpha_impaired=None,
                    alpha_matrix=None, K_nodes_vector=None, K_edges_matrix=None
                   ):
    """
    Checks that only one element is not None in the dictionary given as input
    #exclusive or (True if and if only one of the arguments is True)
    
    generalized_xor(alpha is not None, 
                alpha_c is not None or alpha_d is not None,
                dict_alpha_impaired is not None,
                K_nodes is not None or K_edges is not None)
#     assert xor((alpha_c is not None) and (alpha_d is not None), dict_alpha_impaired is not None) #exclusive or
#     assert not((alpha is None) and ((alpha_c is None) and (alpha_d is None)))
#     assert not((alpha is not None) and ((alpha_c is not None) or (alpha_d is not None)))
#     if alpha is None: #(alpha_c, alpha_d) is given
#         assert alpha_c == alpha_d #alpha_c != alpha_d is not implemented
#         alpha = alpha_c
    
    Example:
    check_alpha_obj({'alpha': None, 'alpha_c': 0.8, 'alpha_d': 0.9, 'dict_alpha_impaired' : {(0,1):0.9, (2,3):1.2} )
    """
#     alpha, alpha_c, alpha_d, dict_alpha, K_nodes, K_edges = d_alpha['alpha'], d_alpha['alpha_c'], d_alpha['alpha_d'], d_alpha['dict_alpha_impaired'], d_alpha['K_nodes'], d_alpha['K_edges']
#     assert not(alpha is None and dict_alpha is None and K_nodes is None and K_edges is None)
    assert generalized_xor(
        alpha is not None,
        (alpha_c is not None) or (alpha_d is not None),
        (K_nodes is not None) or (K_edges is not None),
        dict_alpha_impaired is not None,
        alpha_matrix is not None,
        (K_nodes_vector is not None) or (K_edges_matrix is not None),
    )
    return True


def create_alpha_dict(graph,
                      alpha=None, alpha_c=None, alpha_d=None, dict_alpha_impaired=None, K_nodes=None, K_edges=None,
                      alpha_matrix=None, K_nodes_vector=None, K_edges_matrix=None
                     ):
    """
    Creates a dictionnary with all the alpha_ij (for all directed edges (i,j))
    It also checks that exactly one among {alpha; (alpha_c, alpha_d); dict_alpha_impaired} is defined
    dict_alpha_impaired is defined for any oriented edge (i,j)
    The information of graph is only useful to know the orientation of edges (= whether it is going down or up)
    """
    assert check_alpha_obj(alpha=alpha, alpha_c=alpha_c, alpha_d=alpha_d, K_nodes=K_nodes, K_edges=K_edges, 
                           dict_alpha_impaired=dict_alpha_impaired, 
                           alpha_matrix=alpha_matrix, K_nodes_vector=K_nodes_vector, K_edges_matrix=K_edges_matrix)
    
    dict_alpha = {}
    
    if alpha is not None:
        dict_alpha = dict(zip(get_all_oriented_edges(graph), repeat(alpha)))
    
    elif (alpha_c is not None) or (alpha_d is not None): #alpha_c and alpha_d are uniform i.e. identical for each edge
        if alpha_c is None:
            alpha_c = 1
        if alpha_c is None:
            alpha_d = 1
        dict_alpha_edge = {'down': alpha_c,
                           'up': alpha_d} #because M_ij = F(B_i - alpha_ij M_ji)
        for node1, node2, d in graph.edges(data=True): #we do not count the same edge twice because graph only has (node1,node2), not both (node1,node2) and (node2,node1)
            type_edge = d['orientation']
            type_edge_opposite = 'up' if type_edge == 'down' else 'down'
            dict_alpha[node1, node2] = dict_alpha_edge[type_edge]
            dict_alpha[node2, node1] = dict_alpha_edge[type_edge_opposite]
            
    elif (K_nodes is not None) or (K_edges is not None):
        if (K_nodes is not None) and (K_edges is not None): #alpha_ij = K_i / K_ij
            for (node1, node2) in graph.edges:
                dict_alpha[node1, node2] = K_nodes[node1] / K_edges[node1, node2]
                dict_alpha[node2, node1] = K_nodes[node2] / K_edges[node1, node2] #"inverted" K_nodes
        elif K_nodes is None: #alpha_ij = 1 / K_ij
            for (node1, node2) in graph.edges:
                dict_alpha[node1, node2] = 1 / K_edges[node1, node2]
                dict_alpha[node2, node1] = 1 / K_edges[node1, node2]
        elif K_edges is None: #alpha_ij = K_i
            for (node1, node2) in graph.edges:
                dict_alpha[node1, node2] = K_nodes[node1]
                dict_alpha[node2, node1] = K_nodes[node2]
    
    elif dict_alpha_impaired is not None: #non-uniformity of alpha, indicated by dict_alpha_impaired
        if len(dict_alpha_impaired) == 2 * len(graph.edges): #all oriented edges are already represented
            dict_alpha = dict_alpha_impaired
        else:
            for (node1, node2) in get_all_oriented_edges(graph):
                if (node1, node2) in dict_alpha_impaired.keys():
                    dict_alpha[node1, node2] = dict_alpha_impaired[node1, node2]
                else:
                    dict_alpha[node1, node2] = 1
    
    elif alpha_matrix is not None:
        dict_alpha = from_matrix_to_dict(alpha_matrix, list(graph.nodes))
    
    elif (K_nodes_vector is not None) or (K_edges_matrix is not None):
        print("Calling create_alpha_dict with K_nodes_vector and/or K_edges_matrix - This seems useless")
        corresp_nodes = dict(zip(list(graph.nodes), range(len(list(graph.nodes)))))
        if (K_nodes_vector is not None) and (K_edges_matrix is not None): #alpha_ij = K_i / K_ij
            assert np.all(K_edges_matrix == K_edges_matrix.T)
            for (node1, node2) in graph.edges:
                i_node1, i_node2 = corresp_nodes[node1], corresp_nodes[node2]
                dict_alpha[node1, node2] = K_nodes_vector[i_node1] / K_edges_matrix[i_node1, i_node2]
                dict_alpha[node2, node1] = K_nodes_vector[i_node2] / K_edges_matrix[i_node1, i_node2] #"inverted" K_nodes
        elif K_nodes_vector is None: #alpha_ij = 1 / K_ij
            assert np.all(K_edges_matrix == K_edges_matrix.T)
            for (node1, node2) in graph.edges:
                i_node1, i_node2 = corresp_nodes[node1], corresp_nodes[node2]
                dict_alpha[node1, node2] = 1 / K_edges_matrix[i_node1, i_node2]
                dict_alpha[node2, node1] = 1 / K_edges_matrix[i_node1, i_node2] #"inverted" K_nodes
        elif K_edges_matrix is None: #alpha_ij = K_i
            for (node1, node2) in graph.edges:
                i_node1, i_node2 = corresp_nodes[node1], corresp_nodes[node2]
                dict_alpha[node1, node2] = K_nodes_vector[i_node1] 
                dict_alpha[node2, node1] = K_nodes_vector[i_node2]
            
    else:
        raise NotImplemented
                    
    return dict_alpha
