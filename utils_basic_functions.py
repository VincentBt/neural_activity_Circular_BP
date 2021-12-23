import numpy as np
from scipy.special import expit
import itertools


# @jit
def sig(x):
#     return expit(x) #faster according to Marc? But it doesn't seem the case for me... --> check
    return 1 / (1 + np.exp(-x))


def tanh_inverse(x):
    return 1/2 * np.log((1 + x) / (1 - x))


def from_dict_to_matrix(d, order_keys, default_value=1, make_symmetrical=False):
    """
    TODO: improve speed
    
    See https://www.geeksforgeeks.org/python-convert-coordinate-dictionary-to-matrix/
    """
#     print("TODO: check everywhere in the code that from_dict_to_matrix is used appropriately, i.e., that make_symmetrical=True is indeed used when needed")
#     d_matrix = np.zeros((len(order_keys), len(order_keys))) + default value
#     for i_node1, node1 in enumerate(list(order_keys)):
#         for i_node2, node2 in enumerate(list(order_keys)):
#             if (node1, node2) not in d.keys():
#                 continue
#             d_matrix[i_node1, i_node2] = d[node1, node2]
#     return d_matrix
    
    if make_symmetrical == False:
        return np.array([[d.get((i, j), default_value) for j in order_keys] 
                         for i in order_keys])
    else:
        #what's commented juts below doesn't work because if the default value != 0
#         M = np.array([[d.get((i, j), default_value) for j in order_keys] 
#                       for i in order_keys])
#         return M + M.T #not (M + M.T) / 2 here: because we only give d[i,j] and not d[j,i], which is the same (unoriented edge)
        d_matrix = np.zeros((len(order_keys), len(order_keys))) + default_value
        corresp_nodes = dict(zip(order_keys, range(len(order_keys))))
        for (node1, node2) in itertools.product(order_keys, order_keys):
            i_node1, i_node2 = corresp_nodes[node1], corresp_nodes[node2]
            if (node1, node2) in d.keys():
                d_matrix[i_node1, i_node2] = d[node1, node2]
            elif (node2, node1) in d.keys():
                d_matrix[i_node1, i_node2] = d[node2, node1]
#             else:
#                 print("Problem: the dictionnary is not filled")
        return d_matrix


def from_dict_to_vector(d, order_keys, default_value=1):
    return np.array([d.get((i), default_value) for i in order_keys])

    
def from_matrix_to_dict(m, order_keys):
    """
    I haven't found a way yet to be faster than that
    """
    d = {}
    for ikey1, key1 in enumerate(order_keys):
        for ikey2, key2 in enumerate(order_keys):
            d[key1, key2] = m[ikey1, ikey2]
    return d


def transf_dict_list_into_list_dict(B_CI):
    """
    TODO: look whether this can be made faster

    Transforms B_CI, initially {node: [B_node for all examples] for all nodes}, into [{node:B_node for all nodes} for all examples]
    """
#     assert list(B_CI.keys()) == list(graph.nodes)
    B_CI_new = [{} for _ in range(len(list(B_CI.values())[0]))]
    for node, B_CI_node in B_CI.items():
        for i, B_CI_val in enumerate(B_CI_node):
            B_CI_new[i][node] = B_CI_val
    B_CI = B_CI_new
    return B_CI


def change_order_descending_p(list_data_file):
    list_p = reversed([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    list_p_file = ['p' + str(p).replace('.','') for p in list_p]
    list_data_file_new = []
    for p_file in list_p_file:
        for file in list_data_file:
            if p_file in file:
                list_data_file_new.append(file)
#     pprint(list_data_file_new)
    list_data_file = list_data_file_new
    return list_data_file

def change_order_descending_struct(list_data_file):
    list_struct = reversed(['star', 'binary_tree', 'path', 'cycle', 'ladder', 'grid', 'circular_ladder', 'barbell', 'lollipop', 'wheel', 'bipartite', 'tripartite', 'complete'])
    list_struct_file = list_struct
    list_data_file_new = []
    for struct_file in list_struct_file:
        for file in list_data_file:
            if struct_file in file and file not in list_data_file_new: #2nd condition: in order to differenciate ladder from circular_ladder
                list_data_file_new.append(file)
#     pprint(list_data_file_new)
    list_data_file = list_data_file_new
    return list_data_file