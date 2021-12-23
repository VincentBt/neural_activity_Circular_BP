# -*- coding: utf-8 -*-

import pickle
from scipy.io import savemat
import scipy
import networkx as nx
import pandas as pd
import numpy as np
import os


def save_pickle(filename, obj, 
                path_folder = "../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/",
                which_python='python3'):
    path_file = str(path_folder) + str(filename) #+ ".p"
    assert which_python in ['python2', 'python3']
    if which_python == 'python3': #default
        #     pickle.dump(var, open(path_file, "wb")) #I don't know whether it's better than what's below
        with open(path_file, 'wb') as fout:
            pickle.dump(obj, fout)
    elif which_python == 'python2':
        with open(path_file, 'wb') as fout:
            pickle.dump(obj, fout, protocol=2)
    
    
def load_pickle(filename, 
                path_folder="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/", 
                which_python='python3'):
    path_file = str(path_folder) + str(filename) #+ ".p"
    assert which_python in ['python2', 'python3']
    if which_python == 'python3': #default
#         return pickle.load(open(path_file, "rb")) #I don't know whether it's better than what's below
        with open(path_file, 'rb') as fout:
            obj = pickle.load(fout)
    elif which_python == 'python2':
        with open(path_file, 'rb') as fout:
            obj = pickle.load(fout, encoding='latin1') # protocol=2)
    return obj


def load_from_mat_list(filename, path_folder="../../results_code/simulations_CI_BP/Better_BP/",
                       print_shapes=False):
    """
    Filename is graph_and_local_fields_{...}.mat"
    """
    path_file = str(path_folder) + str(filename) + ".mat"
    dict_save_matlab_loaded = scipy.io.loadmat(path_file) 
    list_graph = []
    
    if 'list_graph_wij' in dict_save_matlab_loaded.keys():
        list_graph_wij = dict_save_matlab_loaded['list_graph_wij'] #adjacency matrix with w_ij
        for graph_wij in list_graph_wij:
            if print_shapes:
                print(graph_wij.shape)
            graph_Jij = 2 * graph_wij - 1 #J_ij = 2*w_ij - 1
            graph = nx.from_numpy_matrix(graph_Jij) #0 means no edge (no possibility to indicate nonedge in from_numpy_matrix --> that is why I to 2*w-1 and then go back to w by doing (J+1)/2)
            for node1, node2, d in graph.edges(data=True):
                assert 'weight' in d.keys()
                d['weight'] = (d['weight'] + 1) / 2
            list_graph.append(graph)
    
    else: #factors
        list_graph_fij = dict_save_matlab_loaded['list_graph_fij'] #matrix with all f_ij
        for graph_fij in list_graph_fij:
            graph = nx.Graph()
            for i, j in itertools.product(range(len(graph_fij)), range(len(graph_fij))):
                if i > j:
                    continue
                assert np.sum(graph_fij[i,j] != graph_fij[j,i].T) == 0 #f_ji = f_ij.T
#                 print("edge ({},{}): f_ij={}, f_ji={}".format(i, j, graph_fij[i,j], graph_fij[j,i]))
                if np.sum(graph_fij[i,j] != np.ones((2,2))) != 0: #an edge exists
#                     print("add edge ({},{}) with f_ij={}".format(i,j,graph_fij[i,j]))
                    graph.add_edge(i, j, factor=graph_fij[i,j])
            list_graph.append(graph)
    
    list_M_ext = dict_save_matlab_loaded['list_M_ext'] #list of numpy elements
    list_X = [pd.DataFrame(M_ext) for M_ext in list_M_ext] #from numpy to pandas
    if print_shapes:
        for X in list_X_loaded:
            print(X.shape)
    if 'true_marginals' in dict_save_matlab_loaded.keys():
        list_y = dict_save_matlab_loaded['true_marginals'] #list of numpy elements
        list_y = [pd.DataFrame(y) for y in list_y] #from numpy to pandas
        return list_graph, list_X, list_y
    else:
        return list_graph, list_X

    

def load_from_mat(filename, 
                  path_folder="../../results_code/simulations_CI_BP/Better_BP/"):
    """
    graph_loaded, X_loaded = load_from_mat('graph_and_local_fields')
    
    Test that it does the job well:
    graph_loaded, X_loaded = load_from_mat('graph_and_local_fields')
    print(test_graph_equality(graph, graph_loaded))
    print(np.sum(np.sum(X != X_loaded)))
    """
    path_file = str(path_folder) + str(filename) + ".mat"
    dict_save_matlab_loaded = scipy.io.loadmat(path_file) #scipy.io.loadmat("../../results_code/simulations_CI_BP/Better_BP/graph_and_local_fields.mat")
    if 'graph_wij' in dict_save_matlab_loaded.keys(): #weights
        graph_wij = dict_save_matlab_loaded['graph_wij'] #adjacency matrix with w_ij
        assert np.sum(graph_wij != graph_wij.T) == 0 #symmetric weights
        #graph_wij[graph_wij == 0] = 0.5 #useless, because graph_wij has 0.5 on non-edges (contrary to before where it was 0)
        graph_Jij = 2*graph_wij - 1 #J_ij = 2*w_ij - 1
        graph = nx.from_numpy_matrix(graph_Jij) #0 means no edge (no possibility to indicate nonedge in from_numpy_matrix --> that is why I to 2*w-1 and then go back to w by doing (J+1)/2)
        for node1, node2, d in graph.edges(data=True):
            assert 'weight' in d.keys()
            d['weight'] = (d['weight'] + 1) / 2 #getting w_ij from J_ij
    else: #factors
        graph_fij = dict_save_matlab_loaded['graph_fij'] #adjacency matrix with w_ij
        graph = nx.Graph()
        for i, j in zip(range(len(graph_fij)), range(len(graph_fij))):
            if j > i:
                continue
            assert np.sum(graph_fij[i,j] != graph_fij[j,i].T) == 0 #f_ji = f_ij.T
            if np.sum(graph_fij != np.ones((2,2))) != 0: #an edge exists
                graph.add_edge(i, j, factor=graph_fij[i,j])
    M_ext = dict_save_matlab_loaded['M_ext'] #numpy
    X = pd.DataFrame(M_ext) #from numpy to pandas
    if 'true_marginals' in dict_save_matlab_loaded.keys():
        y = dict_save_matlab_loaded['true_marginals']
        y = pd.DataFrame(y)
        return graph, X, y
    else:
        return graph, X

    
def save_into_mat(filename, graph, X, 
                  path_folder="../../results_code/simulations_CI_BP/Better_BP/"):
    """
    Saves into mat file (because the unsupervised learning algorithm for alpha_ij is in Matlab)
    """
    dict_save_matlab = {'M_ext': X.to_numpy()}
    if 'weight' in list_graph[0].edges[list(list_graph[0].edges)[0]].keys():
        adj_matrix = nx.to_numpy_matrix(graph, nonedge=0.5) #matrix with {w_ij}
        dict_save_matlab['graph_wij'] = adj_matrix
    else:
        factors_matrix = np.zeros((len(graph.nodes), len(graph.nodes), 2, 2))
        node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
        for node1, node2, d in graph.edges(data=True):
            factor = d['factor']
            i_node1 = node_to_inode[node1]
            i_node2 = node_to_inode[node2]
            factors_matrix[i_node1, i_node2] = factor
            factors_matrix[i_node2, i_node1] = factor.T
        dict_save_mat['graph_fij'] = factors_matrix
    path_file = str(path_folder) + str(filename) + ".mat"
    savemat(path_file, dict_save_matlab)
    

def save_into_mat_list(filename, list_graph, list_X, list_y, 
                       path_folder="../../results_code/simulations_CI_BP/Better_BP/"):
    """
    Saves into mat file (because the unsupervised learning algorithm for alpha_ij is in Matlab)
    """
    list_X_numpy = []
    for X in list_X:
        if type(X) == pd.DataFrame: #pd.core.frame.DataFrame:
            X = X.to_numpy()
        list_X_numpy.append(X) 
#         list_X_numpy.append(X.to_numpy())
#     list_X_numpy = [X.to_numpy() for X in list_X]
    dict_save_matlab = {'list_M_ext': list_X_numpy}
    if 'weight' in list_graph[0].edges[list(list_graph[0].edges)[0]].keys():
        list_adj_matrix = [nx.to_numpy_matrix(graph, nonedge=0.5) for graph in list_graph] #matrix with {w_ij}
        dict_save_matlab['list_graph_wij'] = list_adj_matrix
    else:
        list_factors_matrix = []
        for graph in list_graph:
            factors_matrix = np.zeros((len(graph.nodes), len(graph.nodes), 2, 2)) + 1
            node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
            for node1, node2, d in graph.edges(data=True):
                factor = d['factor']
                i_node1 = node_to_inode[node1]
                i_node2 = node_to_inode[node2]
                factors_matrix[i_node1, i_node2] = factor
                factors_matrix[i_node2, i_node1] = factor.T
            list_factors_matrix.append(factors_matrix)
        dict_save_matlab['list_graph_fij'] = list_factors_matrix
    list_y_numpy = []
    for y in list_y:
        if type(y) == pd.DataFrame: #pd.core.frame.DataFrame:
            y = y.to_numpy()
        list_y_numpy.append(y) 
#         list_y_numpy.append(y.to_numpy())
#     print("list_X_numpy = {}".format(list_X_numpy))
#     print("list_y_numpy = {}".format(list_y_numpy))
    dict_save_matlab['true_marginals'] = list_y_numpy
#     print(len(list_adj_matrix), list_adj_matrix[0].shape)
#     print(len(list_X_numpy), list_X_numpy[0].shape)
    path_file = str(path_folder) + str(filename) + ".mat"
    savemat(path_file, dict_save_matlab)
 
    
def load_results_unsupervised(filename, varname, list_graph, 
                              path_folder="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/"):
    """
    #filename = 'Alpha2_one_example'
    #varname = 'Alpha1' 
    
    Transforms matrix {alpha_ij} into dictionnary {edge: alpha_edge}
    
    list_graph is needed in order to indicate the names of nodes (and thus, of the edges)
    """
    path_file = str(path_folder) + str(filename) + ".mat"
    dict_save_matlab_loaded = scipy.io.loadmat(path_file)
#     print(dict_save_matlab_loaded.keys())
    
    list_alpha_mat = - dict_save_matlab_loaded[varname]
    if len(list_alpha_mat.shape) == 2: #only one example (graph, {M_ext})
        list_alpha_mat = np.array([alpha_mat])
#     print(list_alpha_mat.shape)
        
    list_alpha_dict = []
    for alpha_mat, graph in zip(list_alpha_mat, list_graph):
        #check that edges which are not present in graph have alpha_ij=0 (or ~0 because of round issues)
        for i_node1, node1 in enumerate(list(graph.nodes)):
            for i_node2, node2 in enumerate(list(graph.nodes)):
                if (node1, node2) not in graph.edges: #Note that (node1, node2) is in graph.edges iff (node2, node1) is in graph.edges
                    assert np.abs(alpha_mat[i_node1, i_node2]) < 1e-10 #I should be 0, but what Sophie sent had tiny values, probably a matter of precision error (but still checking that it's close to 0 so that it's indeed the same graphs)
        #transform mat into dict
        alpha_dict = {}
        node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
        for (node1, node2) in graph.edges:
            i_node1 = node_to_inode[node1]
            i_node2 = node_to_inode[node2]
            alpha_dict[node1, node2] = alpha_mat[i_node1, i_node2]
            alpha_dict[node2, node1] = alpha_mat[i_node2, i_node1]
        list_alpha_dict.append(alpha_dict)
    return list_alpha_dict


def load_libdai_beliefs(data_file, i_graph, verbose=False, algo=None,
                        path_folder="../../results_code/simulations_CI_BP/Better_BP/libdai/"):
    """
    Example of data_file: 'graph_and_local_fields_random_connected_9nodes_p06_normalJ_stdweighting1_stdMext1'
    """
    if type(data_file) == str: #default
        libdai_file = 'result_' + data_file #beliefs
        list_list_dict_y_predict = load_pickle(libdai_file, path_folder=path_folder, which_python='python2')
    else:
        list_list_dict_y_predict = data_file
        
    if i_graph == 'all': #this will not load the file for each i_graph --> potentially a lot faster to execute
        return [load_libdai_beliefs(list_list_dict_y_predict, i_graph, verbose=verbose, algo=algo, path_folder=path_folder)
                for i_graph in range(len(list_list_dict_y_predict))]
        
    list_dict_y_predict = list_list_dict_y_predict[i_graph]
    #list_dict_y_predict is: [{alg: [b_node for all nodes] for all algos} for all examples]
    if algo is None:
        list_algos = list(list_dict_y_predict[0].keys())
        dict_y_predict_libdai = {algo : np.array([dict_y_predict[algo] for dict_y_predict in list_dict_y_predict])
                                 for algo in list_algos}
        if verbose:
            print(list(dict_y_predict_libdai.keys()), dict_y_predict_libdai['mf'].shape)
        return dict_y_predict_libdai #{algo: y_predict_algo for all algos}
    else:
        assert type(algo) == str
        if algo not in list_dict_y_predict[0].keys():
            print("libDAI was run for this graph but not with this model ({})".format(algo))
            if algo == 'lcbp':
                print("filling lcbp with np.nan")
#                 print("this shape", np.array([[np.nan] * len(dict_y_predict['jt']) 
#                                               for dict_y_predict in list_dict_y_predict]).shape)
                y_predict_libdai = np.array([[np.nan] * len(dict_y_predict['jt']) for dict_y_predict in list_dict_y_predict])
                return y_predict_libdai
        y_predict_libdai = np.array([dict_y_predict[algo] for dict_y_predict in list_dict_y_predict])
#         print("y_predict_libdai.shape", y_predict_libdai.shape)
        return y_predict_libdai
    

