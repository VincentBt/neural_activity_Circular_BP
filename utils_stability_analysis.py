import numpy as np
# from utils_plot_dict import plot_dict
from graph_generator import *
import sys
from utils_basic_functions import tanh_inverse
from itertools import repeat
from utils_basic_functions import from_matrix_to_dict
from utils_alpha_obj import *

def get_rank(M):
    return np.linalg.matrix_rank(M)


def get_stability_matrix(*args, **kwargs):
    if 'which_CI' not in kwargs.keys():
        print("Please give which_CI as input")
        sys.exit()
    else:
        which_CI = kwargs['which_CI']
        assert which_CI in ['BP', 'CI', 'CIbeliefs', 'CIpower'] #not implemented for other algos
        if which_CI == 'BP':
            assert 'alpha' in kwargs.keys() and kwargs['alpha'] == Alpha_obj({'alpha': 1})
        k_wargs_new = {key: val for key, val in kwargs.items() if key != 'which_CI'}
        dict_get_stability_matrix = {
            'BP': get_stability_matrix_CI,
            'CI': get_stability_matrix_CI,
            'CIbeliefs': get_stability_matrix_CIbeliefs,
            'CIpower': get_stability_matrix_CIpower
        }
        fun_get_stability_matrix = dict_get_stability_matrix[which_CI]
        return fun_get_stability_matrix(*args, **k_wargs_new)
            
    
def get_stability_matrix_CI(graph, 
                            alpha=None,
                            theta=None, damping=0):
    """
    Returns S = F'(0) for CI: m_{t+1} = F(m_t) where m is a vector of all the m_{ij}
    i.e. returns the Jacobian of the system = stability matrix (for the fixed point 0)
    theta us a dict containing the constant external inputs (local fields) - by default 0
    
    Careful: 0 is not a fixed point anymore when theta != 0 --> be careful with using theta != 0 (for theta != 0 I should be using F'(fixed_point) where fixed_point != 0 but instead it can be approximated with F(fixed_point)=fixed_point
    In general stability of the trivial fixed point without local fields (in this case =0) implicates the stability of the trivial fixed point with local fields (!= 0) - see work by Christian Knoll -----> it's enough to check the stability of the fixed point 0 with theta=0 (as it implicates the stability of the equivalent fixed point with theta != 0)
    """
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij
    w_matrix = get_w_matrix(graph)
    assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
    
    if theta is not None:
        print("Careful: theta should be Mext / 2, not Mext")
        
#     print("damping = {}".format(damping))
    alpha_dict = alpha.get_alpha_dict(graph)
    
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i #before: "graph.keys()"
    
    if which == 'general_case':
#         print("Recover f_ij from the graph (for now it's impossible as we only associate w_ij)")
#         sys.exit()
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected #G.to_undirected()
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in graph.edges: #in graph.keys():
                        f_ij = graph.edges[edge1[0], edge1[1]]['factor'] #graph[edge1[0], edge1[1]][0]
                        k_ij = f_ij[1,1]/(f_ij[1,0]+f_ij[1,1]) - f_ij[0,1]/(f_ij[0,0]+f_ij[0,1]) #(d/(c+d) - b/(a+b))
                    else:
                        f_ij = graph.edges[edge1[1], edge1[0]]['factor'] #graph[edge1[1], edge1[0]][0]
                        k_ij = f_ij[1,1]/(f_ij[0,1]+f_ij[1,1]) - f_ij[1,0]/(f_ij[0,0]+f_ij[1,0]) #(d/(b+d) - c/(a+c)) #because f_ji = f_ij.T
                    S[ind_1, ind_2] = k_ij
                    if k == j:
                        S[ind_1, ind_2] *= (1 - alpha_dict[i, j]) #alpha
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected #G.to_undirected()
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in graph.edges: #graph.keys():
                        w_ij = graph.edges[edge1[0], edge1[1]]['weight'] #graph[edge1[0], edge1[1]][0]
                    else:
                        w_ij = graph.edges[edge1[1], edge1[0]]['weight'] #graph[edge1[1], edge1[0]][0] #because we take here symmetrical weights
                    tanh_Jij = 2 * w_ij - 1
                    if theta is None:
                        S[ind_1, ind_2] = tanh_Jij #for theta=0, it gives tanh_Jij
                    else: #case where some local fields are != 0
                        S[ind_1, ind_2] = tanh_Jij * (1 - np.tanh(theta[i])**2) / (1 - tanh_Jij**2 * np.tanh(theta[i])**2)
                    if k == j:
                        S[ind_1, ind_2] *= (1 - alpha_dict[i, j])
    
    return (1-damping) * S + damping * np.identity(S.shape[0])
    
    
def get_stability_matrix_CIpower(graph, 
                                 alpha=None,
                                 theta=None, damping=0):
    """
    Computes F'(0) for CIpower (i.e. F'(x) for all messages M_ij = 0)
    
    adapted from get_stability_matrix_CI
    
    See my notes in Mooij's PhD thesis (on Mendeley)
    
    Careful: theta should be Mext / 2, not Mext !!
    """
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij
    w_matrix = get_w_matrix(graph)
    assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
    
#     if theta is not None:
#         print("Careful: theta should be Mext / 2, not Mext")
        
#     print("damping = {}".format(damping))
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i
    
    if damping == '1-alpha.T': #i.e. amount of damping used in alpha-BP, for instance
        alpha_dict = alpha.get_alpha_dict(graph)
        damping = np.array([1 - alpha_dict[j,i] for (i,j) in list_edges]).reshape((-1,1)) #careful: put in the right order (= same as for S below)
    
    #recover K_nodes from alpha (transform if needed into a dict)
    if alpha.get('K_nodes_vector') is not None:
        K_nodes_vector = alpha.get('K_nodes_vector')
        K_nodes = dict(zip(list(graph.nodes), K_nodes_vector))
    elif alpha.get('K_nodes') is not None:
        K_nodes = alpha.get('K_nodes')
    else:
        K_nodes = dict(zip(list(graph.nodes), repeat(1)))
#     print("K_nodes", K_nodes)
    
    #recover K_edges from alpha (transform if needed into a dict)
    if alpha.get('K_edges_matrix') is not None:
        K_edges_matrix = alpha.get('K_edges_matrix')
        K_edges = from_matrix_to_dict(K_edges_matrix, list(graph.nodes))
    elif alpha.get('K_edges') is not None:
        K_edges = alpha.get('K_edges')
    else:
        K_edges = dict(zip(list(graph.edges), repeat(1))) #dict(zip(list_edges, repeat(1)))
#     print("K_edges", K_edges)
        
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    if which == 'general_case':
        raise NotImplemented #not implemented yet for CIpower
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in list(graph.edges): #graph.keys():
                        w_ij = graph.edges[edge1[0], edge1[1]]['weight']
                        K_edges_ij = K_edges[edge1[0], edge1[1]]
                    else:
                        w_ij = graph.edges[edge1[1], edge1[0]]['weight'] #because we take here symmetrical weights
                        K_edges_ij = K_edges[edge1[1], edge1[0]]
                    tanh_Jij = 2 * w_ij - 1
                    tanh_Jtildeij = np.tanh(tanh_inverse(tanh_Jij) / K_edges_ij) #tanh(J_ij / K_ij) = tanh(Jtilde_ij)
                    if theta is None:
                        S[ind_1, ind_2] = K_edges_ij / K_nodes[j] * tanh_Jtildeij #for theta=0, it gives tanh_Jij
                    else: #case where some local fields are != 0
                        S[ind_1, ind_2] = K_edges_ij / K_nodes[j] * tanh_Jtildeij * (1 - np.tanh(theta[i]/K_nodes[i])**2) / (1 - tanh_Jtildeij**2 * np.tanh(theta[i]/K_nodes[i])**2)
                    if k == j:
                        S[ind_1, ind_2] *= (1 - K_nodes[i] / K_edges_ij)
    return (1-damping) * S + damping * np.identity(S.shape[0])

    
def get_stability_matrix_CIbeliefs(graph, alpha=None, damping=0):
    """
    Returns S = F'(0) for CIbeliefs: X_{t+1} = F(X_t) where X is a vector of all the B_i(t-k) (where k in [0,k_max])
    i.e. returns the Jacobian of the system = stability matrix (for the fixed point 0)
    """
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij
    w_matrix = get_w_matrix(graph)
    assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
    
    dict_alpha = alpha.get('dict_alpha_impaired') #for CIbeliefs, don't use self.get_alpha_dict because it considers all other algorithms except from CIbeliefs (for which dict_alpha_impaired is a dict {node: [alpha_{node, k} for all k]})
#     print("dict_alpha = {}".format(dict_alpha))
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i #before: "graph.keys()"
    list_nodes = list(graph.nodes)
    k_max = len(dict_alpha[list(dict_alpha.keys())[0]])
    print("k_max = {}".format(k_max)) #mistake here: it's k_max - 1 ---> is the kmax+1 below ok???
    
    if which == 'general_case':
        S = np.zeros((len(list_nodes)*(k_max+1), len(list_nodes)*(k_max+1)))
        for ind_1, (i,k1) in enumerate(itertools.product(list_nodes, range(k_max+1))):
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected
            for ind_2, (j,k2) in enumerate(itertools.product(list_nodes, range(k_max+1))):
                if k1 == 0:
                    if j in neighbors_i:
                        if (i,j) in graph.edges: #graph.keys():
                            f_ij = graph.edges[i, j]['factor'] #graph[edge1[0], edge1[1]][0]
                            k_ij = f_ij[1,1]/(f_ij[1,0]+f_ij[1,1]) - f_ij[0,1]/(f_ij[0,0]+f_ij[0,1]) #(d/(c+d) - b/(a+b))
                        else:
                            f_ij = graph.edges[edge1[1], edge1[0]]['factor'] #graph[edge1[1], edge1[0]][0]
                            k_ij = f_ij[1,1]/(f_ij[0,1]+f_ij[1,1]) - f_ij[1,0]/(f_ij[0,0]+f_ij[1,0]) #(d/(b+d) - c/(a+c)) #because f_ji = f_ij.T
                        S[ind_1, ind_2] = k_ij
#                         print(i,j,k1,k2, S[ind_1, ind_2])
                    elif (i == j) and (k2>=1):
                        S[ind_1, ind_2] = - dict_alpha[j][-k2]
#                         print("alpha",i,j,k1,k2, S[ind_1, ind_2])
                elif (i == j) and (k1 - 1 == k2): #and (k1 >=1) automatically
                    S[ind_1, ind_2] = 1 #just a copy: the function is the identity
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_nodes)*(k_max+1), len(list_nodes)*(k_max+1)))
        for ind_1, (i,k1) in enumerate(itertools.product(list_nodes, range(k_max+1))):
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected
            for ind_2, (j,k2) in enumerate(itertools.product(list_nodes, range(k_max+1))):
                if k1 == 0:
                    if j in neighbors_i:
                        if (i,j) in graph.edges: #graph.keys():
                            w_ij = graph.edges[i, j]['weight']
                        else:
                            w_ij = graph.edges[j, i]['weight'] #because we take here symmetrical weights
                        tanh_Jij = 2 * w_ij - 1
                        S[ind_1, ind_2] = tanh_Jij
#                         print(i,j,k1,k2, S[ind_1, ind_2])
                    elif (i == j) and (k2>=1):
                        S[ind_1, ind_2] = - dict_alpha[j][-k2] #alpha[-1] corresponds to k=2
#                         print("alpha",i,j,k1,k2, S[ind_1, ind_2])
                elif (i == j) and (k1 - 1 == k2): #and (k1 >=1) automatically
                    S[ind_1, ind_2] = 1 #just a copy: the function is the identity
    
    return (1 - damping) * S + damping * np.identity(S.shape[0])
    

def get_A_matrix(*args, **kwargs):
    if 'which_CI' not in kwargs.keys():
        print("Please give which_CI as input")
        sys.exit()
    else:
        which_CI = kwargs['which_CI']
        assert which_CI in dict_get_stability_matrix.keys() #not implemented for other algos
        if which_CI == 'BP':
            assert 'alpha' in kwargs.keys() and kwargs['alpha'] == Alpha_obj({'alpha': 1})
        k_wargs_new = {key: val for key, val in kwargs.items() if key != 'which_CI'}
        if ('full_' in which_CI or which_CI == 'rate_network'):
            k_wargs_new['full'] = True #indicates that there is no subtraction in the algo (i.e. F_ij(B_j), not F_ij(B_j - alpha_ij * M_ji))
        
        fun_get_A_matrix = dict_get_stability_matrix[which_CI]
        return fun_get_A_matrix(*args, **k_wargs_new)
    
    
def get_A_CIpower_matrix(graph, alpha=None, beta=None, damping=0, full=False):
    """
    Get the matrix A defined in Eq 2.11 of Mooij's PhD thesis:
    f'(nu)_{i \to j, k \to l} = A_{i \to j, k \to l} . B_{i \to j}(nu)   where nu = M / 2   (where |B_{i \to j}| <= 1)
    
    There are 2 ways to try and show that the algo converges:
    - If ||abs(A)|| < 1 (for any norm) then the algorithm converges (undamped algorithm, i.e. f'(nu)_{i \to j, i \to j} = 0)
    - If A > 0 and the spectral radius of A < 1, then the algorithm converges (undamped algorith)
    
    I think that damped version of algo can only help the algo to converge, but I'm not sure 100% (I tried to show that if x(t+1) = f(x(t)) converges then x(t+1) = (1-eps).f(x(t)) + eps.x(t) converges, but I didn't manage to show it)
    
    Careful: damping != 0 is ok, but only when looking at norm(A), not spectral_radius(A)!! Indeed, the spectral radius criterion only applies without damping (because with damping, F'(nu) !=  A * B(nu))
    With damping, the norm criterion is norm(|1-damping|*A + |damping|*Id) < 1  (with the absolute values; the damping could be outside of [0,1] if we take damping=1-alpha.T with strange alpha, but in practice we will probably have damping in [0,1])
    
    Careful: there are numerical problems sometimes for weight = 0 or 1, coming from a high J_ij (resp -J_ij): w = 1/2 + 1/2*np.tanh(J_ij) ~ 1 (resp 0), but = 1 (resp 0) for Python  ----> while trying to recover tanh_inverse(tanh_Jij) in the code, we will get inf (resp -inf)  so tanh_Jtildeij will be 1 (resp 0) no matter what K_edges is...   One solution would be to remove all w and work only with J in the graph
    """
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij
    w_matrix = get_w_matrix(graph)
    assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i
    
    if damping == '1-alpha.T': #i.e. amount of damping used in alpha-BP, for instance
        alpha_dict = alpha.get_alpha_dict(graph)
        damping = np.array([1 - alpha_dict[j,i] for (i,j) in list_edges]).reshape((-1,1)) #careful: put in the right order (= same as for S below)
        assert np.min(damping) >= 0 and np.max(damping) <= 1 #not really necessary, but just checking
    
#     print(list(graph.edges))
#     print("alpha = {}".format(alpha))
    
    #recover K_nodes from alpha (transform if needed into a dict)
    if alpha.get('K_nodes_vector') is not None:
        K_nodes_vector = alpha.get('K_nodes_vector')
        K_nodes = dict(zip(list(graph.nodes), K_nodes_vector))
    elif alpha.get('K_nodes') is not None:
        K_nodes = alpha.get('K_nodes')
    else:
        K_nodes = dict(zip(list(graph.nodes), repeat(1)))
#     print("K_nodes", K_nodes)
    
    #recover K_edges from alpha (transform if needed into a dict)
    if alpha.get('K_edges_matrix') is not None:
        K_edges_matrix = alpha.get('K_edges_matrix')
        K_edges = from_matrix_to_dict(K_edges_matrix, list(graph.nodes))
    elif alpha.get('K_edges') is not None:
        K_edges = alpha.get('K_edges')
    else:
        K_edges = dict(zip(list(graph.edges), repeat(1))) #dict(zip(list_edges, repeat(1)))
#     print("K_edges", K_edges)
    
    #recover dict beta from beta_obj
    if beta is not None:
        if beta.get('alpha') is not None:
            beta_val = beta.get('alpha')
            beta = dict(zip(list(graph.edges), repeat(beta_val)))
        else:
            raise NotImplemented
    
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
#     print("edges", list(graph.edges))
    
    if which == 'general_case':
        raise NotImplemented #not implemented yet for CIpower
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in list(graph.edges):
                        w_ij = graph.edges[i, j]['weight']
#                         K_edges_ij = K_edges[i, j]
                    else:
                        w_ij = graph.edges[j, i]['weight'] #because we take here symmetrical weights
#                         K_edges_ij = K_edges[j, i]
                    K_edges_ij = K_edges[i, j] if (i,j) in K_edges.keys() else K_edges[j, i]
                    tanh_Jij = 2 * w_ij - 1
                    if beta is None:
                        tanh_Jtildeij = np.tanh(tanh_inverse(tanh_Jij) / K_edges_ij) #tanh(J_ij / K_ij) = tanh(Jtilde_ij)
                    else:
                        beta_ij = beta[i, j] if (i,j) in beta.keys() else beta[j, i]
                        tanh_Jtildeij = np.tanh(beta_ij * tanh_inverse(tanh_Jij) / K_edges_ij) #tanh(beta_ij * J_ij / K_ij) = tanh(Jtilde_ij)
                        
                    S[ind_1, ind_2] = np.abs(K_edges_ij / K_nodes[j] * tanh_Jtildeij)
                    if k == j and full == False: #full = True means that we consider full CIpower (no subtraction)
                        S[ind_1, ind_2] *= (1 - K_nodes[i] / K_edges_ij)
    return np.abs(1 - damping) * S + np.abs(damping) * np.identity(S.shape[0])


def get_A_CIpower_approx_matrix(graph, alpha=None, beta=None, damping=0, full=False):
    """
    Inspired from function get_A_CIpower_matrix
    Only difference: beta_ij / gamma_j * tanh(J_ij / beta_ij) in CIpower becomes 1 / gamma_j * tanh(J_ij) in CIpower_approx
    """
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij  ------------> actually no need to check: no tanh_inverse function as for CIpower
#     w_matrix = get_w_matrix(graph)
#     assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i
    
    if damping == '1-alpha.T': #i.e. amount of damping used in alpha-BP, for instance
        alpha_dict = alpha.get_alpha_dict(graph)
        damping = np.array([1 - alpha_dict[j,i] for (i,j) in list_edges]).reshape((-1,1)) #careful: put in the right order (= same as for S below)
        assert np.min(damping) >= 0 and np.max(damping) <= 1 #not really necessary, but just checking
    
#     print(list(graph.edges))
#     print("alpha = {}".format(alpha))
    
    #recover K_nodes from alpha (transform if needed into a dict)
    if alpha.get('K_nodes_vector') is not None:
        K_nodes_vector = alpha.get('K_nodes_vector')
        K_nodes = dict(zip(list(graph.nodes), K_nodes_vector))
    elif alpha.get('K_nodes') is not None:
        K_nodes = alpha.get('K_nodes')
    else:
        K_nodes = dict(zip(list(graph.nodes), repeat(1)))
#     print("K_nodes", K_nodes)
    
    #recover K_edges from alpha (transform if needed into a dict)
    if full == False:
        if alpha.get('K_edges_matrix') is not None:
            K_edges_matrix = alpha.get('K_edges_matrix')
            K_edges = from_matrix_to_dict(K_edges_matrix, list(graph.nodes))
        elif alpha.get('K_edges') is not None:
            K_edges = alpha.get('K_edges')
        else:
            K_edges = dict(zip(list(graph.edges), repeat(1))) #dict(zip(list_edges, repeat(1)))
    #     print("K_edges", K_edges)
        
    #recover dict beta from beta_obj
    if beta is not None:
        if beta.get('alpha') is not None:
            beta_val = beta.get('alpha')
            beta = dict(zip(list(graph.edges), repeat(beta_val)))
        else:
            raise NotImplemented
    
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
#     print("edges", list(graph.edges))
    
    if which == 'general_case':
        raise NotImplemented #not implemented yet for CIpower_approx
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
#                     print(i,j,k,l)
                    if edge1 in list(graph.edges):
                        w_ij = graph.edges[i, j]['weight']
#                         K_edges_ij = K_edges[i, j]
                    else:
                        w_ij = graph.edges[j, i]['weight'] #because we take here symmetrical weights
#                         K_edges_ij = K_edges[j, i]
                    if full == False:
                        K_edges_ij = K_edges[i, j] if (i,j) in K_edges.keys() else K_edges[j, i]
                    tanh_Jij = 2 * w_ij - 1
                
                    if beta is not None:
                        beta_ij = beta[i, j] if (i,j) in beta.keys() else beta[j, i]
                        tanh_Jij = np.tanh(beta_ij * tanh_inverse(tanh_Jij)) #tanh(beta_ij * J_ij ) = tanh(Jtilde_ij)
                
                    S[ind_1, ind_2] = np.abs(1 / K_nodes[j] * tanh_Jij)
                    if k == j and full == False:
                        S[ind_1, ind_2] *= (1 - K_nodes[i] / K_edges_ij) #why no absolute value here??
#                     print(S[ind_1, ind_2])
                elif (i == l) and (k not in neighbors_i):
                    print("This is strange")
                    sys.exit()
    return np.abs(1 - damping) * S + np.abs(damping) * np.identity(S.shape[0])


def get_A_CI_matrix(graph, alpha=None, beta=None, damping=0, full=False):
    #checking that there won't be any numerical error, i.e., that weights are not exactly 0 or 1. It corresponds to infinite J which is impossible in neural networks, but in practice Python can round np.tanh(J) into exactly 1 --> a solution would be to work with J_ij directly instead of w_ij
    w_matrix = get_w_matrix(graph)
    assert np.min(w_matrix) != 0 and np.max(w_matrix) != 1
        
    alpha_dict = alpha.get_alpha_dict(graph)
    
    #recognize automatically whether the graph has weights or factors
    if 'weight' in list(graph.edges(data=True))[0][2].keys():
        which = 'symmetrical_with_w' #with w_ij (symmetrical case)
    else:
        which = 'general_case' #with f_ij
    
    #recover dict beta from beta_obj
    if beta is not None:
        if beta.get('alpha') is not None:
            beta_val = beta.get('alpha')
            beta = dict(zip(list(graph.edges), repeat(beta_val)))
        else:
            raise NotImplemented
    
    list_edges = list(set(graph.edges).union(set({(node2,node1) for (node1,node2) in graph.edges}))) #so that there are both i->j and j->i #before: "graph.keys()"
    
    if damping == '1-alpha.T': #i.e. amount of damping used in alpha-BP, for instance
        alpha_dict = alpha.get_alpha_dict(graph)
        damping = np.array([1 - alpha_dict[j,i] for (i,j) in list_edges]).reshape((-1,1)) #careful: put in the right order (= same as for S below)
        assert np.min(damping) >= 0 and np.max(damping) <= 1 #not really necessary, but just checking
    
    if which == 'general_case':
#         print("Recover f_ij from the graph (for now it's impossible as we only associate w_ij)")
#         sys.exit()
        raise NotImplemented
#         S = np.zeros((len(list_edges), len(list_edges)))
#         for ind_1, edge1 in enumerate(list_edges):
#             i, j = edge1
#             neighbors_i = list(graph.neighbors(i)) #because graph is undirected #G.to_undirected()
#             for ind_2, edge2 in enumerate(list_edges):
#                 k, l = edge2
#                 if (i == l) and (k in neighbors_i):
#                     if edge1 in graph.edges: #in graph.keys():
#                         f_ij = graph.edges[i, j]['factor'] #graph[i, j][0]
#                         k_ij = f_ij[1,1]/(f_ij[1,0]+f_ij[1,1]) - f_ij[0,1]/(f_ij[0,0]+f_ij[0,1]) #(d/(c+d) - b/(a+b))
#                     else:
#                         f_ij = graph.edges[j, i]['factor'] #graph[j, i][0]
#                         k_ij = f_ij[1,1]/(f_ij[0,1]+f_ij[1,1]) - f_ij[1,0]/(f_ij[0,0]+f_ij[1,0]) #(d/(b+d) - c/(a+c)) #because f_ji = f_ij.T
#                     S[ind_1, ind_2] = k_ij
#                     if k == j:
#                         S[ind_1, ind_2] *= (1 - alpha_dict[i, j]) #alpha
    
    elif which == 'symmetrical_with_w':
        S = np.zeros((len(list_edges), len(list_edges)))
        for ind_1, edge1 in enumerate(list_edges):
            i, j = edge1
            neighbors_i = list(graph.neighbors(i)) #because graph is undirected #G.to_undirected()
            for ind_2, edge2 in enumerate(list_edges):
                k, l = edge2
                if (i == l) and (k in neighbors_i):
                    if edge1 in graph.edges: #graph.keys():
                        w_ij = graph.edges[i, j]['weight'] #graph[i, j][0]
                    else:
                        w_ij = graph.edges[j, i]['weight'] #graph[j, i][0] #because we take here symmetrical weights
                    tanh_Jij = 2 * w_ij - 1
                    if beta is not None:
                        beta_ij = beta[i, j] if (i,j) in beta.keys() else beta[j, i]
                        tanh_Jij = np.tanh(beta_ij * tanh_inverse(tanh_Jij)) #tanh(beta_ij * J_ij ) = tanh(Jtilde_ij)
                    S[ind_1, ind_2] = np.abs(tanh_Jij)
                    if k == j and full == False:
                        S[ind_1, ind_2] *= (1 - alpha_dict[i, j])
    return np.abs(1 - damping) * S + np.abs(damping) * np.identity(S.shape[0]) #(1 - damping) * S + damping * np.identity(S.shape[0])  #because F'(nu) = (1-damping) * A * B(nu) + damping * Id, so |F('nu)|
    
    
dict_get_stability_matrix = {
    'BP': get_A_CI_matrix,
    'CI': get_A_CI_matrix,
    'CIpower': get_A_CIpower_matrix,
    'full_CIpower': get_A_CIpower_matrix,
    'CIpower_approx': get_A_CIpower_approx_matrix,
    'CIpower_approx_tanh': get_A_CIpower_matrix,
    'full_CIpower_approx_tanh': get_A_CIpower_matrix,
    'CIpower_approx_approx_tanh': get_A_CIpower_approx_matrix,

    'full_CIpower_approx_approx_tanh': get_A_CIpower_approx_matrix, #impossible to guarantee convergence: no parameter K_edges --> in fact ok with the parameter K_nodes orbeta
    'rate_network': get_A_CIpower_approx_matrix, #impossible to guarantee convergence: no parameter K_edges --> in fact ok with the parameter K_nodes or beta
    'full_CIpower_approx': get_A_CIpower_approx_matrix #impossible to guarantee convergence: no parameter K_edges --> in fact ok with the parameter K_nodes or beta
}
    
    
def get_Knodes_Kedges_val_convergence_CIpower(graph):
    print("get_Knodes_Kedges_val_convergence_CIpower is a deprecated function - use instead get_Knodes_Kedges_val_convergence_algo")
    sys.exit()
    
    
def get_Knodes_Kedges_val_convergence_algo(graph, which_CI):
    """
    Returns a value K such that, for Knodes = K and Kedges = K, then the algorithm converges (CIpower or CIpower_approx)
    Such a value exists, because for K --> inf then the algorithm converges (a priori for any damping, but I couldn't show it)
    
    The returned value serves as initialization for Pytorch (because we want something which converges, 
    even if the approximated marginals are not so good)
    
    Example: 
    which_CI = 'CIpower'
    data_file = 'graph_and_local_fields_randomconnected_9nodes_p02_normalJ'
    list_graph, list_X, list_y = load_from_mat_list(data_file)
    graph = list_graph[3]
    K_init = get_Knodes_Kedges_val_convergence_CIpower(graph)
    print(K_init)
    """
    assert which_CI in dict_get_stability_matrix.keys()
    assert which_CI not in ['full_CIpower_approx_approx_tanh', 'rate_network', 'full_CIpower_approx']
    K = 1
    converge = 'unknown'
    while converge != True:
        alpha = Alpha_obj({'K_nodes': dict(zip(list(graph.nodes), repeat(K))),
                           'K_edges': dict(zip(list(graph.edges), repeat(K)))
                          })
        M = get_A_matrix(graph, which_CI=which_CI, alpha=alpha)

        if get_spectral_radius(M) < 1: #get_largest_singular_value(M) < 1: The spectral radius is always lower than the largest singular value, so the criterion is more powerful  -  and it is valid to use the spectral radius criterion because K_nodes/K_edges <= 1, making M non-negative (this is the condition for the application of the criterion)
            converge = True
            return K
        K = K + 1
        
        
def get_Knodes_val_convergence_algo(graph, which_CI):
    """
    Returns a value K such that, for Knodes = K and Kedges = 1, then the algorithm converges (CIpower or CIpower_approx)
    
    I'm not sure that such a value exists for all algorithms though   (but in practice, this function seems to return something in all cases... maybe because my weights are not too high)
    """
    assert which_CI in dict_get_stability_matrix.keys()
    K = 1
    converge = 'unknown'
    while converge != True:
        alpha = Alpha_obj({'K_nodes': dict(zip(list(graph.nodes), repeat(K))),
                           'K_edges': dict(zip(list(graph.edges), repeat(1)))
                          })
        M = get_A_matrix(graph, which_CI=which_CI, alpha=alpha)

        if get_spectral_radius(M) < 1: #get_largest_singular_value(M) < 1: The spectral radius is always lower than the largest singular value, so the criterion is more powerful  -  and it is valid to use the spectral radius criterion because K_nodes/K_edges <= 1, making M non-negative (this is the condition for the application of the criterion)
            converge = True
            return K
        K = K + 1
        

def get_beta_val_convergence_algo(graph, which_CI):
    """
    Returns a value beta_val such that, for (beta=beta_val, K_nodes=1, Kedges=1), then the algorithm converges (CIpower or CIpower_approx)
    
    tanh_Jtildeij = np.tanh(tanh_inverse(tanh_Jij) / K_edges_ij) #tanh(J_ij / K_ij) = tanh(Jtilde_ij)
    2*w-1 = tanh(beta*alpha*J_ij) = tanh(beta*alpha*arctanh(2*wgraph_ij-1)) for CIpower models
    2*w-1 = tanh(beta*J_ij) = tanh(beta*arctanh(2*wgraph_ij-1)) for CIpower_approx models
    """
    assert which_CI in dict_get_stability_matrix.keys()
    beta_val = 1
    converge = 'unknown'
    while converge != True:
        beta = Alpha_obj({'alpha': beta_val})
        if 'full' not in which_CI and 'rate_network' not in which_CI: #which_CI == 'BP':
            alpha = Alpha_obj({'alpha': 1})
        else:
            alpha = Alpha_obj({})
#             raise NotImplemented #maybe alpha = Alpha_obj({})?
#             sys.exit()
        M = get_A_matrix(graph, which_CI=which_CI, alpha=alpha, beta=beta)

        if get_spectral_radius(M) < 1: #get_largest_singular_value(M) < 1: The spectral radius is always lower than the largest singular value, so the criterion is more powerful  -  and it is valid to use the spectral radius criterion because K_nodes/K_edges <= 1, making M non-negative (this is the condition for the application of the criterion)
            converge = True
            return beta_val
        beta_val = beta_val * 0.95
    
    
def plot_eigenvalues_from_graph(graph, 
                                alpha=None,
                                order='1', damping=0):
    """
    Can help to determine the stability of CI on some graph 
    (where CI is exact or approximated with rate model, at different possible orders)
    """
    #run some checks
    assert not((alpha is None) and (order != '0'))
    assert order in ['exact', '0', '1', 'inf'] #exact means stability matrix (F'(0) where M^{t+1} = F'(M^t)); '0'/'1'/'inf' mean the order of the approximation (while approximating the system with a "rate network" i.e. without M but only with B)
    if order != 0: #checking that whether alpha is given, or (alpha_c,alpha_d) is given, or dict_alpha_impaired is given
        assert alpha is not None
    # N = 100
    # J = np.random.normal(size=(N,N), scale=1/np.sqrt(N)) #random Gaussian matrix
    if order == 'exact':
        S = get_stability_matrix_CI(graph, alpha=alpha, 
                                    damping=damping)
        plot_eigenvalues(S)
    else:
        print("This is an approximation (order = {})".format(order))
        J = get_adjacency_matrix(graph)
        J_corrected = get_corrected_matrix(J, alpha=alpha, order=order, graph=graph)
        plot_eigenvalues(J_corrected)
    
    
def plot_eigenvalues(M=None, eig_values=None, show=True, figure=True):
    """
    M is the connectivity matrix (or the stability matrix)
    """
    assert ((M is not None) or (eig_values is not None)) and not((M is not None) and (eig_values is not None)) #= exclusive OR
    
    #compute the eigenvalues of M
    if M is not None: #otherwise eig_values are already given
        eig_values = np.linalg.eig(M)[0]
    eig_values_real = eig_values.real
    eig_values_imag = eig_values.imag

    #plot the eigenvalues of M
    if figure:
        plt.figure(figsize=(6,6))
    plt.scatter(eig_values_real, eig_values_imag, color='red', s=5)
    #plot the unity circle
    def circle(angle):
        return (np.cos(angle), np.sin(angle))
    angle_all = np.linspace(0, 2*np.pi, 100)
    x_circle, y_circle = circle(angle_all)
    plt.plot(x_circle, y_circle, color='black')
    if show:
        plt.show()


def get_minimum_damping_BP(graph):
    """
    Returns the minimum value of damping for which all eigenvalues of F'(0) 
    lie inside the unity circle
    (where F is the function evolution of BP: M_{t+1} = F(M_{t}))
    
    Consequence (if we assume that Mext != 0 helps for convergence, see Knoll's work):
    BP will converge, whatever Mext is
    
    Note that as Mext helps for convergence, it is probable that a lower damping will
    be enough for BP to converge under the Mext given for training (and test).
    
    Example:
    damping_min = get_minimum_damping_BP(graph)
    S = get_stability_matrix_CI(graph, alpha=Alpha_obj({'alpha': 1}), damping=damping_min)
    eigenvals = np.linalg.eigvals(S)
    plot_eigenvalues(eig_values=eigenvals, show=False)
    plt.xlim(0.5,1,0.5)
    plt.ylim(-1,1,0.5)
    plt.show()
    """
    S = get_stability_matrix_CI(graph, alpha=Alpha_obj({'alpha': 1}), damping=0)
    eigenvals = np.linalg.eigvals(S)
    if np.max(eigenvals.real) > 1:
        return None #no damping value can make the spectral radius be < 1
#     plot_eigenvalues(eig_values=eigenvals)
#     print(eigenvals.shape)
    eigenvals = eigenvals[eigenvals.real < 1]
#     plot_eigenvalues(eig_values=eigenvals)
    imag = eigenvals.imag
    real = eigenvals.real
#     print(np.all(np.round(np.abs(eigenvals) - np.sqrt(imag**2 + real**2), 8) == 0))
    damping = 1 - 2 * np.cos(np.arctan(imag/(1-real))) / np.sqrt(imag**2 + (1-real)**2)
#     print(damping[damping > 0])
    return max(np.max(damping), 0)
        
    
def get_corrected_matrix(J, 
                         alpha=None, 
                         order='1', graph=None):
    """
    order: order of the linearization of CI (0 or 1 or inf) to have something similar to a "rate network"
    
    For non-uniform alpha (i.e. alpha is not provided or (alpha_c, alpha_d) is provided with alpha_c != alpha_d) then graph needs to be provided as input to form alpha_matrix (with corresponding edges with J)
    """
    assert order in ['0', '1', 'inf']
    
    if order == '0': #as if alpha was = 0
        return J
    
    #Other cases: order != '0'
    assert alpha is not None
    alpha.check() #checking that some information about alpha is given
    if (alpha.get('alpha_c') is not None) and (alpha.get('alpha_d') is not None):
        alpha_c = alpha.get('alpha_c')
        alpha_d = alpha.get('alpha_d')
        if alpha_c == alpha_d:
            alpha = Alpha_obj({'alpha': alpha_c})
        else:
            print("alpha_c != alpha_d: not implemented") #if alpha_c != alpha_d, then we need to know the alpha_ij for each directed edge (i,j), thus we need to know the orientation of each edge: use graph ---> TODO: define alpha_matrix based on graph and dict_alpha_impaired
            assert alpha_c == alpha_d
        
    if alpha.get('alpha') is not None:
        alpha = alpha.get('alpha')
        if order == '1': #order 1 in alpha
            return J - alpha * np.diag(np.diagonal(J.dot(J))) #d= np.sum(J * J.T, axis=1)
        elif order == 'inf':
            J_hat = J / (1 - alpha**2 * J * J.T)
            J_hat_hat = np.diag(np.sum(J * J.T / (1 - alpha**2 * J * J.T), axis=1))
            return J_hat - alpha * J_hat_hat
    elif alpha.get('dict_alpha_impaired') is not None:
        print("not implemented") #TODO: define alpha_matrix based on graph and dict_alpha_impaired
        assert False
    else:
        print("not implemented")
#         if order == '1': #order 1 in alpha
#             return J - np.diag(np.diagonal(J.dot(J * alpha_matrix))) #TODO: check that the alpha_matrix is here indeed
#         else:
#             #TODO (but I think that there is no formula in the case where alpha_ij != alpha_ji)
        assert False
        
        
def plot_hist_real_part_eigenvalues(J, list_alpha, order='1'):
    """
    input: 'order' is the order where we stop writing the imbricated messages in CI
    Non-applicable to non-uniform alpha
    """
    assert order in ['1', 'inf']
    for alpha in list_alpha:
        J_corrected = get_corrected_matrix(J, alpha)
        #compute and plot the eigenvalues of J_corrected
        eig_values = np.linalg.eig(J_corrected)[0]
        eig_values_real = eig_values.real
        eig_values_imag = eig_values.imag
        sns.distplot(eig_values_real, hist = False, kde = True,
                     kde_kws = {'linewidth': 3}, #kde_kws = {'shade': True, 'linewidth': 3}, 
                     label = 'alpha = {}'.format(alpha)) #color='darkblue'
#         plt.hist(eig_values_real, label='alpha = {}'.format(alpha), alpha=0.5)
    plt.legend()
    plt.xlabel('Re(eigenvalue)')
    plt.ylabel('distribution')
    plt.show()


def upper_bound_largest_eigenvalue(M):
    """
    Gives quickly an upper bound of the largest eigenvalue (with having to compute them)
    Corrolary of Gershgorin circle theorem - see https://math.stackexchange.com/questions/2005314/link-between-the-largest-eigenvalue-and-the-largest-entry-of-a-symmetric-matrix
    See also Corollary 2.4 and 2.5 (+ maybe other parts in Ch.2) from Mooij's PhD thesis for additional criteria
    """
    assert M.shape[0] == M.shape[1] #check that M is a square matrix
    M1 = np.abs(M.copy())
    for i in range(M.shape[0]):
        M1[i,i] = M[i,i]
    upper_bound_1 = np.max(np.sum(M1, axis=1))
    if upper_bound_1 < 1:
        print("Re(largest_eigenvalue)<1")
    #weaker result than above
    upper_bound_2 = M.shape[0]
    if upper_bound_2 < 1:
        print("Re(largest_eigenvalue)<1")
    
    
def lower_bound_largest_eigenvalue(M):
    """
    Gives quickly a lower bound of the largest eigenvalue (with having to compute them)
    Corrolary of Gershgorin circle theorem - see https://math.stackexchange.com/questions/2005314/link-between-the-largest-eigenvalue-and-the-largest-entry-of-a-symmetric-matrix
    See also Corollary 2.4 and 2.5 (+ maybe other parts in Ch.2) from Mooij's PhD thesis for additional criteria
    """
    lower_bound = np.max(np.diagonal(M))
    if lower_bound > 1:
        print("Re(largest_eigenvalue)>1")
    M1 = - np.abs(M.copy())
    for i in range(M.shape[0]):
        M1[i,i] = M[i,i]
    lower_bound_1 = np.min(np.sum(M1, axis=1))
    if lower_bound_1 > 1:
        print("Re(largest_eigenvalue)>1")
    
    
def get_spectral_radius(M):
    eigenvals = np.linalg.eigvals(M)
    spectral_radius = np.max(np.abs(eigenvals))
    return spectral_radius
    
    
def get_largest_singular_value(M):
    return np.max(np.linalg.svd(M)[1])
    
    
def check_BP_converges(list_graph, damping=0, plot=True):
    """
    Checking whether BP converges for all the graphs in list_graph
    (or rather looking whether the spectral radius is <1, which is a sufficient condition for convergence of BP for any Mext)
    """
    converges = True
    for i_graph, graph in enumerate(list_graph):
#         S = get_stability_matrix_CI(graph, alpha=Alpha_obj({'alpha': 1}), damping=damping)
        S = get_A_CIpower_matrix(graph, alpha=Alpha_obj({'alpha':1}), damping=damping)
        spectral_radius = get_spectral_radius(S)
#         eigenvals = np.linalg.eigvals(S)
# #     eigenvals_real = eigenvals.real
# #     eigenvals_imag = eigenvals.imag
#         spectral_radius = np.max(np.abs(eigenvals))
        if spectral_radius >= 1:
            print("i_graph = {}: BP might not converge (spectral radius >=1)".format(i_graph)) #rather than convergence, it means that 0 (fixed point of BP without local fields) is not stable - there could be convergence to other fixed points or frustration
            eigenvals = np.linalg.eigvals(S)
            if plot:
                plot_eigenvalues(eig_values=eigenvals)
            print("max(Re(eigenvalue)) = {}".format(np.max(eigenvals.real)))
            converges = False
    return converges