#This file allows to simulate BP (or variants like Circular BP and Fractional BP) for probability distributions with pairwise factors and binary variables.


# from jax import jit #I removed that as it is not speeding up the computations
# import jax.numpy as np #I removed that as it is not speeding up the computations
# import numpy as onp
import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from operator import xor
from graph_generator import get_all_oriented_edges, get_w_matrix
from itertools import repeat
from utils_alpha_obj import *
import time
from utils_basic_functions import sig


def F(x, w):
    print("function F is deprecated: use F_w (or F_f) instead")
    return F_w(x, w)

#@jit
def F_w(x, w):
    """
    Compute F_x = F(x, w)
    
    Previously called F
    
    MAYBE INSTEAD I CAN TAKE PRE-BUILT FUNCTIONS (E.G. SIGMOID) HERE: https://github.com/google/jax/blob/master/jax/experimental/stax.py (it could be faster)
    Note that we have: F(x) = 2 * artanh( (2*w_ij-1) * tanh(x/2) )
    """
#     print("x = {}, w = {}".format(x, w))
#     print("min(x) = {}, max(x) = {}".format(np.min(x), np.max(x)))
#     print("min(w) = {}, max(w) = {}".format(np.min(w), np.max(w)))
#     print("F:  x.shape = {}, w.shape = {}".format(x.shape, w.shape))
#     print("F:  x.type = {}, w.type = {}".format(type(x), type(w)))
    exp_x = np.exp(x)
#     print("min(exp_x) = {}, max(exp_x) = {}".format(np.min(exp_x), np.max(exp_x)))
#     print("denominator = \n{}".format((1-w)*exp_x + w))
#     print("min(denominator) = {}, max(denominator) = {}".format(np.min((1-w)*exp_x + w), np.max((1-w)*exp_x + w)))
#     print("F_w with log     = {}, {}".format(np.min(np.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))), np.max(np.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w)))))
#     print("F_w with arctanh = {}, {}".format(np.min(2 * np.arctanh( (2*w-1) * np.tanh(x/2) )), np.max(2 * np.arctanh( (2*w-1) * np.tanh(x/2) ))))
    return np.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))
#     return 2 * np.arctanh( (2*w-1) * np.tanh(x/2) ) #same thing as above, though it is around 40% slower

def F_w_power(x, w, p):
    """
    Compute F_x = F(x, factor^p), where factor = [[w,1-w],[1-w,w]]
    """
    return F_f(x, np.array([[w, 1-w],[1-w, w]])**p)
    #return F_w(x, transf_w_power(w, p)) #slightly more intense computationnally (I checked)

def dF_w(x, w):
    """
    Compute the derivative (w.r.t. x) of F(x, w)
    dF/dx = (2w-1) * (1 - tanh(x/2)^2) / (1 - (2w-1)^2 * tanh(x/2)^2)
    I checked the formula (+ by approximating numerically dF_w(x,w) ~ (F_w(x+dx,w) - F_w(x,w) ) / dx --> ok
    """
    sqr_tanh_xdiv2 = np.tanh(x/2)**2
    return (2*w-1) * (1 - sqr_tanh_xdiv2) / (1 - (2*w-1)**2 * sqr_tanh_xdiv2)
    
def F_f(x, factor):
    """
    Compute F_x = F(x, factor)
    MAYBE INSTEAD I CAN TAKE PRE-BUILT FUNCTIONS (E.G. SIGMOID) HERE: https://github.com/google/jax/blob/master/jax/experimental/stax.py (it could be faster)
    """
    exp_x = np.exp(x)
    F_x = np.log((factor[1,1]*exp_x + factor[0,1]) / (factor[1,0]*exp_x + factor[0,0]))
    return F_x
#     try:
#         F_x = np.log((factor[1,1]*exp_x + factor[0,1]) / (factor[1,0]*exp_x + factor[0,0]))
#         return F_x
#     except RuntimeWarning:
#         print(x, factor)

def F_f_power(x, factor, p):
    """
    Compute F_x = F(x, factor^p)
    """
    return F_f(x, factor**p)

def F_w_approx_tanh(x, w):
    """
    Linearizes F_w(x) ~ 2 * (2*w-1) * tanh(x/2)   
    That is the same as true F but without the arctanh: indeed, F(x) = 2 * arctanh( (2*w-1) * tanh(x/2))
    (the approximation is really good for w<0.7, and even for w~0.7 it starts to fail only for x>2)
    """
    return 2 * (2*w-1) * np.tanh(x/2)

def F_w_approx_linear(x, w):
    """
    Linearizes F_w(x) ~ (2*w-1) * x  
    """
    return (2*w-1) * x

def F_w_approx(x, w):
    """
    Approximation of F_x = F(x, w) ~ a + b.p (where p=sig(x) is the probability as x is the log-odds)
    """
    eps_1 = w - 0.5
    eps_2 = -(w - 0.5)
    eps_3 = -(w - 0.5)
    eps_4 = w - 0.5
    p = 1 / (1 + np.exp(-x)) #p = sig(x)
    
    #Approx where bounds are fitted but the approx around x=0 is bad (because the arbitrary sigmoid function is fitted based on the values at -inf and +inf). Arbitrary sigmoid function which is designed to fit well the curve for x=-inf and +inf. But this function often doesn't fit well for x close to 0
#     low = np.log((0.5+eps_2) / (0.5+eps_1))
#     high = np.log((0.5+eps_4) / (0.5+eps_3))
#     return low + (high-low)*p
    #Approx which fits well for low x but not well for the bounds (except if all eps_i are small). Based on the Taylor expansion for low eps_i. This function often doesn't fit well for strong |x| (unless all |eps_i|<0.05 for instance)
    return 2*(eps_2-eps_1) + p*2*(eps_4-eps_2-eps_3+eps_1) #= - 4*eps_1 + p*8*eps_1

def F_f_approx(x, f):
    """
    Approximation of F_x = F(x, factor)
    """
    eps_1 = f[0,0]-0.5
    eps_2 = f[0,1]-0.5
    eps_3 = f[1,0]-0.5
    eps_4 = f[1,1]-0.5
    p = 1 / (1 + np.exp(-x)) #p = sig(x)
    
    #Approx where bounds are fitted but the approx around x=0 is bad (because the arbitrary sigmoid function is fitted based on the values at -inf and +inf)
#     low = np.log((0.5+eps_2) / (0.5+eps_1))
#     high = np.log((0.5+eps_4) / (0.5+eps_3))
#     return low + (high-low)*p
    #Approx which fits well for low x but not well for the bounds (except if all eps_i are small)
    return 2*(eps_2-eps_1) + p*2*(eps_4-eps_2-eps_3+eps_1)


def transf_w_power(w, p):
    """
    p = 1/beta
    Get tanh(J/beta) from tanh(J)
    (2*w-1 = tanh(J) vs 2*w_new-1 = tanh(J/beta))
    
    tanh(x) = 2*sig(2*x)-1 thus w = sig(2*J), respectively w_new = sig(2*J/beta)
    --> w_new = sig(2/beta * J) with J = sig^{-1}(w) / 2  with sig^{-1}(x) = log(x/(1-x))
    --> w_new = sig(1/beta * log(w/(1-w)))
    """
    return sig(p * np.log(w/(1-w))) #if returns 0 or 1, possible approximation of sig(x): 1-exp(-x) if x->+inf, and exp(x) if x->-inf (but I don't know if the result would be different from 0, resp. 1 ...


# def transf_w_power(w, p):
#     """
#     Finding the w_new such that w_new / (1-w_new) = ( w / (1-w) )^p
#     """
#     ratio = w / (1-w)
#     ratio_power = ratio**p
# #     print("ratio_power = {}".format(ratio_power))
#     w_power = ratio_power / (1 + ratio_power)
#     print("w_power = {}".format(w_power))
#     print("min(w_power) = {}, max(w_power) = {}".format(np.min(w_power), np.max(w_power)))
#     if np.max(ratio_power) == np.inf:
#         print("Numerical problem!")
#         w_power_2 = sig(p * np.log(w/(1-w))) #2nd method to compute w_power (--> better)
#         print("Alternative computation: {}".format(w_power_2))
#         print("checking on one example: for w = {} and p = {}: {} vs {}".format(w[0,1], p[0,1], w_power[0,1], w_power_2[0,1]))
#     return w_power


# class Graph:
#     def __init__(self, graph):
#         super().__init__()
        
# # #         self.nodes = np.unique(list(graph.keys())) #with numpy
# #         self.nodes = set(list(itertools.chain.from_iterable(graph.keys()))) #with jax.numpy
        
#         self.n = len(self.nodes)
        
# # #         self.edges = []
# # #         for (i, j) in graph.keys():
# # #             self.edges.append((i, j))
# #         #better (?) to have a dictionnary of all possible pairs, with True if connection
        
# #         self.neighbors = {}
# #         for i in self.nodes:
# #             self.neighbors[i] = [j for j in self.nodes if (((i,j) in graph) or ((j,i) in graph))]
 

def transpose(M):
    """
    Only inverts the first two dimensions: the shape goes from (i1,i2,i3,...,in) to (i2,i1,i3,...,in)
    """
    if len(M.shape) == 2:
        return M.T
    elif len(M.shape) == 3:
        return M.transpose((1,0,2))
    else:
        raise NotImplemented
    


class Network:
    
    def __init__(self, graph, M_ext, 
                 alpha=None, w=None, w_input=None,
                 damping=0, keep_history_beliefs=False, keep_history_messages=False,
                 which_CI='CI',
                 parallel_CI=True, parallel_Mext=True,
                 niter=100
                ):
        """
        damping=0 corresponds to BP/CI : M_new = (1-damping)*F(M_old) + damping*M_old
        Note that damping=1-dt if we write M_new = M_old + dt*(F(M_old) - M_old) i.e. dM/dt = - M_old + F(M_old)   (tau=1/dt)
        
        graph is undirected (has type Graph) but has information about the directionality ('up' or 'down' associated to each undirected edge (node1,node2), i.e. (node2,node1) does not exist in graph)
        
        input 'alpha': One can include uniform alpha, different alpha_c and alpha_d, non-uniform dict_alpha_impaired (of type {edge:alpha} or  {edge:(alpha_c,alpha_d)} (with all edges for which alpha is impaired - if an edge isn't indicated in the dictionnary but exists in the graph then it means alpha=1 for this edge)
        
        parallel_CI indicates whether updates in CI are made in parallel or not (M_ij^new = F(M_ij^old) for all (i,j)) 
        ---> M is a matrix of size n_nodes*n_nodes
        parallel_CI = True makes the function 5 times faster for small networks (probably more for bigger networks) by using matrices M_ij
        
        parallel_Mext indicates whether updates in CI are made in parallel on the examples (same graph but different M_ext)
        ---> M is a matrix of size n_nodes*n_nodes*n_examples
        parallel_Mext can be True only if parallel_CI is True
        """
        super().__init__()

        assert not(parallel_Mext == True and parallel_CI == False) #parallel_Mext can be True only if parallel_CI is True
        
        #with_factors is inferred directly based on graph
        with_factors = 'factor' in graph.edges[list(graph.edges)[0]].keys() #if False, weights
        
        alpha.check()
        
        self.graph = graph
        self.damping = damping
        self.parallel_CI = parallel_CI
        self.keep_history_beliefs = keep_history_beliefs
        self.keep_history_messages = keep_history_messages
        self.parallel_Mext = parallel_Mext
        
        #Initiate the external messages
        if self.parallel_Mext == False:
            ex_Mext = list(M_ext.values())[0] #M_ext[list(M_ext.keys())[0]]
        else:
            self.n_examples = len(M_ext)
#             print("n_examples = len(M_ext) = {}".format(n_examples))
            ex_Mext = list(M_ext[0].values())[0] #M_ext is a list of dictionnaries: M_ext = [{node: [M_ext_ex[node][t] for all t]} for M_ext_ex in M_ext]
        self.constant_Mext = isinstance(ex_Mext, int) or isinstance(ex_Mext, np.int64) or isinstance(ex_Mext, float)
        assert not(self.constant_Mext == False and self.parallel_Mext == True) #TODO: implement (if possible) - probably requires to add an extra dimensions to all matrices
        if self.constant_Mext: #M_ext is constant over the whole simulation (M_ext = {node: M_ext_node})
            # print("M_ext is constant")
            self.T = max(len(self.graph) * 2, niter) #80) #this is arbitrary - it is possible that BP/CI does not have time to fully converge with that number of iterations
            if self.parallel_Mext == False:
                self.M_ext = {node: M_ext.get(node, 0) for node in self.graph.nodes} #self.M_ext = {node: [M_ext.get(node, 0)]*self.T for node in self.graph.nodes}
            else:
                self.M_ext = {node: [Mext_ex.get(node, 0) for Mext_ex in M_ext] for node in self.graph.nodes}
            #introduce a bit of noise at the beginning
# #             np.random.seed()
#             np.random.seed(0) #always the same noise, for the simulation to be reproducible (and functions using Network like diff_CI_true to be reproducible)
#             T_perturb = 4
#             for key in self.M_ext.keys():
#                 self.M_ext[key][:T_perturb] = self.M_ext[key][:T_perturb] + np.random.normal(scale=0.6, size=T_perturb)
        else: #M_ext varies with time (M_ext = {node: [M_ext_node[t] for t in range(T)]}, or M_ext = [{node: [M_ext_ex[node][t] for t in range(T)]} for M_ext_ex in M_ext])
            # print("M_ext varies with time")
            self.T = len(M_ext[list(M_ext.keys())[0]])
            self.M_ext = {node : M_ext.get(node, [0]*self.T) for node in self.graph.nodes} #because M_ext only has keys corresponding to stimulated nodes only (if for instance n_stimulated_nodes=1)
#         print("self.M_ext created", self.M_ext)
        #change M_ext using K_nodes
#         print("alpha = {}".format(alpha))
        if 'power' in which_CI and ((alpha.get('K_nodes') is not None) or (alpha.get('K_nodes_vector') is not None)):
            if alpha.get('K_nodes') is not None:
                K_nodes_dict = alpha.get('K_nodes')
            else:
                K_nodes_dict = dict(zip(list(graph.nodes), alpha.get('K_nodes_vector')))
#             print("K_nodes_dict", K_nodes_dict)
#             print("Mext", self.M_ext)
            self.M_ext = {node: val / K_nodes_dict[node] for node, val in self.M_ext.items()} #Be careful: by modifying M_ext, M_ext is not the true M_ext anymore. As a consequence, if it's used only to compute the beliefs (by running CI or another algorithm) then it's fine. On the contrary, if it's recovered (Network.M_ext) or used for something else (Network() instantiated and then used for CI,CIpower,...etc), then I should change the code and take the powers into account only when updating the beliefs (with M_ext)
        if w_input is not None: #multiplying M_ext by the input weights
            assert type(w_input) not in [list, int, float] #not list but np.array (to be able to do the operation below)
            if type(w_input) != dict: #w_input is a vector
                w_input = dict(zip(list(graph.nodes), w_input))
            self.M_ext = {node: np.array(val) * w_input[node] for node, val in self.M_ext.items()} #self.M_ext = {node: list(np.array(val) * w_input[node]) for node, val in self.M_ext.items()}
        if self.parallel_CI: #change M_ext[node][t] into M_ext[t][node] in order to catch M_ext[t] easily
            assert list(self.M_ext.keys()) == list(self.graph.nodes)
            if self.constant_Mext == False:
                self.M_ext = np.array([np.array([self.M_ext[node][t] for node in self.M_ext.keys()]) for t in range(self.T)]) #{t: np.array([self.M_ext[node][t] for node in self.M_ext.keys()]) for t in range(len(self.M_ext[list(self.M_ext.keys())[0]]))}
#                 print("M_ext.shape = {}".format(self.M_ext.shape))
            else:
                M_ext_t = np.array(list(self.M_ext.values())) #np.array([self.M_ext[node][0] for node in self.M_ext.keys()]) #with t=0 (because constant M_ext)
                self.M_ext = M_ext_t #{t: M_ext_t for t in range(len(self.M_ext[list(self.M_ext.keys())[0]]))}
#                 print("M_ext.shape = {} (the number of examples should be the last number)".format(self.M_ext.shape))
                if self.parallel_Mext == True:
                    assert self.M_ext.shape[-1] == self.n_examples
#         print("created M_ext", self.M_ext)
        
        #Define the connections weights (if w != None, then it represents the weights of the graph: it's not a multiplicative coefficient)
        self.with_factors = with_factors
        list_models_change_weights = ['CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh']
        if self.parallel_CI == False:
            assert not(which_CI in list_models_change_weights and alpha.get('K_edges_matrix') is not None) #K_edges should be != None (but K_edges_matrix should be None)
            if self.with_factors == False:
                if w is None:
                    self.w = {}
                    for node1, node2, d in graph.edges(data=True):
                        w_edge = d['weight']
                        if which_CI in list_models_change_weights and alpha.get('K_edges') is not None:
                            K_edges = alpha.get('K_edges')
                            w_edge = transf_w_power(w_edge, 1 / K_edges[node1, node2]) #Be careful: by setting self.w this way, self.w are not the true weights. As a consequence, if it's used only to compute the beliefs (by running CI or another algorithm) then it's fine. On the contrary, if it's recovered (Network.w) or used for something else (Network() instantiated and then used for CI,CIpower,...etc), then I should change the code and take the powers into account only when updating the messages
                        self.w[node1, node2] = w_edge
                        self.w[node2 ,node1] = w_edge
                else:
                    assert which_CI not in list_models_change_weights #not implemented (do the transformation)
                    self.w = {}
                    if w.get('K_edges') is not None:
                        w_unoriented = w.get('K_edges')
                        assert list(w_unoriented) == list(graph.edges)
                        for node1, node2 in graph.edges:
                            w_edge = w_unoriented[node1, node2]
                            self.w[node1, node2] = w_edge
                            self.w[node2, node1] = w_edge
                    elif w.get('dict_alpha_impaired') is not None:
                        w_oriented = w.get('dict_alpha_impaired')
                        assert list(w_oriented) == list(get_all_oriented_edges(graph.edges))
                        self.w = w_oriented
            else:
                if w is None:
                    self.factor = {}
                    for node1, node2, d in graph.edges(data=True):
                        factor = d['factor']
                        if which_CI in list_models_change_weights and alpha.get('K_edges') is not None:
                            K_edges = alpha.get('K_edges')
                            factor = factor**(1 / K_edges[node1, node2]) #Be careful: by setting self.factor this way, self.factor are not the true factors. As a consequence, if it's used only to compute the beliefs (by running CI or another algorithm) then it's fine. On the contrary, if it's recovered (Network.factor) or used for something else (Network() instantiated and then used for CI,CIpower,...etc), then I should change the code and take the powers into account only when updating the messages
                        self.factor[node1, node2] = factor
                        self.factor[node2, node1] = factor.T
                else:
                    raise NotImplemented
        else: #parallel_CI == True:
            assert not('power' in which_CI and alpha.get('K_edges') is not None) #K_edges_matrix should be != None (but K_edges should be None)
            if self.with_factors == False:
                #create the matrix of weights
                if w is None:
                    self.w = get_w_matrix(graph) #, check_infinite_weights='power' in which_CI and not(('CIpower_approx' in which_CI and 'CIpower_approx_tanh' not in which_CI))) #checking only for CIpower models (for which 2*w-1 = tanh(beta*alpha*J_ij) = tanh(beta*alpha*arctanh(2*wgraph_ij-1)) which will create numerical problems if arctanh = inf --> in fact there are also numerical mistakes for w=0 or 1 with CIpower_approx models, but at another point: beliefs become nan or inf
                    if which_CI in list_models_change_weights and alpha.get('K_edges_matrix') is not None:
                        K_edges_matrix = alpha.get('K_edges_matrix')
#                         print("before transformation")
    #                     print(self.w)
                        self.w = transf_w_power(self.w, 1 / K_edges_matrix)
#                         print("min(w_eff) = {}, max(w_eff) = {}".format(np.min(self.w), np.max(self.w)))
#                         print("after transformation with K_edges_matrix = {}".format(K_edges_matrix))
#                         print(self.w)
    #                 self.w = np.zeros((len(graph.nodes), len(graph.nodes))) + 0.5 #default value: 0.5
    #                 node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
    #                 print("alpha = {}".format(alpha))
    #                 if 'power' in which_CI and alpha.get('K_edges_matrix') is not None:
    # #                     K_edges = alpha.get('K_edges')
    #                     K_edges_matrix = alpha.get('K_edges_matrix')
    #                 for node1, node2, d in graph.edges(data=True):
    #                     w_edge = d['weight']
    #                     i_node1 = node_to_inode[node1] #list(graph.nodes).index(node1)
    #                     i_node2 = node_to_inode[node2]
    #                     if 'power' in which_CI and alpha.get('K_edges_matrix') is not None:
    #                         w_edge = transf_w_power(w_edge, 1 / K_edges_matrix[i_node1, i_node2]) #Be careful: by setting self.w this way, self.w are not the true weights. As a consequence, if it's used only to compute the beliefs (by running CI or another algorithm) then it's fine. On the contrary, if it's recovered (Network.w) or used for something else (Network() instantiated and then used for CI,CIpower,...etc), then I should change the code and take the powers into account only when updating the messages
    #                     self.w[i_node1, i_node2] = w_edge
    #                     self.w[i_node2, i_node1] = w_edge
                else:
                    self.w = w.get('alpha_matrix')
                    if which_CI in list_models_change_weights and alpha.get('K_edges_matrix') is not None:
                        K_edges_matrix = alpha.get('K_edges_matrix')
                        self.w = transf_w_power(self.w, 1 / K_edges_matrix)
#                 print("self.w_eff = {}".format(self.w))
                if self.parallel_Mext:
#                     print("before (self.w)", self.w.shape, type(self.w))
                    self.w = self.w[..., np.newaxis] #expand the dimension
#                     print("after (self.w)", self.w.shape, type(self.w))
            else:
                if self.parallel_Mext:
                    raise NotImplemented #TODO: check that it's ok
                #create the matrix of factors
                if w is None:
                    self.factor = np.zeros((2, 2, len(graph.nodes), len(graph.nodes))) + 1 #default value: [[1,1],[1,1]]
                    node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
                    for node1, node2, d in graph.edges(data=True):
                        factor = d['factor']
                        if which_CI in list_models_change_weights and alpha.get('K_edges_matrix') is not None:
                            K_edges_matrix = alpha.get('K_edges_matrix')
                            factor = factor**(1 / K_edges_matrix[i_node1, i_node2]) #Be careful: by setting self.factor this way, self.factor are not the true factors. As a consequence, if it's used only to compute the beliefs (by running CI or another algorithm) then it's fine. On the contrary, if it's recovered (Network.factor) or used for something else (Network() instantiated and then used for CI,CIpower,...etc), then I should change the code and take the powers into account only when updating the messages
                        i_node1 = node_to_inode[node1]
                        i_node2 = node_to_inode[node2]
                        self.factor[i_node1, i_node2] = factor
                        self.factor[i_node2, i_node1] = factor.T
                else:
                    raise NotImplemented
                if self.parallel_Mext:
                    self.factor = self.factor[..., np.newaxis] #expands the dimension

                    
        #Define the alpha (for each each edge). Convention: to compute M_ij, alpha[i,j] is considered (not alpha[j,i]) 
        #i.e. M_ij = F(B_i - alpha_ij M_ji)
        # print("dict_alpha_impaired", dict_alpha_impaired)
        #Set self.temporal_alpha
        if which_CI != 'CIbeliefs':
            self.temporal_alpha = alpha.is_temporal(self.graph)
        else: #which_CI == 'CIbeliefs':
            self.temporal_alpha = False
        assert self.temporal_alpha == False
#         print("self.temporal_alpha", self.temporal_alpha)
#         print("self.alpha", self.alpha)
#         print(self.parallel_CI, self.temporal_alpha)
        #Transform alpha into a dictionnary {alpha[i,j] for all oriented edges (i,j)}
        if self.parallel_CI == False:
            if which_CI != 'CIbeliefs':
                self.alpha = alpha.get_alpha_dict(self.graph)
    #             print("self.alpha for a given node = ")
    #             if type(self.alpha[list(self.alpha)[0]]) == int:
    #                 print(self.alpha)
    #             plt.plot(self.alpha[list(self.alpha)[0]])
    #             plt.show()
            else: #which_CI == 'CIbeliefs':
                self.alpha = alpha.get('dict_alpha_impaired')
        else: #if self.parallel_CI:
            assert which_CI != 'CIbeliefs'
            if self.temporal_alpha:
                print("TODO: find a faster way (inside call to Network)")
                self.alpha = alpha.get_alpha_dict(self.graph)
                #change alpha[i,j][t] into alpha[t][i,j] (in order to catch alpha[t] quickly)
                self.alpha = {t: np.array([self.alpha[key][t] for key in self.alpha.keys()])
                              for t in range(self.alpha[list(self.alpha.keys())[0]])}
            else:
                self.alpha = alpha.to_matrix(graph)
                if self.parallel_Mext:
#                     print("before (self.alpha)", self.alpha.shape, type(self.alpha))
                    self.alpha = self.alpha[..., np.newaxis] #expand the dimension
#                     print("after (self.alpha)", self.alpha.shape, type(self.alpha))
#         print("self.alpha = {}".format(self.alpha))
#         print("self.alpha: from {} to {}".format(np.min(self.alpha), np.max(self.alpha))) #only if self.alpha is a matrix (i.e., sef.parallel_CI = True)
        
        #define K_node (for each node) - only needed for CIpower_approx models (because M_ij = 1/gamma_j * F(B_i - gamma_i / K_ij * M_ji)
        if ('CIpower_approx' in which_CI and 'CIpower_approx_tanh' not in which_CI) or (which_CI == 'rate_network'): #CIpower_approx_tanh = CIpower with F ~ F_w_approx / CIpower_approx (resp. CIpower_approx_approx_tanh) is CIpower_approx model (resp. its approximation)
            assert self.parallel_CI == True
            assert alpha.get('K_nodes') is None
            self.K_nodes = alpha.get('K_nodes_vector', np.array([1.]))
            if self.K_nodes is None: #happens for instance for which_alpha = 'undirected'
                self.K_nodes = np.array([1.])
#             print("self.K_nodes = {}".format(self.K_nodes))
#             print("before (self.K_nodes)", self.K_nodes.shape, type(self.K_nodes))
            self.K_nodes = self.K_nodes[..., np.newaxis]
            if self.parallel_Mext:
                self.K_nodes = self.K_nodes[..., np.newaxis]
#             print("after (self.K_nodes)", self.K_nodes.shape, type(self.K_nodes))
            
        #Initiate the messages
        if which_CI != 'CIbeliefs':
            if self.parallel_CI == False:
                self.M = dict(zip(list(get_all_oriented_edges(graph)), repeat(0)))
            else: #parallel CI
                if self.parallel_Mext == False:
                    self.M = np.zeros((len(graph.nodes), len(graph.nodes)))
                else:
                    self.M = np.zeros((len(graph.nodes), len(graph.nodes), self.n_examples))
        
        #Initiate B_history (history of the beliefs)
        if self.keep_history_beliefs:
            self.B_history = {}
            for node in self.graph.nodes:
                self.B_history[node] = []
        if self.keep_history_beliefs and self.parallel_CI:
            self.B_history = []
            
        #Initiate M_history (history of the messages)
        if self.keep_history_messages:
            self.M_history = {}
            for edge in self.graph.edges:
                self.M_history[edge] = []
        if self.keep_history_messages and self.parallel_CI:
            self.M_history = []
            
        #Initiate the beliefs
#         self.B = {i: 0 for i in self.graph.nodes}
        
        if self.parallel_CI == False:
            self.neighbors = {i: self.get_neighbors(i) for i in self.graph.nodes} #{i: [j for j in self.graph.nodes if (j, i) in self.graph] for i in ....}   #compute it once for all

            
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node)) #ok because graph is undirected; otherwise use nx.all_neighbors(node) #self.graph.neighbors[node]
        
    def get_Mext(self, t):
        if self.constant_Mext:
            return self.M_ext
        else:
            return self.M_ext[t]
        
    def step_message_passing_CI(self, t):
        """
        Non-vectorized message-passing (for vectorized version, see step_message_passing_CI_matrix)
        
        Careful with the convention: M_ij = F_ij(B_i - alpha_ij M_ji)
        i.e. alpha_ij appears in the computation of M_ij, but is in front of the term M_ji
        
        In CI we consider the matrix alpha to be potentially non-symmetric --> without any constraint ("directed") / decomposed as a product of nodal term and edge term ("directed ratio": alpha_ij = Knode_i / Kedge_ij) / symmmetry contraint ("undirected": alpha = 1 / K_edge_ij) / only as a nodal term ("nodal": alpha_ij = Knode_i) 
        """
#         M_old = M.copy()
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs(t-1)
        else:
            sum_M = {node: val[-1] for node, val in self.B_history.items()}
#         print("sum_M = {}".format(sum_M))
        
        #Copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy? --> shallow, but OK: one can modify one without modifying the other - equivalent to self.M.copy()
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_w(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
                    #self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    #self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
                    self.M[i, j] = (1-self.damping) * F_f(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
        else: #default
            if self.with_factors == False:
#                 for (i,j) in self.M:
#                     #self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
#                     self.M[i, j] = (1-self.damping) * F_w(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
                #compute_message = lambda edge: F_w(sum_M[edge[0]] - self.alpha[edge] * M_old[edge[1], edge[0]], self.w[edge])
                compute_message = lambda edge: (1-self.damping) * F_w(sum_M[edge[0]] - self.alpha[edge] * M_old[edge[1], edge[0]], self.w[edge]) + self.damping * M_old[edge[0], edge[1]]
                self.M = dict(zip(self.M.keys(), map(compute_message, self.M.keys())))
            else:
                for (i,j) in self.M:
                    #self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
                    self.M[i, j] = (1-self.damping) * F_f(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
#         print({key: np.array(val).item() for key, val in self.M.items()})
        del M_old
#         print(self.M)
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs(t)
            for node in self.graph.nodes:
                self.B_history[node].append(B_current[node])
        
        if self.keep_history_messages:
            for edge in self.graph.edges:
                self.M_history[edge].append(self.M[edge])
                
    def step_message_passing_CI_matrix(self, t):
        """
        Vectorized message-passing
        Easy vectorization = using matrices(alternatively one could use dataframes) for {M, B, alpha, W}. In this case the update equation is very simple: M_new = F(B-alpha*M^T, W)
        It's probably faster than step_message_passing, but it won't automatically be faster: indeed, we will be doing operations on N_nodes*N_nodes instead of N_edges, i.e. on non-existent edges as well (for which the result will be 0)
        TODO: What would be even better would be to vectorize on all edges (by using Mooij's PhD thesis?) if it's possible... (TODO = think about it)
        
        Careful with the convention: M_ij = F(B_i - alpha_ij M_ji)
        i.e. alpha_ij appears in the computation of M_ij, but is in front of the term M_ji
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
#         print("before (sum_M)", sum_M.shape, type(sum_M))
        sum_M = sum_M[:, np.newaxis] #sum_M.reshape((-1,1))
#         print("after (sum_M)", sum_M.shape, type(sum_M))
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = F_w(sum_M - self.alpha[t] * transpose(self.M), self.w)
            else:
                dM = F_f(sum_M - self.alpha[t] * transpose(self.M), self.factor)
        else: #default
            if self.with_factors == False:
#                 print("sum_M.shape = {}, self.alpha.shape = {}, self.M.shape = {}, transpose(self.M).shape = {}, self.w.shape = {}".format(sum_M.shape, self.alpha.shape, self.M.shape, transpose(self.M).shape, self.w.shape))
                dM = F_w(sum_M - self.alpha * transpose(self.M), self.w) 
            else:
                dM = F_f(sum_M - self.alpha * transpose(self.M), self.factor)
        
#         print("dM.shape = {}".format(dM.shape))
        self.M = (1-self.damping) * dM + self.damping * self.M #damping
#         print({key: np.array(val).item() for key, val in self.M.items()})
#         print(self.M)
#         sys.exit()
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
#             print("B_current = {}".format(B_current))
            self.B_history.append(B_current)
    
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_CIpower_matrix(self, t):
        """
        Extension of Fractional BP
        Like step_message_passing_CI_matrix but with CIpower instead of CI
        CIpower: M_ij = 1/alpha_ji . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        CInew: M_ij = 1/alpha_ji . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij (we still use step_message_passing_CIpower_matrix, with unchanged M_ext and w_ij)
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
#         print("before (sum_M)", sum_M.shape, type(sum_M))
        sum_M = sum_M[:, np.newaxis]
#         print("after (sum_M)", sum_M.shape, type(sum_M))
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha[t]) * F_w(sum_M - self.alpha[t] * transpose(self.M), self.w)
            else:
                dM = 1 / transpose(self.alpha[t]) * F_f(sum_M - self.alpha[t] * transpose(self.M), self.factor)
        else: #default
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha) * F_w(sum_M - self.alpha * transpose(self.M), self.w)
            else:
                dM = 1 / transpose(self.alpha) * F_f(sum_M - self.alpha * transpose(self.M), self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M #damping  #before: (1-self.damping*self.alpha.T) (resp. (1-self.damping*self.alpha[t].T)) and no 1/self.alpha.T in dM ---> but it was a mistake)
#         print({key: np.array(val).item() for key, val in self.M.items()})
#         print(self.M)
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
#             print("B_current = {}".format(B_current))
            self.B_history.append(B_current)
    
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_CIpower_approx_tanh_matrix(self, t):
        """
        Approximation of CIpower (= Fractional BP's extension)
        CIpower_approx_tanh: M_ij = 1/alpha_ji . F_w_approx(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F_w_approx is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower: M_ij = 1/alpha_ji . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha[t]) * F_w_approx_tanh(sum_M - self.alpha[t] * transpose(self.M), self.w)
            else:
                dM = 1 / transpose(self.alpha[t]) * F_f_approx_tanh(sum_M - self.alpha[t] * transpose(self.M), self.factor)
        else: #default
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha) * F_w_approx_tanh(sum_M - self.alpha * transpose(self.M), self.w)
            else:
                dM = 1 / transpose(self.alpha) * F_f_approx_tanh(sum_M - self.alpha * transpose(self.M), self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_full_CIpower_matrix(self, t):
        """
        full_CIpower: M_ij = 1/alpha_ji . F_ij(B_i), F is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower: M_ij = 1/alpha_ji . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        
        There is no K_edges (in fact the model is by definition CIpower with K_edges = + inf)
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha[t]) * F_w(sum_M, self.w)
            else:
                dM = 1 / transpose(self.alpha[t]) * F_f(sum_M, self.factor)
        else: #default
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha) * F_w(sum_M, self.w)
            else:
                dM = 1 / transpose(self.alpha) * F_f(sum_M, self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
    
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_full_CIpower_approx_tanh_matrix(self, t):
        """
        full_CIpower_approx_tanh: M_ij = 1/alpha_ji . F_w_approx(B_i)  where alpha_ij = K_i / K_ij, F_w_approx is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower_approx_tanh: M_ij = 1/alpha_ji . F_w_approx(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F_w_approx is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha[t]) * F_w_approx_tanh(sum_M, self.w)
            else:
                dM = 1 / transpose(self.alpha[t]) * F_f_approx_tanh(sum_M, self.factor)
        else: #default
            if self.with_factors == False:
                dM = 1 / transpose(self.alpha) * F_w_approx_tanh(sum_M, self.w)
            else:
                dM = 1 / transpose(self.alpha) * F_f_approx_tanh(sum_M, self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
    
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_CIpower_approx_matrix(self, t):
        """
        Extension of CI in the case where alpha_ij in CI is a symmetrical matrix
        Approximation of CIpower (= extension of Fractional BP)
        CIpower_approx: M_ij = 1/K_j . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower: M_ij = 1/alpha_ji . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij**1/K_edges, and M_ext is replaced by 1/K_nodes * M_ext
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        assert self.temporal_alpha == False
        if self.with_factors == False:
#             print("F_w(sum_M - self.alpha * transpose(self.M), self.w).shape = {}".format(F_w(sum_M - self.alpha * transpose(self.M), self.w).shape))
#             print("self.K_nodes.shape = {}".format(self.K_nodes.shape))
#             dM = 1 / self.K_nodes * F_w(sum_M - self.alpha * transpose(self.M), self.w) #does indeed 1/K_j (= not 1/K_i) ---> actually wrong!! (it would be the case if self.K_nodes was a 1d-array, but here it is 2d or even 3d because of [:,np.newaxis])
            dM = 1 / transpose(self.K_nodes) * F_w(sum_M - self.alpha * transpose(self.M), self.w)
        else:
#             dM = 1 / self.K_nodes * F_f(sum_M - self.alpha * transpose(self.M), self.factor) #does indeed 1/K_j (= not 1/K_i) ---> actually wrong!!
            dM = 1 / transpose(self.K_nodes) * F_f(sum_M - self.alpha * transpose(self.M), self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M #damping  #before: (1-self.damping*self.alpha.T) (resp. (1-self.damping*self.alpha[t].T)) and no 1/self.alpha.T in dM ---> but it was a mistake)
#         print({key: np.array(val).item() for key, val in self.M.items()})
#         print(self.M)
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
#             print("B_current", B_current)
            self.B_history.append(B_current)
    
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_CIpower_approx_approx_tanh_matrix(self, t):
        """
        Approximation of CIpower_approx (which is the extension of CI) with F = F_w ~ F_w_approx_tanh (without the arctanh)
        
        CIpower_approx_approx_tanh: M_ij = 1/K_j . F_w_approx_tanh(B_i - alpha_ij . M_ji)  where F_w_approx_tanh is F without the arctanh (just the tanh), alpha_ij = K_i / K_ij, F_w_approx_tanh is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower_approx: M_ij = 1/K_j . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        assert self.temporal_alpha == False
        if self.with_factors == False:
            dM = 1 / transpose(self.K_nodes) * F_w_approx_tanh(sum_M - self.alpha * transpose(self.M), self.w)
        else:
            dM = 1 / transpose(self.K_nodes) * F_f_approx_tanh(sum_M - self.alpha * transpose(self.M), self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok

    def step_message_passing_full_CIpower_approx_matrix(self, t):
        """
        full_CIpower_approx: M_ij = 1/K_j . F_ij(B_j)  where F is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower_approx: M_ij = 1/K_j . F_ij(B_i - alpha_ij . M_ji)  where alpha_ij = K_i / K_ij, F is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        
        There is no K_edges (in fact the model is by definition CIpower_approx with K_edges = + inf)
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        assert self.temporal_alpha == False
        if self.with_factors == False:
            dM = 1 / transpose(self.K_nodes) * F_w(sum_M, self.w)
        else:
            dM = 1 / transpose(self.K_nodes) * F_f(sum_M, self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
            
    def step_message_passing_full_CIpower_approx_approx_tanh_matrix(self, t):
        """
        Approximation of full_CIpower_approx (which is the extension of CI) with F = F_w ~ F_w_approx_tanh (without the arctanh)
        
        full_CIpower_approx_approx_tanh: M_ij = 1/K_j . F_w_approx_tanh(B_i)  where F_w_approx_tanh is F without the arctanh (just the tanh), F_w_approx_tanh is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        CIpower_approx_approx_tanh: M_ij = 1/K_j . F_w_approx_tanh(B_i - alpha_ij . M_ji)  where F_w_approx_tanh is F without the arctanh (just the tanh), alpha_ij = K_i / K_ij, F_w_approx_tanh is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        full_CIpower_approx: M_ij = 1/K_j . F_ij(B_j)  where F is applied to f_ij, and M_ext is replaced by 1/K_nodes * M_ext
        
        There is no K_edges (in fact the model is by definition CIpower_approx_approx_tanh with K_edges = + inf)
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis]
        
        assert self.temporal_alpha == False
        if self.with_factors == False:
            dM = 1 / transpose(self.K_nodes) * F_w_approx_tanh(sum_M, self.w)
        else:
            dM = 1 / transpose(self.K_nodes) * F_f_approx_tanh(sum_M, self.factor)
        self.M = (1-self.damping) * dM + self.damping * self.M
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
    
    def step_message_passing_CIbeliefs2(self, t):
        """
        M_ij(t+1) = F_ij(B_i(t) - alpha_ij B_j(t-1))
         
        Question: should we use B_j(t+1) = sum_i F_ij(B_i(t) - alpha_ij B_j(t-1)), or M_ij(t+1) = F_ij(B_i(t) - alpha_ij B_j(t-1))?
        I implemented the 2nd version
        keep_history_beliefs needs to be True (the history of the beliefs is needed in the equation - actually the last belief)
        """
        assert self.damping == 0 #TODO: think of damping != 0 in this model --> one would need to compute M_old, right?
        sum_M = {node: self.B_history[node][-1] for node in self.graph.nodes} #self.compute_beliefs(t-1)
        sum_M_previous = {node: self.B_history[node][-2] for node in self.graph.nodes}
    
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j][t] * sum_M_previous[j], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j][t] * sum_M_previous[j], self.factor[i, j])
        else: #default
            if self.with_factors == False:
#                 for (i,j) in self.M:
#                     self.M[i, j] = F_w(sum_M[i] - self.alpha[i, j] * sum_M_previous[j], self.w[i, j])
                compute_message = lambda edge: F_w(sum_M[edge[0]] - self.alpha[edge] * sum_M_previous[edge[1]], self.w[edge])
                self.M = dict(zip(self.M.keys(), map(compute_message, self.M.keys())))
            else:
                for (i,j) in self.M:
                    self.M[i, j] = F_f(sum_M[i] - self.alpha[i, j] * sum_M_previous[j], self.factor[i, j])
#         print(self.M)
    
        B_current = self.compute_beliefs(t)
        for node in self.graph.nodes:
            self.B_history[node].append(B_current[node])
#         print("t = {}, len(B_history) = {}".format(t, len(self.B_history[list(self.graph.nodes)[0]])))
                    
        if self.keep_history_messages:
            for edge in self.graph.edges:
                self.M_history[edge].append(self.M[edge])
                
    #@jit  
    def step_message_passing_CIbeliefs(self, t):
        """
        Uses the following approximation:
        $$B_j^t = \sum\limits_{i \in N(j)} F_{ij}(B_i^{t-1}) - \sum\limits_{k=2}^{k_{max}} \alpha_{j,k} B_j^{t-k}$$
        (alpha are associated to nodes j, and not to edges as in CI)
    
        keep_history_beliefs needs to be True (the history of the beliefs is needed in the equation)
    
        Here alpha are in the opposite order: alpha_kmax, alpha_kmax-1, ..., alpha_3, alpha_2
    
        TODO: Try to write it in a vectorized way
        """
        
        assert self.damping == 0 #TODO: think of damping != 0 in this model --> B(t+1) = (1-damping)*F(B(t)) + damping*B(t) for instance?
        if self.temporal_alpha: #alpha varies with time
            print("not implemented")
            sys.exit()
        else: #default
            if self.with_factors == False:
                for j in self.B_history:
#                     print("alpha[j]", self.alpha[j])
                    F_wij_t_minus_1 = lambda i: F_w(self.B_history[i][t-1], self.w[i,j])
                    neighbors_contrib = np.sum(list(map(F_wij_t_minus_1 , self.neighbors[j]))) #np.sum([F_w(self.B_history[i][t-1], self.w[i,j]) for i in self.neighbors[j]])
                    self_inhib = np.dot(self.alpha[j][max(len(self.alpha[j])-len(self.B_history[j][:-1]),0):], 
                                        np.array(self.B_history[j])[-len(self.alpha[j]) - 1: -1]) #self.alpha[j][-len(self.B_history[j][:-1]):] (without the "len(self.alpha[j])") does not work: for instance [1,2,3][-0:] = [1,2,3] and not [] as we would like   #added the np.array(list) in order to use jax.numpy (the input must be of type array in function dot)
#                     print("neighbors_contrib", neighbors_contrib)
#                     print("self_inhib", self_inhib)
#                     print("self.M_ext[j][t-1]", self.M_ext[j][t-1])
                    self.B_history[j].append(neighbors_contrib - self_inhib + self.M_ext[j][t-1]) #self.alpha[j] = ... , alpha_{j,3}, alpha_{j,2}.  #self.alpha[j][-len(self.B_history[j]):] because at the beginning of the simulation the list self.alpha[j] has a bigger size than self.B_history[j]
            else:
                for j in self.B_history:
                    self.B_history[j].append(
                        np.sum([F_f(self.B_history[i][t-1], self.factor[i,j]) for i in self.neighbors[j]]) - 
                        np.dot(self.alpha[j][-len(self.B_history[j][:-1]):], 
                               np.array(self.B_history[j])[-len(self.alpha[j]) - 1: -1]) +
                        self.M_ext[j][t-1]
                    ) #added the np.array(list) in order to use jax.numpy (the input must be of type array in function dot)
                
                
    def step_message_passing_CIapprox(self, t):
        """
        M_ij = F_w_approx(B_i - alpha_ij.M_ji, w_ij) where F_w_approx is an "arbitrary" function trying to fit F as a+b.p where p = sig(x) is the probability
        """
        sum_M = self.compute_beliefs(t-1)
        
        #copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy?
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    #self.M[i, j] = F_w_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
                    self.M[i, j] = (1-self.damping) * F_w_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
            else:
                for (i,j) in self.M:
                    #self.M[i, j] = F_f_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
                    self.M[i, j] = (1-self.damping) * F_f_approx(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
        else: #default
            if self.with_factors == False:
                for (i,j) in self.M:
                    #self.M[i, j] = F_w_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
                    self.M[i, j] = (1-self.damping) * F_w_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
            else:
                for (i,j) in self.M:
                    #self.M[i, j] = F_f_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
                    self.M[i, j] = (1-self.damping) * F_f_approx(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
        del M_old
        
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs(t)
            for node in self.graph.nodes:
                self.B_history[node].append(B_current[node])
                
        if self.keep_history_messages:
            for edge in self.graph.edges:
                self.M_history[edge].append(self.M[edge])
    
    def step_message_passing_CIapprox_tanh_matrix(self, t):
        """
        M_ij = F_w_approx_tanh(B_i - alpha_ij.M_ji, w_ij) where F_w_approx_tanh is F without the arctanh (just the tanh)
        
        "linearized" CI (using the linearization artanh(x)~x --> same equation as Srdjan's work, with a tanh)
        To be more precise, we use artanh(J*tanh(x))~J*tanh(x)
        So the "linearized" CI is not fully fully linearized but has a tanh
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis] #sum_M.reshape((-1,1))
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = F_w_approx_tanh(sum_M - self.alpha[t] * transpose(self.M), self.w)
            else:
                dM = F_f_approx_tanh(sum_M - self.alpha[t] * transpose(self.M), self.factor)
        else: #default
            if self.with_factors == False:
                dM = F_w_approx_tanh(sum_M - self.alpha * transpose(self.M), self.w) 
            else:
                dM = F_f_approx_tanh(sum_M - self.alpha * transpose(self.M), self.factor)
        
        self.M = (1-self.damping) * dM + self.damping * self.M #damping
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
                
    def step_message_passing_CIapprox_tanh(self, t):
        """
        M_ij = F_w_approx_tanh(B_i - alpha_ij.M_ji, w_ij) where F_w_approx_tanh is F without the arctanh (just the tanh)
        
        "linearized" CI (using the linearization artanh(x)~x --> same equation as Srdjan's work, with a tanh)
        To be more precise, we use artanh(J*tanh(x))~J*tanh(x)
        So the "linearized" CI is not fully fully linearized but has a tanh
        """
        sum_M = self.compute_beliefs(t-1)
        
        #copy self.M
        M_old = copy.copy(self.M) #shallow/deep copy?

        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_w_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_w_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_f_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_f_approx_tanh(sum_M[i] - self.alpha[i, j][t] * M_old[j, i], self.factor[i, j])
        else: #default
            if self.with_factors == False:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_w_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_w_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.w[i, j])
            else:
                for (i,j) in self.M:
                    self.M[i, j] = (1-self.damping) * F_f_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j]) + self.damping * M_old[i, j]
#                     self.M[i, j] = F_f_approx_tanh(sum_M[i] - self.alpha[i, j] * M_old[j, i], self.factor[i, j])
        del M_old
        
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs(t)
            for node in self.graph.nodes:
                self.B_history[node].append(B_current[node])
#             print(self.B_history)
#             print({key:val[-1] for key,val in self.B_history.items()})

        if self.keep_history_messages:
            for edge in self.graph.edges:
                self.M_history[edge].append(self.M[edge])

    def step_message_passing_CIapprox_linear_matrix(self, t):
        """
        M_ij = (2*w_ij(1) * (B_i - alpha_ij.M_ji)
        
        linearized CI
        """
        if self.keep_history_beliefs == False:
            sum_M = self.compute_beliefs_matrix(t-1)
        else:
            sum_M = self.B_history[-1]
        sum_M = sum_M[:, np.newaxis] #sum_M.reshape((-1,1))
        
        if self.temporal_alpha: #alpha varies with time
            if self.with_factors == False:
                dM = F_w_approx_linear(sum_M - self.alpha[t] * transpose(self.M), self.w)
            else:
                raise NotImplemented
        else: #default
            if self.with_factors == False:
                dM = F_w_approx_linear(sum_M - self.alpha * transpose(self.M), self.w) 
            else:
                raise NotImplemented
        
        self.M = (1-self.damping) * dM + self.damping * self.M #damping
    
        if self.keep_history_beliefs:
            B_current = self.compute_beliefs_matrix(t)
            self.B_history.append(B_current)
            
        if self.keep_history_messages:
            self.M_history.append(copy.copy(self.M)) #shallow copy but ok
            
    def compute_beliefs(self, t):
#         print(self.M_ext)
        B = {i: self.M_ext[i][t] + np.sum([self.M[j, i] for j in self.graph.neighbors(i)]) #self.graph.neighbors[i]])
             for i in self.graph.nodes
            } #should it be M_ext[i][t] or M_ext[i][t-1]??
        return B
    
    def compute_beliefs_matrix(self, t):
#         print('M_ext', self.M_ext)
#         print('M', self.M)
#         B = (self.M_ext[t] + np.sum(self.M, axis=0)).reshape((-1,1))
#         #should it be M_ext[i][t] or M_ext[i][t-1]??
        if t == -1:
            t = self.T - 1 #temporary solution
#         print("M_ext.shape = {}, np.sum(M, axis=0) = {}".format(self.get_Mext(t).shape, np.sum(self.M, axis=0).shape))
        return self.get_Mext(t) + np.sum(self.M, axis=0) #self.M_ext[t] + np.sum(self.M, axis=0) #return B
    
    #@jit
    def run_CI(self, verbose=False, which_CI='CI'):
        assert not('which_CI' in ['CIbeliefs', 'CIbeliefs2']
                   and self.keep_history_beliefs == False) #for CIbeliefs and CIbeliefs2, we need keep_history_beliefs = True
        
        if self.parallel_CI == False:
            assert which_CI not in ['CIpower', 'CIpower_approx', 'CInew'] #unparallel CIpower is not implemented - same for CInew
            
            #initialize B_history
            if self.keep_history_beliefs:
                for node in self.graph.nodes:
                    self.B_history[node].append(0)
                if which_CI != 'CIbeliefs':
                    B_current = self.compute_beliefs(0) #0th iteration (just taking into account the external messages)
                    for node in self.graph.nodes:
                        self.B_history[node].append(B_current[node]) #this line is not mandatory (starting at 0 is enough) #--> remove?
            #initialize M_history
            if self.keep_history_messages:
                for edge in self.graph.edges:
                    self.M_history[edge].append(0)
            
            dict_fun_message_passing = {
                'CI': self.step_message_passing_CI,
                'CIbeliefs2': self.step_message_passing_CIbeliefs2,
                'CIbeliefs': self.step_message_passing_CIbeliefs,
                'CIapprox': self.step_message_passing_CIapprox,
                'CIapprox_tanh': self.step_message_passing_CIapprox_tanh
            } #gives all the possible non-parallel message passing implementations
            fun_message_passing = dict_fun_message_passing[which_CI]
            if which_CI == 'CI':
                print('Using the unparallel version of CI (there is a parallel equivalent)')
            
            t_list = range(self.T) if which_CI != 'CIbeliefs' else range(1, self.T+1) #starts at 1 for CIbeliefs!!!
            for t in t_list:
                fun_message_passing(t)
                
            #final belief
            if self.keep_history_beliefs == False: #which_CI in ['CI', 'CIapprox', 'CIapprox_tanh']:
                B = self.compute_beliefs(self.T - 1)
            else: #elif which_CI == ['CIbeliefs', 'CIbeliefs2']:
                B = {node: B_history_node[-1] for node, B_history_node in self.B_history.items()} #self.compute_beliefs(self.T - 1) 
        
        else: #parallel CI
            dict_fun_message_passing_matrix = {
                'CI': self.step_message_passing_CI_matrix,
                'CIpower': self.step_message_passing_CIpower_matrix,
                'CIpower_approx_tanh': self.step_message_passing_CIpower_approx_tanh_matrix,
                'CIpower_approx': self.step_message_passing_CIpower_approx_matrix,
                'CIpower_approx_approx_tanh': self.step_message_passing_CIpower_approx_approx_tanh_matrix,
                'full_CIpower': self.step_message_passing_full_CIpower_matrix,
                'full_CIpower_approx_tanh': self.step_message_passing_full_CIpower_approx_tanh_matrix,
                'full_CIpower_approx': self.step_message_passing_full_CIpower_approx_matrix,
                'full_CIpower_approx_approx_tanh': self.step_message_passing_full_CIpower_approx_approx_tanh_matrix,
                'CInew': self.step_message_passing_CIpower_matrix, #same message-passing equation (but M_ext and w_ij are unchanged)
                'CIapprox_tanh': self.step_message_passing_CIapprox_tanh_matrix,
                'CIapprox_linear': self.step_message_passing_CIapprox_linear_matrix,
                'rate_network': self.step_message_passing_full_CIpower_approx_approx_tanh_matrix
            } #gives all the possible parallel message passing implementations
            assert which_CI in dict_fun_message_passing_matrix.keys()
            if self.keep_history_beliefs:
                if self.parallel_Mext == True:
                    self.B_history.append(np.zeros((len(self.graph.nodes), self.M_ext.shape[1])))
                else:
                    self.B_history.append(np.zeros(len(self.graph.nodes)))
                B_current = self.compute_beliefs_matrix(0) #0th iteration (just taking into account the external messages
                self.B_history.append(B_current) #this line is not mandatory (starting at 0 is enough) #--> remove?
#             print("self.B_history", self.B_history)
            if self.keep_history_messages:
                if self.parallel_Mext == True:
                    self.M_history.append(np.zeros((len(self.graph.nodes), len(self.graph.nodes), self.M_ext.shape[1])))
                else:
                    self.M_history.append(np.zeros((len(self.graph.nodes), len(self.graph.nodes))))
#             print("self.M_history", self.M_history)
            
            fun_message_passing_matrix = dict_fun_message_passing_matrix[which_CI]
            for t in range(self.T):
                fun_message_passing_matrix(t)
            B = self.compute_beliefs_matrix(self.T - 1) #final belief
        
        if verbose:
            print("self.M")
            print(self.M)
            print("self.B")
            print(self.B)
        
        return B
    
def run_algo(graph, M_ext, 
             which_CI='CI',
             alpha=None, w=None, w_input=None,
             damping=0, verbose=False, keep_history_beliefs=False, keep_history_messages=False, return_final_M=False, 
             parallel_CI=True, parallel_Mext=True,
             transform_into_dict=False, niter=100):
#     print("w = {} (in run_algo function in utils_CI_BP.py)".format(w))
    alpha.check()
    assert not(parallel_Mext == True and parallel_CI == False)
    assert not('BP' in which_CI and alpha != 1)
#     assert not((return_final_M == True) and (keep_history_beliefs == True)) #why such a constraint?
    assert not(which_CI in ['CIbeliefs', 'CIbeliefs2'] and keep_history_beliefs == False) #beliefs' history is needed
    assert not(which_CI == 'CIbeliefs' and damping != 0) #what would damping be for CIbeliefs?
    assert not(which_CI == 'CIbeliefs' and dict_alpha_impaired is None) #dict_alpha_impaired where keys = all nodes
    net = Network(graph, M_ext, 
                  alpha=alpha, w=w, w_input=w_input,
                  damping=damping, keep_history_beliefs=keep_history_beliefs, keep_history_messages=keep_history_messages,
                  which_CI=which_CI, 
                  parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, niter=niter)
    B_final = net.run_CI(verbose=verbose, which_CI=which_CI)
#     if which_CI == 'CIpower':
#         print("parallel_CI = {}".format(parallel_CI))
#         print("transform_into_dict = {}".format(transform_into_dict))
        
    if parallel_CI == False:
        if keep_history_beliefs:
            net.B_history = {key:np.array(val) for key,val in net.B_history.items()} #need to transform the lists into numpy arrays
        if keep_history_messages:
            net.M_history = {key:np.array(val) for key,val in net.M_history.items()} #need to transform the lists into numpy arrays
    
    else: #parallel_CI == True
        if transform_into_dict:
            B_final = dict(zip(list(graph.nodes), B_final))
            if keep_history_beliefs:
#                 print("np.array(net.B_history).shape", np.array(net.B_history).shape)
                net.B_history = dict(zip(list(graph.nodes), list(transpose(np.array(net.B_history))))) #not np.array(net.B_history).T (because wrong if parallel_Mext=True, i.e., if np.array(net.B_history) has dimension (T, n_nodes, n_examples) and not simply (T, n_nodes))
            if keep_history_messages:
                print("check things here")
                print("np.array(net.M_history)).shape = {}".format(np.array(net.M_history)).shape)
                sys.exit()
                net.M_history = dict(zip(list(graph.nodes), list(transpose(np.array(net.M_history)))))
            if return_final_M:
                M_copy = net.M.copy()
                #transform mat into dict
                net.M = {}
                node_to_inode = dict(zip(list(graph.nodes), range(len(graph.nodes))))
                for (node1, node2) in graph.edges:
                    i_node1 = node_to_inode[node1]
                    i_node2 = node_to_inode[node2]
                    net.M[node1, node2] = M_copy[i_node1, i_node2]
                    net.M[node2, node1] = M_copy[i_node2, i_node1]
                del M_copy
#         if keep_history_beliefs:
#             net.B_history = np.array([net.B_history[t][i_node] for i_node in range(
    return B_final, net

    
def run_CI(*args, **kwargs):
    return run_algo(*args, which_CI='CI', **kwargs)
    
def run_BP(*args, **kwargs):
    if 'alpha' in kwargs.keys():
        assert kwargs['alpha'] == Alpha_obj({'alpha': 1}), "Alpha_obj should be 1 - it is = {}".format(kwargs['alpha'])
        return run_CI(*args, **kwargs)
    return run_CI(*args, alpha=Alpha_obj({'alpha': 1}), **kwargs)

def run_CIpower(*args, **kwargs):
    return run_algo(*args, which_CI='CIpower', **kwargs)

def run_CIpower_approx_tanh(*args, **kwargs):
    return run_algo(*args, which_CI='CIpower_approx_tanh', **kwargs)

def run_CIpower_approx(*args, **kwargs):
    return run_algo(*args, which_CI='CIpower_approx', **kwargs)

def run_CIpower_approx_approx_tanh(*args, **kwargs):
    return run_algo(*args, which_CI='CIpower_approx_approx_tanh', **kwargs)

def run_full_CIpower(*args, **kwargs):
    return run_algo(*args, which_CI='full_CIpower', **kwargs)

def run_full_CIpower_approx_tanh(*args, **kwargs):
    return run_algo(*args, which_CI='full_CIpower_approx_tanh', **kwargs)

def run_full_CIpower_approx(*args, **kwargs):
    return run_algo(*args, which_CI='full_CIpower_approx', **kwargs)

def run_full_CIpower_approx_approx_tanh(*args, **kwargs):
    return run_algo(*args, which_CI='full_CIpower_approx_approx_tanh', **kwargs)

def run_rate_network(*args, **kwargs):
    return run_algo(*args, which_CI='rate_network', **kwargs)
    
def run_CInew(*args, **kwargs):
    return run_algo(*args, which_CI='CInew', **kwargs)
    
def run_CIbeliefs2(*args, **kwargs):
    return run_algo(*args, which_CI='CIbeliefs2', keep_history_beliefs=True, **kwargs)

def run_CIbeliefs(*args, **kwargs):
    """
    Indicate dict_alpha_impaired (where keys = all nodes)
    """
    return run_algo(*args, which_CI='CIbeliefs', damping=0, keep_history_beliefs=True, **kwargs)
    
def run_CI_approx(*args, **kwargs):
    return run_algo(*args, which_CI='CIapprox', **kwargs)
    
def run_BP_approx(*args, **kwargs):
    return run_CI_approx(*args, alpha=Alpha_obj({'alpha': 1}), **kwargs)
    
def run_CI_approx_tanh(*args, **kwargs):
    return run_algo(*args, which_CI='CIapprox_tanh', **kwargs)
    
def run_BP_approx_tanh(*args, **kwargs):
    return run_CI_approx_tanh(*args, alpha=Alpha_obj({'alpha': 1}), **kwargs)
    
def run_CI_approx_linear(*args, **kwargs):
    return run_algo(*args, which_CI='CIapprox_linear', **kwargs)
    
def run_BP_approx_linear(*args, **kwargs):
    return run_CI_approx_tanh(*args, alpha=Alpha_obj({'alpha': 1}), **kwargs)
    
    
def sum_squared_updates(updates, begin=0):
    return {key: np.sum(val[begin:]**2) for key, val in updates.items()}

def sum_abs_updates(updates, begin=0):
    return {key: np.sum(np.abs(val[begin:])) for key, val in updates.items()}

def get_activations(B_history, method='square_updates_B', k=None):
    assert method in ['square_updates_B', 'leaky_belief']
    if method == 'square_updates_B':
#         updates = {key: B[1:] - B[:-1] for key, B in B_history.items()}
#         squared_updates = {key:val**2 for key,val in updates.items()}        
#         sum_squared_updates = sum_squared_updates(updates, begin=begin)
#         sum_squared_updates = np.array([sum_squared_updates[node] for node in G.nodes])
#         sum_abs_updates = sum_abs_updates(updates, begin=begin)
#         sum_abs_updates = np.array([sum_abs_updates[node] for node in G.nodes])
#         activations_history = {key: val**2 for key, val in updates.items()}
        activations_history = {key: (B[1:] - B[:-1])**2 for key, B in B_history.items()}
    elif method == 'leaky_belief':
        assert k != None
        if k == np.inf: #k=+inf means in fact that activity=|B| (instead of |dB/dt + k*B|) --> I use that to look at overconfidence
            activations_history = {key: np.abs(B)
                                   for key, B in B_history.items()
                                  }
        else: #default
            activations_history = {key: np.abs(B[1:] - B[:-1] + k * B[:-1])
                                   for key, B in B_history.items()
                                  }  #abs(updates + k*B)
    return activations_history


def get_total_activation(activations_history, begin=0):
    return {key: np.sum(val[begin:]) for key, val in activations_history.items()}


def detect_bistability_dict(B_history, verbose=False):
    """
    Returns True is one of the nodes has some bistable behavior
    """
    dict_is_bistable = {key: detect_bistability(val) for key,val in B_history.items()}
    list_bistable = list(dict_is_bistable.values())
    exists_bistable_node = list_bistable.count(True) >= 1
    if verbose and exists_bistable_node:
        for node, val in B_history.items():
            if dict_is_bistable[node] == True:
                plt.plot(val)
                break
        plt.title('Example of node for which there is bistability')
        plt.show()
    return exists_bistable_node


def detect_bistability(B_history):
    """
    B_history is an array, not a dict: only one node
    TODO: possibly add a criterion: that the value of B at the end of each period is less than 1 in absolute value
    """
    hist, bin_edges = np.histogram(B_history, bins=np.linspace(-6,6,40))
    # plt.plot(hist)
    # plt.show()
    # print(bin_edges)
    mean_bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
#     plt.plot(mean_bin_edges, hist)
#     plt.show()
    d = dict(zip(mean_bin_edges, hist))
    # print(d)
    key_highest_val = list(d.keys())[np.argmax(list(d.values()))]
    # print(key_highest_val)
    if np.abs(key_highest_val) > 1:
#         print("bistability")
        return True
    else:
        return False

    
def detect_frustration_dict(B_history, begin=0):
    """
    TODO: update this function (see low frustration detection in function load_simulated_data from analyze_effects_CI_vs_BP.py)
    """
    squared_updates_CI = {node:((val[1:] - val[:-1])**2)[begin:] for node, val in B_history.items()}
    res = np.max(np.array(list(squared_updates_CI.values()))) > 3 #> 1
    return res


def test_convergence(arr):
    """
    Very simple convergence test: that the last update is smaller than 0.1
    """
    return np.abs(arr[-1] - arr[-2]) < 0.1


def test_convergence_dict(B_history):
    for val in B_history.values():
        if test_convergence(val) == False:
            return False
    return True