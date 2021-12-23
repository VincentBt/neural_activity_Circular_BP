import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from graph_generator import *
import scipy
from scipy import spatial
from multiprocessing import cpu_count, Pool
n_cpus = 4 #cpu_count() #4 gives the fastest results (even though there are cpu_count()=8 cpus on my machine...)
from itertools import repeat
import pandas as pd


def generate_M_ext(type_M_ext, graph, n_stimulated_nodes=8, stimulated_nodes=None, 
                   values_constant_input=None,
                   T=None, n_periods=4, type_graph=None, T_period=500,
                   variance_Mext=None, mean_Mext=0,
                   add_trigger_hallu=False,
                   simple_Mext=False, seed=None
                  ):
    """
    n_stimulated_nodes is the number of stimulated nodes at each period
    stimulated_nodes is the list of stimulated nodes (to give only if there is only 1 period)
    
    TODO: include the possibility of having only one external signal, projected into nodes with (potentially) different input weights (which is what people usually do, instead of generating uncorrelated external input for each neuron)
    
    values_constant_input is only useful in the case of constant stimulation (standard BP/CI)
    
    Note that the output only has keys corresponding to stimulated nodes only (if for instance n_stimulated_nodes=1)
    
    For constant input (e.g., to run BP or exact inference):
    values_constant_input = np.random.normal(scale=std_Mext, size=len(list(graph.nodes)))
    M_ext = generate_M_ext('constant', graph, stimulated_nodes='all', values_constant_input=values_constant_input)

    """
    if seed is None:
        np.random.seed() #to have different results
    else:
        np.random.seed(seed)
        
    #M_ext = {0: 0.5, 14: 0.7, 23:-0.9, 8:-0.5}
    # M_ext = {'L': 0.5, 'S': 0.7, 'T':-0.9, 'J':-0.5} #given nodes and values
    # M_ext = dict(zip(np.random.choice(list(G.nodes()), size=4, replace=False),
    #                  [0.5, 0.7, -0.9, -0.5]
    #                 )) #random
    #Poisson process: generation of the external messages
    #T = 1000
    #lam = 0.05 #lambda of the Poisson process
    #nodes_Mext = np.random.choice(list(G.nodes()), size=10, replace=False)
    #M_ext = {key:5*np.random.poisson(lam=lam, size=T) for key in nodes_Mext}
    #M_ext = {key:np.random.choice([1,-1],size=T)*np.random.poisson(lam=lam, size=T) for key in nodes_Mext}

#     n_processes = 10
#     variance_Mext = 10
    #generate_gaussian_process(T, n_processes, variance_Mext)
    #M_ext = dict(zip(np.random.choice(list(G.nodes()), size=n_processes, replace=False),generate_gaussian_process(T, n_processes, variance)))
    
    # #Gaussian process
    # T = 1500
    # n_processes = 15
    # variance_Mext = 10
    # period=150
    # M_ext = dict(zip(np.random.choice(list(G.nodes()), size=n_processes, replace=False),generate_gaussian_process(T, n_processes, variance_Mext)))

    # #Pulse on one node
    # #T = 100
    # #M_ext = dict(zip(np.random.choice(list(G.nodes()), size=1, replace=False),
    # #                  [[1] + [0]*(T-1)]
    # #                  [[5] + [0]*int((T/2-1)) + [0.1] + [0]*int((T/2-1))]


#     T = 1000
#     n_processes = 10
#     variance_Mext = 10
    #generate_gaussian_process(T, n_processes, variance_Mext)
    #M_ext = dict(zip(np.random.choice(list(G.nodes()), size=n_processes, replace=False),generate_gaussian_process(T, n_processes, variance_Mext)))
    
#     print(simple_Mext, type_graph, stimulated_nodes)
#     print("type_M_ext =", type_M_ext)
#     print(stimulated_nodes)

    #run some checks
    assert type_M_ext in ['constant', 'pulse', 'poisson_process', 'white_noise', 'gaussian_process', 
                          'gaussian_process_indep', 'gaussian_process_by_periods', 'gaussian_process_by_periods_not_all_nodes']
    assert not((simple_Mext or stimulated_nodes in ["auditory","visual","language","parahipp_L"]) and ('realistic_connectome' not in type_graph))
#     if stimulated_nodes is not None:
#         assert len(stimulated_nodes) == n_stimulated_nodes #actually removed this test (stimulated_nodes can be "auditory","visual","all", ...etc
#     assert not((stimulated_nodes is None) and (n_stimulated_nodes is None)) #removed this test (gaussian_process_by_periods determines stimulated_nodes and n_stimulated_nodes)
    assert not((values_constant_input is not None) and (type_M_ext != 'constant')) 
    
    if isinstance(graph, nx.classes.graph.Graph): #default
        list_nodes_graph = list(graph.nodes())
    else:
        list_nodes_graph = graph
        
    if (simple_Mext or stimulated_nodes in ["auditory","visual","language","parahipp_L"]) and ('realistic_connectome' in type_graph): #stimulate only the primary cortices
        # print("Stimulating only the primary cortices")
        if '_not_all_nodes' not in type_M_ext:
            type_M_ext = type_M_ext + '_not_all_nodes'
        list_i_node, list_names = read_AAL2_file(type_graph.replace('realistic_connectome_', ''))
        mapping_nodes = dict(zip(list_i_node, list_names))
        primary_cortices = {"auditory": [83,84] if 'AAL2' in type_graph else [29,30], #primary auditory cortex (83,84) #but 84 is unconnected to the rest of the graph for AAL2) --> no, in fact it's connected...
                            "visual": [49,50] if 'AAL2' in type_graph else [19,20], #primary visual cortex (49,50)
                            "language": [7,8,9,10,11,12,65,66,67,68,85,86] if 'AAL2' in type_graph else sys.exit(), #7-12 Broca, 65-68 aire de wernicke, 85-86 ?
                            "parahipp_L": [41,43] if 'AAL2' in type_graph else sys.exit()
                           }
#         associative_cortices = {"auditory": [85,86,87,88,89,90] if 'AAL2' in type_graph else [71,72],
#                                 "visual": [51,52,53,54,55,56,57,58,59,60] if 'AAL2' in type_graph else [35,36,37,38,27,28] #not sure for the associative_cortices of the visual in AAL-merged: check with Renaud
#                                }
        #Renaud: try also associative cortices like the temporo-parietal junction = 65-68 + 85-86
        assert stimulated_nodes in primary_cortices.keys()
#         if stimulated_nodes in ["auditory","visual"]:
        stimulated_nodes = [mapping_nodes[el] for el in primary_cortices[stimulated_nodes]]
#         else:
#             stimulated_nodes = [mapping_nodes[el] for el in primary_cortices["visual"]] #primary auditory cortices
#         print("stimulated_nodes", stimulated_nodes)
        n_stimulated_nodes = len(stimulated_nodes)
        return generate_M_ext(type_M_ext, graph, n_stimulated_nodes=n_stimulated_nodes, stimulated_nodes=stimulated_nodes, 
                              T=T, n_periods=n_periods, type_graph=type_graph, T_period=T_period,
                              variance_Mext=variance_Mext, mean_Mext=mean_Mext,
                              add_trigger_hallu=add_trigger_hallu,
                              simple_Mext=simple_Mext, seed=seed)
    
    if type_M_ext == 'gaussian_process_by_periods': #in this case n_stimulated_nodes != len(stimulated_nodes)
        stimulated_nodes = list_nodes_graph #stimulated_nodes doesn't matter anyway (not used)
        n_stimulated_nodes = int(len(graph)/2) if len(graph)%2==0 else [int(len(graph)/2)+1, int(len(graph)/2)] #16 #for each period (they are not necessarily the same across periods) #If G has an uneven number of nodes, then we don't stimulate the same number of nodes in each period, but each node is stimulated the same number of times in the whole simulation
    elif stimulated_nodes is None:
        stimulated_nodes = list(np.random.choice(list_nodes_graph, size=n_stimulated_nodes, replace=False))
    elif stimulated_nodes == 'all':
        stimulated_nodes = list_nodes_graph
        n_stimulated_nodes = len(stimulated_nodes)
    else:
        stimulated_nodes = list(stimulated_nodes) #just transform into list (e.g. stimulated_nodes is a single node)
        n_stimulated_nodes = len(stimulated_nodes)

    
    if type_M_ext == 'constant':
#         M_ext = {0: 0.5, 14: 0.7, 23:-0.9, 8:-0.5}
#         M_ext = dict(zip(np.random.choice(list_nodes_graph, size=1, replace=False),
#                          [+1]
#                         ))
        #n_stimulated_nodes = 4
        if values_constant_input is None:
            print("Generating random M_ext in [-1,1]")
            values_constant_input = np.random.choice([1,-1], size=n_stimulated_nodes)
        M_ext = dict(zip(stimulated_nodes,
                         values_constant_input
#                          [0.5, 0.7, -0.9, -0.5]
                        )) #random choice of nodes

    elif type_M_ext == 'poisson_process':
        if T == None:
            T = 1000
        lam = 0.05 #lambda of the Poisson process
        #n_stimulated_nodes = 10
        # M_ext = {key:5*np.random.poisson(lam=lam, size=T) for key in stimulated_nodes}
        M_ext = {key : np.random.choice([1,-1],size=T)*np.random.poisson(lam=lam, size=T) 
                 for key in stimulated_nodes}
    
    elif type_M_ext == 'pulse': #Pulse on one node
        if T == None:
            T = 100
        #n_stimulated_nodes = 1
        M_ext = dict(zip(stimulated_nodes,
                         np.concatenate((np.random.choice([-1,1],n_stimulated_nodes).reshape((-1,1)), np.zeros((n_stimulated_nodes,T-1))), axis=1)
#                          [[-1] + [0]*(T-1)]
        #                  [[5] + [0]*int((T/2-1)) + [0.1] + [0]*int((T/2-1))]
                        ))
#         M_ext = dict(zip(np.random.choice(list_nodes_graph, size=1, replace=False),
#                          [[-1] + [0]*(T-1)]
#         #                  [[5] + [0]*int((T/2-1)) + [0.1] + [0]*int((T/2-1))]
#                         ))
        
    elif type_M_ext == 'white_noise':
        if T == None:
            T = 1000
        #n_stimulated_nodes = 4
        if variance_Mext is None:
            variance_Mext = 1
        M_ext = dict(zip(stimulated_nodes,
                         generate_white_noise(T, n_stimulated_nodes, variance_Mext)
                        ))
        
    elif type_M_ext == 'gaussian_process':
        if T == None:
            T = 1000
        #n_stimulated_nodes = 4
        if variance_Mext is None:
            variance_Mext = 50
        assert isinstance(variance_Mext, float) or isinstance(variance_Mext, int)
        M_ext = dict(zip(stimulated_nodes,
                         generate_gaussian_process(T, n_stimulated_nodes, variance_Mext)
                        ))

#     elif type_M_ext == 'gaussian_process_indep': #actually I'm not sure of the difference with gaussian_process (is it really created independent variables?)
# #         print("entering gaussian_process_indep")
# #         print(n_stimulated_nodes, stimulated_nodes)
#         if T == None:
#             T = 1000
#         #n_stimulated_nodes = 4
#         if variance_Mext == None:
#             variance_Mext = 50
# #         list_gaussian_processes = generate_gaussian_process(T, n_stimulated_nodes, variance_Mext) #not independant processes (actually I'm not sure of that...)
#         list_gaussian_processes = np.array([generate_gaussian_process(T, 1, variance_Mext) 
#                                             for _ in range(n_stimulated_nodes)])[:,0,:] #independant processes (actually I'm not sure of that...)
# #         print(list_gaussian_processes.shape)
#         M_ext = dict(zip(stimulated_nodes,
#                          list_gaussian_processes
#                         ))
        
    elif type_M_ext == 'gaussian_process_by_periods': #here we force all nodes to be stimulated the same number of times
        if n_stimulated_nodes == None:
            n_stimulated_nodes = 16 #for each period (they are not necessarily the same across periods)
        if isinstance(n_stimulated_nodes, int):
            assert (n_stimulated_nodes*n_periods) % len(list_nodes_graph) == 0 #because we want each node to be stimulated the same number of times (as for instance during stimulation, there is not much overactivation)
        # n_periods = 4
        # T_period = 500 #for example
        T = n_periods * T_period
        stimulated_nodes_all_periods = create_stimulated_nodes_all_periods(list_nodes_graph, n_periods, n_stimulated_nodes, type_graph=type_graph) #creates a list of stimulated nodes for each period (in order for all nodes to be stimulated the same number of periods)
#         print("stimulations:", [len(el) for el in stimulated_nodes_all_periods])
        if not isinstance(n_stimulated_nodes, int):
            n_periods_per_motif = len(n_stimulated_nodes) #where n_stimulated_nodes is the list containing the number of stimulated nodes for each motif
            n_motives = n_periods // n_periods_per_motif
#             intervals = [0] + list(np.cumsum(n_stimulated_nodes))
            for motif in range(n_motives):
#                 print("motif {}, periods {} to {}".format(motif, motif*n_periods_per_motif, (motif+1)*n_periods_per_motif))
                stimulated_nodes_motif = stimulated_nodes_all_periods[motif*n_periods_per_motif: (motif+1)*n_periods_per_motif]
                assert len(stimulated_nodes_motif) == n_periods_per_motif
                stimulated_nodes_motif_flattened = list(itertools.chain.from_iterable(stimulated_nodes_motif))
#                 print(len(stimulated_nodes_motif_flattened), len(np.unique(stimulated_nodes_motif_flattened)))
        M_ext_array = np.zeros((len(list_nodes_graph), T))
        for period in range(n_periods):
#             M_ext_period = generate_M_ext('gaussian_process_indep', graph, n_stimulated_nodes=n_stimulated_nodes, T=T_period) #"recurrent" call
#             M_ext_period = generate_M_ext('gaussian_process_indep', graph, n_stimulated_nodes=n_stimulated_nodes, stimulated_nodes=stimulated_nodes_all_periods[period], T=T_period) #"recurrent" call
#             M_ext_period = generate_M_ext('gaussian_process_indep', graph, n_stimulated_nodes=len(stimulated_nodes_all_periods[period]), stimulated_nodes=list(stimulated_nodes_all_periods[period]), T=T_period) #"recurrent" call
            M_ext_period = generate_M_ext('gaussian_process', graph, n_stimulated_nodes=len(stimulated_nodes_all_periods[period]), stimulated_nodes=list(stimulated_nodes_all_periods[period]), T=T_period, seed=seed) #"recurrent" call
            M_ext_new = transf_Mext(M_ext_period)
            M_ext_new_array = np.array([M_ext_new[node] if node in M_ext_new.keys() else np.zeros(T_period) 
                                        for node in list_nodes_graph])
        #     print(M_ext_new_array.shape)
            M_ext_array[:, period*T_period : (period+1)*T_period] = M_ext_new_array

        M_ext = {node : M_ext_array[list_nodes_graph.index(node)] for node in list_nodes_graph}
        
    elif type_M_ext == 'gaussian_process_by_periods_not_all_nodes': #allows to stimulate specific nodes, not necessarily all nodes the same number of times
        # print("entering not_all_nodes")
        # T_period = 500 #for example
        T = n_periods * T_period

        M_ext_array = np.zeros((len(list_nodes_graph), T))
        for period in range(n_periods):
            M_ext_period = generate_M_ext('gaussian_process', graph, 
                                          n_stimulated_nodes=n_stimulated_nodes, stimulated_nodes=stimulated_nodes, 
                                          T=T_period, n_periods=1, type_graph=type_graph, T_period=T_period, 
                                          variance_Mext=variance_Mext, 
                                          add_trigger_hallu=add_trigger_hallu, 
                                          simple_Mext=simple_Mext, seed=seed) #"recurrent" call #'gaussian_process_indep'
            M_ext_new = transf_Mext(M_ext_period)
            M_ext_new_array = np.array([M_ext_new[node] if node in M_ext_new.keys() else np.zeros(T_period) for node in list_nodes_graph])
            M_ext_array[:, period*T_period : (period+1)*T_period] = M_ext_new_array
        M_ext = {node : M_ext_array[list_nodes_graph.index(node)] for node in list_nodes_graph}
        
    #add triggers of the hallucination
    if add_trigger_hallu and type_graph == 'realistic_connectome_AAL':
    #     list_regions_trigger_hallu = ['insula_L', 'insula_R', 'hippocampus_parahippocampus_L', 'hippocampus_parahippocampus_R']
    #     list_regions_trigger_hallu = ['insula_L', 'insula_R']
        list_regions_trigger_hallu = ['hippocampus_parahippocampus_L', 'hippocampus_parahippocampus_R']
        times_triggering = np.linspace(0, len(M_ext[list(M_ext.keys())[0]])-1, 50).astype(int)
        times_triggering = times_triggering[times_triggering > 150] #remove the earlier triggers
        times_triggering = times_triggering[times_triggering < T-10] #remove the late triggers
        print("times_triggering", times_triggering)
        for node in list_regions_trigger_hallu:
            for time_triggering in times_triggering:
                M_ext[node][time_triggering] = M_ext[node][time_triggering] + 10
        times_hallu_all = [(i in times_triggering) or
                           (i-1 in times_triggering) or
                           (i-2 in times_triggering) or
                           (i-3 in times_triggering) or
                           (i-4 in times_triggering)
                           for i in range(T)]
        times_hallu_all = np.array([False] + times_hallu_all, bool)
    
    #adding a constant input ("prior")
#     print("herehere mean_Mext", mean_Mext)
#     print("type_M_ext", type_M_ext)
    assert isinstance(mean_Mext,str) or isinstance(mean_Mext,int) or isinstance(mean_Mext,float)
    if isinstance(mean_Mext, str):
        assert 'random' in mean_Mext
        if mean_Mext == 'random':
            amp = 1
        elif 'random_' in mean_Mext:
            #recover the amplitude of mean_Mext (number after random_)
            amp = float(mean_Mext[7:])
        M_ext = {key: val + np.random.uniform(low=-amp,high=amp) for key,val in M_ext.items()}
    elif mean_Mext != 0:
        M_ext = {key: val + mean_Mext for key,val in M_ext.items()}
        
    return M_ext

    
def add_noise_Mext(M_ext, noise_std):
    T = len(M_ext[list(M_ext.keys())[0]])
    return {key: val + np.random.normal(0, noise_std, size=T) for key,val in M_ext.items()}
    
def rbf_kernel(x1, x2, variance=30):
    """ 
    Useful function for generate_gaussian_process
    RBF kernel = Radial Basis Function kernel = squared exponential kernel = exponential quadratic kernel
    """
    return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))
    
def gram_matrix(xs, variance=30):
    """ 
    Useful function for generate_gaussian_process (returns the covariance matrix of the gaussian process)
    """
#     return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
#     return [rbf_kernel(x1,xs,variance) for x1 in xs] #half-vectorized
    X = np.expand_dims(xs, 1)
    return np.exp(-1/(2*variance) * scipy.spatial.distance.cdist(X, X, 'sqeuclidean')) #vectorized
    
    
def generate_white_noise(T, n_stimulated_nodes, variance=50):
    mean = 0
    std = np.sqrt(variance)
    if isinstance(std, float) or isinstance(std, int): #if std is a float
        return np.random.normal(0, std, size=(n_stimulated_nodes, T)) #the result is of size (n_stimulated_nodes, T)
    else: #np.array
        assert n_stimulated_nodes == len(variance)
        std = std.reshape((-1,1))*np.ones((n_stimulated_nodes, T))
        return np.random.normal(0, std) #the result is of size (n_stimulated_nodes, T)
    
def generate_gaussian_process(T, n_stimulated_nodes, variance=50):#, which='white_noise'):
    """
    Generates dependent Gaussian processes (from https://peterroelants.github.io/posts/gaussian-process-tutorial/ and https://gist.github.com/neubig/e859ef0cc1a63d1c2ea4)
    See also https://fr.wikipedia.org/wiki/Processus_continu
    and https://www.creatis.insa-lyon.fr/~vray/doc_cours/Cours_SgxAl.pdf
    
    The problem is that I'd like to have some correlation between X(t1,w) and X(t2,w),
    but no correlation between X(t,node1) and X(t,node2)  ( = no correlation between input to different nodes)
    and I'm not sure that this is what I get from this code
    So for now I'll be using only white noise (with no correlation between X(t1,w) and X(t2,w), and between X(t,node1) and X(t,node2))
    """
#     print("entering function generate_gaussian_process")
#     assert which in ['RBF', 'white_noise']
    
#     if which == 'white_noise':
#         return generate_white_noise(T, n_stimulated_nodes, variance)
    
#     elif which == 'RBF':
    if True:
        xs = np.linspace(1, T, T)
        mean = np.zeros(len(xs)) #[0 for x in xs]
        gram = gram_matrix(xs, variance)

#     #     plt_vals = []
#         ys_all = []
#         for i in range(0, n_stimulated_nodes):
#             ys = np.random.multivariate_normal(mean, gram) #this can be long (if xs is big, e.g. of size>500)
#             ys_all.append(ys)
#     #         plt_vals.extend([xs, ys, "k"])
#     #     plt.plot(*plt_vals)
#     #     plt.show()
        ys_all = np.random.multivariate_normal(mean, gram, size=n_stimulated_nodes) #one-liner of what's above (I'm not 100% sure that it does the same though... i.e. that processes are still independent. But it looks to be and it's also what is said in https://peterroelants.github.io/posts/gaussian-process-tutorial/)
        return np.array(ys_all)    

def create_stimulated_nodes_all_periods(list_nodes_graph=range(32), n_periods=4, n_stimulated_nodes=16, type_graph=None):
    """
    This function returns an array of shape (n_periods,n_stimulated_nodes) so that
    the i_period element indicated a list of n_stimulated_nodes unique nodes to be stimulated
    The function ensures that every node is stimulated the same number of times (and that
    it's not stimulated more than once in each period, as each element is a list of unique nodes)
    
    TODO: this function is slow (sometimes ~1 second); make it faster by exchanging the elements that cause pb
    instead of reshuffling everything.
    """
    if isinstance(n_stimulated_nodes, int):
        assert (n_stimulated_nodes*n_periods) % len(list_nodes_graph) == 0
        n_times_stimulations = n_stimulated_nodes*n_periods // len(list_nodes_graph)#number of periods that each node is stimulated
    else: #n_stimulated_nodes is a list
        assert n_periods % len(n_stimulated_nodes) == 0
        n_times_stimulations = n_periods // len(n_stimulated_nodes) #number of periods that each node is stimulated
    
    if (type_graph is None) == False:
        if ('realistic_connectome' in type_graph) or (n_periods >= 8): #special case
            #the code proposed below (for any type_graph) takes too long for more than 50 nodes. Instead, I cut the simulation into motives (in which every node is stimulated once), composed of an entire number of periods (= n_periods_per_motive)
            if isinstance(n_stimulated_nodes, int):
                assert len(list_nodes_graph) % n_stimulated_nodes == 0
#                 n_periods_per_motive = n_periods // n_motives
#                 n_motives = len(list_nodes_graph) // n_stimulated_nodes #number of times that the motif will repeat
                n_periods_per_motive = len(list_nodes_graph) // n_stimulated_nodes
                n_motives = n_periods // n_periods_per_motive #number of times that the motif will repeat
                stimulated_nodes_all_motives = np.concatenate([np.random.permutation(list_nodes_graph).reshape((n_periods_per_motive, n_stimulated_nodes)) for _ in range(n_motives)], axis=0)
            else: #n_stimulated_nodes is a list
                assert len(list_nodes_graph) == np.sum(n_stimulated_nodes)
                n_periods_per_motive = len(n_stimulated_nodes)
                n_motives = n_periods // n_periods_per_motive
                intervals = [0] + list(np.cumsum(n_stimulated_nodes))
#                 print("intervals", intervals)
#                 print(len(list_nodes_graph), len(np.unique(list_nodes_graph)))
                stimulated_nodes_all_motives = []
                for _ in range(n_motives):
                    shuffled_list = np.random.permutation(list_nodes_graph)
                    stimulated_nodes_all_motives.append([list(shuffled_list[intervals[i]:intervals[i+1]]) for i in range(len(intervals)-1)])
                stimulated_nodes_all_motives = list(itertools.chain.from_iterable(stimulated_nodes_all_motives))
            return stimulated_nodes_all_motives
    
    stimulated_nodes_all_periods_flatten = np.array(list(itertools.chain.from_iterable([list_nodes_graph]*n_times_stimulations)))

    reshuffle=True
    while reshuffle:
        np.random.shuffle(stimulated_nodes_all_periods_flatten) #shuffles in-place

        #check that no node is stimulated more than once inside a period
        stimulated_nodes_all_periods = stimulated_nodes_all_periods_flatten.reshape((n_periods, n_stimulated_nodes))
        # print(stimulated_nodes_all_periods.shape)
        reshuffle = False
        for period in range(n_periods):
            if max(Counter(stimulated_nodes_all_periods[period]).values()) > 1:
                reshuffle = True
                break
    #     print(reshuffle)
    
    return stimulated_nodes_all_periods

    
def transf_Mext(M_ext):
    """
    Smooths temporally the external stimulus so that it starts at 0 at ends at 0 (by modifying the beginning and the end)
    """
    T = len(M_ext[list(M_ext.keys())[0]])
    T1 = int(T*0.15)
    T3 = T1
    T2 = T - 2*T1
    M_ext_new = {}
    for node, z in M_ext.items():
        z1 = z[:T1] * np.linspace(0, 1, T1)
        z3 = z[T1+T2:] * np.linspace(1, 0, T3)
        z2 = z[T1:T1+T2]
        z_new = np.concatenate((z1,z2,z3))
        M_ext_new[node] = z_new
    return M_ext_new
    
    
def generate_M_ext_one(i, graph, std_Mext=1/2, seed=None):
    """
    i is not used (except that it differenciates 0 for which M_ext=0
    
    Yoon et al 2018 use std(Mext) = 1/2. Indeed, they have std(b_i) = 1/4 with b_i = M_ext / 2
    Before, I used std_Mext = np.sqrt(2)
    """
    if seed is None:
        np.random.seed() #to avoid redundancies in the examples (inside the training set / inside the test set / between the training set and the test set)
    else:
        np.random.seed(seed)
    #Create the external input
#     if i != 0: #default
# #         values_constant_input = np.random.normal(loc=np.random.normal(), size=len(list(graph.nodes))) #correlated inputs
#         values_constant_input = np.random.normal(scale=np.sqrt(2), size=len(list(graph.nodes))) #uncorrelated inputs
#     else:
#         values_constant_input = np.zeros(shape=len(list(graph.nodes)))
    values_constant_input = np.random.normal(scale=std_Mext, size=len(list(graph.nodes)))
    M_ext = generate_M_ext('constant', graph, stimulated_nodes='all', values_constant_input=values_constant_input)#, seed=seed)
    return M_ext
    

def generate_M_ext_all(graph, N=100, n_cpus=4, std_Mext=1/2, seed=None):
    """
    Generates data = couples (M_ext, p_1_true)
    """
#     print("entering generate_M_ext_all")
#     print("seed = {}".format(seed))
    if seed is not None:
        random.seed(seed)
        seeds = random.sample(range(1, N*1000), N)
#         print("seeds = {}".format(seeds))
    else:
        seeds = repeat(seed)
    with Pool(n_cpus) as p:
        list_M_ext = p.starmap(generate_M_ext_one, zip(range(N), repeat(graph), repeat(std_Mext), seeds))
    X = pd.DataFrame(list_M_ext) #transform into pandas dataframe
    return X