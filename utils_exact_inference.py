# import pgmpy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
import numpy as np
from utils_basic_functions import sig
# from utils_CI_BP import *
# from utils_graph_rendering import *
import pickle
import subprocess
import os, sys


def run_exact_inference(graph, M_ext):
    """
    Runs exact inference (BP on the Junction Tree) using library pgmpy
    Created based on function run_pgmpy (in run_pgmpy.py)
    Returns {node: log_odds_node}
    
    #Examples:
    
    #factors
    M_ext = {'a':1, 'b':2, 'c':3}
    graph = G_from_connections([['a','b'],['a','c'],['b','c']])
    graph.edges['a', 'b']['factor'] = np.array([[0.8,0.2],[0.2,0.8]])
    graph.edges['a', 'c']['factor'] = np.array([[0.7,0.3],[0.3,0.7]])
    graph.edges['b', 'c']['factor'] = np.array([[0.7,0.3],[0.3,0.7]])
    print(run_exact_inference(graph, M_ext))

    #weights
    M_ext = {'a':1, 'b':2, 'c':3}
    graph = G_from_connections([['a','b'],['a','c'],['b','c']])
    graph.edges['a', 'b']['weight'] = 0.8
    graph.edges['a', 'c']['weight'] = 0.7
    graph.edges['b', 'c']['weight'] = 0.7
    print(run_exact_inference(graph, M_ext))


    M_ext = {'a':0, 'b':0}
    graph = G_from_connections([['a','b']])
    graph.edges['a', 'b']['factor'] = np.array([[1,0.1],[0.7,0.2]])
    print(run_exact_inference(graph, M_ext))
    print(np.log(0.45/0.55), np.log(0.15/0.85))


    M_ext = {'a':10, 'b':0}
    graph = G_from_connections([['a','b']])
    graph.edges['a', 'b']['factor'] = np.array([[1,0.1],[0.7,0.2]])
    print(run_exact_inference(graph, M_ext))
    print(np.log(0.9/1.1)+10, np.log(0.2/0.7))
    """
    fg = FactorGraph() #alternatively, I could define the graph as a MarkovModel (it's possible to transform a FactorGraph into a MarkovModel and the reverse; it looks a bit simpler to define a MarkovModel; but I think the inference algorithms are just are quick in both cases)
    fg.add_nodes_from(list(graph.nodes))
    
    #binary factors
    for node1, node2, d in graph.edges(data=True):
        #check whether the graph has weights or factors
        if 'weight' in d.keys(): #weights
            w = d['weight']
            factor = np.array([[w, 1-w], [1-w, w]])
        elif 'factor' in d.keys(): #factors
            factor = d['factor'] #np array
        else:
            print("problem!")
            sys.exit()
        f = DiscreteFactor([node1, node2], [2, 2], factor)
        fg.add_factors(f)
        fg.add_edges_from([(node1, f), (node2, f)])
#     print(fg.edges)
        
    #unitary factors
    for node, M_ext_node in M_ext.items():
        unitary_factor = np.array([1 - sig(M_ext_node), sig(M_ext_node)]) #see create_unitary_factor used in run_pgmpy.py
        phi = DiscreteFactor([node], [2], unitary_factor.reshape((-1,1)))
        fg.add_factors(phi)
        fg.add_edges_from([(node, phi)])
    
#     print(fg.check_model())
#     print(fg.get_cardinality('V'))
#     print(fg.get_cardinality())
    
    # CHOICE OF THE INFERENCE METHOD
    use_bp = True #True #BELIEF PROPAGATION OR VARIABLE ELIMINATION

    # INFERING THE POSTERIOR PROBABILITY:
    if use_bp == False:
        from pgmpy.inference import VariableElimination #does not work? Still doesn't work, even without loops in the graph...
        infer = VariableElimination(fg)
    else:
        from pgmpy.inference import BeliefPropagation
        infer = BeliefPropagation(fg)
        
#     t0 = time.time()
    
#     print("node by node")
#     marginals_array = []
#     for node in graph.nodes:
#         posterior_p = infer.query([node], show_progress=False)
# #         print(posterior_p)
#         this = posterior_p#[node] # object of the class Discrete Factor
#         # Here is the code of __str__ of the class Discrete Factor (http://pgmpy.org/_modules/pgmpy/factors/discrete/DiscreteFactor.html#DiscreteFactor)
#         # I changed it so that we can fetch the values instead of having to print them
#     #     print(this.values)
#         marginals_array.append(this.values)
# #         print(posterior_p[node]) #prints in a table
#     marginals_array = np.array(marginals_array)
# #     print(marginals_array)
#     print(marginals_array[:,1] / marginals_array[:,0])
    
#     t1 = time.time()
    
#     print("node by node")
#     marginals_array = []
#     for node in graph.nodes:
#         posterior_p = infer.query([node], joint=False, show_progress=False)
# #         print(posterior_p)
#         this = posterior_p[node] # object of the class Discrete Factor
#         print(this.values)
#         # Here is the code of __str__ of the class Discrete Factor (http://pgmpy.org/_modules/pgmpy/factors/discrete/DiscreteFactor.html#DiscreteFactor)
#         # I changed it so that we can fetch the values instead of having to print them
#     #     print(this.values)
#         marginals_array.append(this.values)
# #         print(posterior_p[node]) #prints in a table
#     marginals_array = np.array(marginals_array)
# #     print(marginals_array)
#     print(marginals_array[:,1] / marginals_array[:,0])
    
#     t2 = time.time()
    
#     print("all nodes at once")
    marginals_array = []
    posterior_p = infer.query(list(graph.nodes), joint=False, show_progress=False) #joint=False means that we look at marginals (and not p(X_1,X_2) for instance)
    for node in graph.nodes:
        marginals_array.append(posterior_p[node].values)
#         print(posterior_p[node]) #prints in a table
    marginals_array = np.array(marginals_array)
#     print(marginals_array)
#     print(marginals_array[:,1] / marginals_array[:,0])

#     t3 = time.time()
    
#     print("t1 - t0 = {}".format(t1-t0))
#     print("t2 - t1 = {}".format(t2-t1))
#     print("t3 - t2 = {}".format(t2-t1))
    
    #Transform into logs
    beliefs_array = np.log(marginals_array[:,1] / marginals_array[:,0])

    ################ Saving the marginals into a file ############################
#     print(path_save_file)
#     np.save(path_save_file, beliefs_array) 
    #later: add in the file some information (apart from the marginals) like the type of graph, the factors, the weights, ...
    
    return dict(zip(list(graph.nodes), beliefs_array))


# def run_multiple_exact_inference(graph, list_M_ext):
#     """
#     Trying to fasten the execution of run_exact_inference by using compilation properties of pytorch
#     ---> doesn't fasten the code
#     """
#     # CHOICE OF THE INFERENCE METHOD
#     use_bp = True #True #BELIEF PROPAGATION OR VARIABLE ELIMINATION
    
#     fg = FactorGraph() #alternatively, I could define the graph as a MarkovModel (it's possible to transform a FactorGraph into a MarkovModel and the reverse; it looks a bit simpler to define a MarkovModel; but I think the inference algorithms are just are quick in both cases)
#     fg.add_nodes_from(list(graph.nodes))
    
#     #binary factors
#     for node1, node2, d in graph.edges(data=True):
#         w = d['weight']
#         factor = np.array([[w, 1-w], [1-w, w]])
#         f = DiscreteFactor([node1, node2], [2, 2], factor)
#         fg.add_factors(f)
#         fg.add_edges_from([(node1, f), (node2, f)])

# #     print(fg.edges)
    
#     list_dict_beliefs = []
#     for M_ext in list_M_ext:
        
#         #unitary factors
# #         print("adding unitary factors")
#         for node, M_ext_node in M_ext.items():
#             unitary_factor = np.array([1 - sig(M_ext_node), sig(M_ext_node)]) #see create_unitary_factor used in run_pgmpy.py
#             phi = DiscreteFactor([node], [2], unitary_factor.reshape((-1,1)))
#             fg.add_factors(phi)
#             fg.add_edges_from([(node, phi)])
# #         print(fg.edges)
        
#         #INFERING THE POSTERIOR PROBABILITY
#         if use_bp == False:
#             infer = VariableElimination(fg) #does not work? Still doesn't work, even without loops in the graph
#         else:
#             infer = BeliefPropagation(fg)
#         marginals_array = []
#         posterior_p = infer.query(list(graph.nodes), joint=False, show_progress=False) #joint=False means that we look at marginals (and not p(X_1,X_2) for instance)
#         for node in graph.nodes:
#             p_test = posterior_p[node].values # Here is the code of __str__ of the class Discrete Factor (http://pgmpy.org/_modules/pgmpy/factors/discrete/DiscreteFactor.html#DiscreteFactor). I changed it so that we can fetch the values instead of having to print them
#             marginals_array.append(p_test)
#         marginals_array = np.array(marginals_array)
#     #     print(marginals_array)
#     #     print(marginals_array[:,1] / marginals_array[:,0])

#         #Transform into logs
#         beliefs_array = np.log(marginals_array[:,1] / marginals_array[:,0])

#         dict_beliefs = dict(zip(list(graph.nodes), beliefs_array))
#         list_dict_beliefs.append(dict_beliefs)
        
# #         print("Removing unitary factors")
#         for node, M_ext_node in M_ext.items():
#             unitary_factor = np.array([1 - sig(M_ext_node), sig(M_ext_node)]) #see create_unitary_factor used in run_pgmpy.py
#             phi = DiscreteFactor([node], [2], unitary_factor.reshape((-1,1)))
#             fg.remove_factors(phi)
# #             fg.add_edges_from([(node, phi)])
# #         print("new list of edges:")
# #         print(fg.edges)
    
#     return list_dict_beliefs


def get_exact_inference_factorgraph_library(graph, M_ext):
    """
    This function is not used anywhere. I think that was a previous version of what's above, 
    using a different library (factorgraph, which only works in Python2)
    
    Gets the "real" marginals of each node (could be the stable state)
    Computes the marginals with the factorgraph library
    --------> actually runs Loopy BP, not exact inference!!!
    --------> TODO: use function run_exact_inference instead
    """
    sys.path.append(os.getcwd() + '/factorgraph')

    dir_to_save = '../../results_code/simulations_CI_BP/factorgraph-marginals/'
    which_python_to_run = "/home/vincent/anaconda3/envs/factorgraph_py2/bin/python" #not just "python" because the script I want to run needs to be in the right environment

    # M_ext = {13: 1}
    with_M_ext = False

    #save the inputs to the function
    filename_graph = 'graph_input_for_function'
    with open(dir_to_save + filename_graph + '.pkl', 'wb') as file:
        pickle.dump(graph, file, protocol=2) #for Python2 to be able to open it
    if with_M_ext == True:
        filename_M_ext = 'M_ext_input_for_function'
        with open(dir_to_save + filename_M_ext + '.pkl', 'wb') as file:
            pickle.dump(M_ext, file, protocol=2) #for Python2 to be able to open it. #M_ext should be constant over time

    # result = os.system(which_python_to_run + " factorgraph/run_factorgraph_general.py 10 a=12 b=13")
    # # result = subprocess.check_output(which_python_to_run + " factorgraph/run_factorgraph_general.py 10 a=12 b=13", shell=True)
    # # result = subprocess.check_output(which_python_to_run + " factorgraph/run_factorgraph_general.py '{}'".format(graph), shell=True)
    if with_M_ext == False:
        result = subprocess.check_output("{} factorgraph/run_factorgraph_general.py {}".format(which_python_to_run, filename_graph), shell=True)
    else:
        result = subprocess.check_output("{} factorgraph/run_factorgraph_general.py {} {}".format(which_python_to_run, filename_graph, filename_M_ext), shell=True)
    # # result = subprocess.run(which_python_to_run + " factorgraph/run_factorgraph_general.py", input=graph, shell=True)
    # import subprocess
    # proc = subprocess.Popen([which_python_to_run, "factorgraph/run_factorgraph_general.py 10 a=12 b=13"], stdout=subprocess.PIPE, shell=True)
    # (out, err) = proc.communicate()
    # print(out, err)

    print(result)
    return result