#This file allows to fit any model (with supervised or unsupervised methods) with given (which_CI, which_alpha, which_w, which_Mext) to perform approximate inference

# -*- coding: utf-8 -*-

# from utils_plot_dict import *
from simulate import *
from graph_generator import create_random_weighted_graph, get_w_matrix #,generate_graph
from generate_Mext import generate_M_ext, generate_M_ext_one, generate_M_ext_all
from utils_exact_inference import *
# import os, sys
# sys.path.append(os.getcwd()[:-len(os.getcwd().split('/')[-1])]) #adds to python path, the parent directory: see print(sys.path)
# from functions_utils import W, W_prime, create_factor, create_unitary_factor#,get_mean_durations,get_durations  #needs the 2 lines above to work correctly
# from scipy.stats import logistic
from sklearn.metrics import r2_score
import pandas as pd
import pandas
from scipy.optimize import least_squares
# from jax import jit #I removed that as it is not speeding up the computations
# import jax.numpy as np #I removed that as it is not speeding up the computations
import numpy as np
from itertools import repeat, chain #, product
import random
from multiprocessing import cpu_count, Pool
n_cpus = 4 #cpu_count() #4 gives the fastest results (even though there are cpu_count()=8 cpus on my machine...)
from utils_basic_functions import sig, from_dict_to_matrix, from_dict_to_vector
import time
from copy import deepcopy, copy #copy is shallow copy (implemented for most objects: obj.copy() is the same)
# from utils_pytorch import *
# import utils_pytorch
# from utils_plot import *
from utils_dict import get_keys_min_val
from utils_save_load_file import *
from utils_data_convert import *
# from utils_stability_analysis import *

    
def KL_divergence(p_1_true_vec, p_1_approx_vec, log=False):
    """
    Returns KL(p_i(x_i) || \hat{p}_i(x_i)) = sum_{x_i}  p_i(x_i) . log(p_i(x_i) / \hat{p}_i(x_i))
    
    log = True means that p_1_approx_vec is in fact B_approx_vec  (p(X=1) = sig(B) and p(X=0) = 1-sig(B) = sig(-B))
    
    I tested the function --> ok
    
    Example:
    p_1_true_vec = np.array([0.95,0.4,0.55,0.001])
    B_1_approx_vec = np.array([35,-1.32,0.1,-80])
    p_1_approx_vec = 1 / (1 + np.exp(-B_1_approx_vec))
    print(KL_divergence(p_1_true_vec, p_1_approx_vec, log=False))
    print(KL_divergence(p_1_true_vec, B_1_approx_vec, log=True))
    """
    p_true = np.array([p_1_true_vec, 1 - p_1_true_vec]) #(p(X=1), p(X=0))
    if log == False:
        p_approx = np.array([p_1_approx_vec, 1 - p_1_approx_vec]) #(p(X=1), p(X=0))
    #     return scipy.stats.entropy(p_true, p_approx) #slower than the line below
        return np.sum(p_true * np.log(p_true / p_approx), axis=0)
    else:
        B_approx = np.array([p_1_approx_vec, - p_1_approx_vec]) #(B, -B) --> sig(...) gives (p(X=1), p(X=0))
        return np.sum(p_true * (np.log(p_true) - logsig(B_approx)), axis=0)

    
def sqrt_loss(x):
    """
    Loss used in least-square function (scipy optimizer)
    
    "If callable, it must take a 1-D ndarray z=f**2 and return an array_like with shape (3, m) where row 0 contains function values, row 1 contains first derivatives and row 2 contains second derivatives. Method ‘lm’ supports only ‘linear’ loss."
    """
#     return np.array([np.sqrt(x), 1/2 * x**(-1/2), -1/4 * x**(-3/2)]) #slower than the lines below
    x_sqrt = np.sqrt(x)
    return np.array([x_sqrt, 1 / (2 * x_sqrt), -1/(4 * x * x_sqrt)])

def is_saturating(l, percent=1, back=20):
    """
    Early stopping criterion: 1% difference max in the last 20 iterations
    """
    converged = (100 * np.abs(l[-1] - l[-back]) / l[-1] < percent) #percent % difference at max
    return converged

def is_valid_model(which_CI, which_alpha, which_w, which_Mext, verbose=True):
    
    if which_CI == 'BP' and which_alpha == '1' and which_w is None and which_Mext is None:
        return True
    
    if which_CI == 'BP' and which_alpha != '1':
        print("PASS (not a valid BP model)")
        return False
    
    if (which_CI == 'BP' or which_alpha in ['1', '0']) and which_w is None and which_Mext is None:
        print("PASS (we want to fit something)")
        return False
    
    if 'CIpower' in which_CI and which_alpha in ['0', 'directed']:
        print("PASS (impossible model)")
        return False
    
    if which_CI in ['full_CIpower_approx', 'full_CIpower_approx_approx_tanh', 'rate_network'] and which_alpha in ['0', 'directed', 'undirected', 'directed_ratio', '1_directed_ratio', 'directed_ratio_directed']: #these models don't use any K_edges (contrary to 'full_CIpower' and 'full_CIpower_approx_tanh')
        print("PASS (impossible model)")
        return False
    
    if ('power' in which_CI and which_CI != 'BP') and which_alpha == '1':
        print("PASS (BP covers this model + it's not implemented)")
        return False
        
    if which_CI == 'CInew' and which_alpha in ['0', '1']:
        print("PASS (impossible model)")
        return False
        
    if (which_CI not in ['BP', 'rate_network']) and which_alpha == '1':
        print("PASS (BP covers this model)")
        return False
        
#     if which_alpha not in ['1', '0'] and which_w is not None:
#         print("PASS (we don't want to fit both w and alpha for now)") #but in the code nothing prevents from fitting both w and alpha
#         return False
    
    return True
    
    
def fit_models_list(list_X, list_y, list_graph,
                    which_CI, which_alpha, which_w, which_Mext, method_fitting,
                    damping=0, k_max=None,
                    options={},
                    plot_graph=False,
                    parallel_graphs=True,
                    parallel_Mext=True,
                    list_which_error=['KL','MSE', 'CE'], portions_dataset={'train': 0.5, 'val': 0.25, 'test': 0.25}
                   ):
        
    class Object(object):
        pass
    res = Object()

    if method_fitting == 'unsupervised_learning_rule':
        #generating a bigger X_train (training set), without y_train (otherwise too long)
        nodes_max = 10 #max number of nodes for the graph used later
        print("check that nodes_max is ok (in function fit_models_list in utils_compensating_alpha.py")
        assert 'N_training_ex' in options.keys()
        N_training_ex = options['N_training_ex']
        print("add possibilities here by giving additionnal arguments (in particular std_Mext)")
        X_all = generate_M_ext_all(nx.erdos_renyi_graph(n=nodes_max, p=1),
                                   N=N_training_ex, std_Mext=1/2
                                  ) #the 1st argument (graph) is used not later, it is just needed for the number of nodes
    else:
        X_all = None
    
    if 'pytorch' in method_fitting:
        utils_pytorch.get_model(list_graph[0], which_CI, which_alpha, which_w, which_Mext, print_free_params=True) #just prints the model's free parameters
    
    if 'pytorch' in method_fitting and torch.cuda.is_available():
        #not using multiprocessing (as it is not compatible with multiprocessing, or at least I couldn't make it work. I identified this by running the code outside of jupyter notebook, i.e., using .py files)
        print("Changing parallel_graphs to False (multiprocessing and CUDA are not compatible)")
        parallel_graphs = False
    
    #Non-parallel implementation
    if parallel_graphs == False:
        if method_fitting != 'unsupervised_learning_rule':
            res.list_error_train = []
        res.list_error_val = []
        res.list_error_test = []
        res.list_alpha_fitted = []
        res.list_w_fitted = []
        res.list_w_input_fitted = []
        if method_fitting == 'unsupervised_learning_rule':
            if 'keep_history_learning' in options.keys() and options['keep_history_learning'] == True:
                res.list_error_val_history = []
                res.list_error_test_history = []
                res.list_alpha_fitted_history = []
                res.list_w_fitted_history = []
                res.list_w_input_fitted_history = []
        for ind, (graph, X, y) in enumerate(zip(list_graph, list_X, list_y)):

#             if ind != 14:
#                 continue
            print("-------- ind = {} --------".format(ind))

            res_one = fit_model(X, y, graph, 
                                which_CI, which_alpha, which_w, which_Mext, method_fitting, 
                                damping=damping, k_max=k_max,
                                X_all=X_all,
                                options=options,
                                plot_graph=plot_graph,
                                parallel_Mext=parallel_Mext,
                                list_which_error=list_which_error, portions_dataset=portions_dataset
                           )
            #maybe I should plot the loss curve if the validation loss did not converge (as in the parallel implementation below)?
            if 'error_train' in res_one.keys():
                res.list_error_train.append(res_one['error_train'])
            res.list_error_val.append(res_one['error_val'])
            res.list_error_test.append(res_one['error_test'])
            res.list_alpha_fitted.append(res_one['alpha_fitted'])
            res.list_w_fitted.append(res_one['w_fitted'])
            res.list_w_input_fitted.append(res_one['w_input_fitted'])
            if 'error_val_history' in res_one.keys():
                res.list_error_val_history.append(res_one['error_test_history']) #error_val_history or error_test_history?
            if 'error_val_history' in res_one.keys():
                res.list_alpha_fitted_history.append(res_one['alpha_history'])
                res.list_w_fitted_history.append(res_one['w_history'])
                res.list_w_input_fitted_history.append(res_one['w_input_history'])

    #Parallel implementation
    else: #if parallel_graphs == True:
        from multiprocessing import cpu_count, Pool
        n_cpus = 4 #cpu_count() #4 gives the fastest results (even though there are cpu_count()=8 cpus on my machine...)
        with Pool(n_cpus) as p:
            list_res = p.starmap(fit_model,
                                 zip(list_X, list_y, list_graph, 
                                     repeat(which_CI), repeat(which_alpha), repeat(which_w), repeat(which_Mext),
                                     repeat(method_fitting), 
                                     repeat(damping), repeat(k_max),
                                     repeat(X_all),
                                     repeat(options),
                                     repeat(plot_graph),
                                     repeat(parallel_Mext), repeat(list_which_error), repeat(portions_dataset),
                                     range(len(list_graph))))
        #plot the loss curve if the validation loss did not converge
        for model, i_graph in zip(list_res, range(len(list_graph))):
            if 'no_convergence' in model.keys() and model['no_convergence'] == True:
                loss_all_epochs = model['loss_all_epochs']
                loss_val_all_epochs = model['loss_test_all_epochs']
                plt.plot(loss_all_epochs, label='train')
                plt.plot(loss_val_all_epochs, label='val')
                plt.legend()
                plt.title("i_graph = {}: loss - try #3 (should work)".format(i_graph))
                plt.xlabel("iteration")
                plt.show()
        
        if 'error_train' in list_res[0].keys():
            res.list_error_train = [res['error_train'] for res in list_res]
        res.list_error_val = [res['error_val'] for res in list_res]
        res.list_error_test = [res['error_test'] for res in list_res]
        res.list_alpha_fitted = [res['alpha_fitted'] for res in list_res]
        res.list_w_fitted = [res['w_fitted'] for res in list_res]
        res.list_w_input_fitted = [res['w_input_fitted'] for res in list_res]
        if 'error_val_history' in list_res[0].keys():
            res.list_error_val_history = [res['error_val_history'] for res in list_res]
        if 'alpha_history' in list_res[0].keys():
            res.list_alpha_fitted_history = [res['alpha_history'] for res in list_res]
            res.list_w_fitted_history = [res['w_history'] for res in list_res]
            res.list_w_input_fitted_history = [res['w_input_history'] for res in list_res]

    return res
    

def fit_model(X, y, graph, 
              which_CI, which_alpha, which_w, which_Mext, method_fitting, 
              damping=0, 
              k_max=None, #for CIbeliefs
              X_all=None,
              options=None,
              plot_graph=False,
              parallel_Mext=True,
              list_which_error=['KL','MSE', 'CE'],
              portions_dataset={'train': 0.5, 'val': 0.25, 'test': 0.25},
              print_index=None, log=False, which_error_print='MSE'
             ):
    """
    Note that list_which_error does not change how the model is fitted (=based on which error measure), but instead only which error is computed between the y_predict and y_true, which is returned.
    Anyway, the error can be computed very easily (using the saved alpha_fitted) with other error measures  (but it's a bit long)
    """
    if plot_graph:
        plot_graph(graph, print_w_edges=True)

    res = {}
    
    #Define the model
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(X, y, portions_dataset=portions_dataset,
                                                                      return_test=True)
#     print("sizes: X_train = {}, X_val = {}, X_test = {}, y_train = {}, y_val = {}, y_test = {}".format(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape))
    if method_fitting == 'unsupervised_learning_rule':
        X_train = X_all[list(graph.nodes)] #unsupervised learning
    model = Model_alpha_ij_all(graph, which_CI, which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext,
                               damping=damping, k_max=k_max)

    #Compute things for BP
#     print("BP")
    y_predict = Model_alpha_ij_CI(graph, damping=damping).predict_BP(X_val, log=log)
    error_BP = accuracy_score(y_predict, y_val, which_error=which_error_print, log=log)
    if print_index is None:
        print("val set: {} error (initial BP) = {}".format(which_error_print, error_BP))
    else:
        print("index = {} - val set: {} error (initial BP) = {}".format(print_index, which_error_print, error_BP))

    #Fit the model
#     print("Fit the model")
    if method_fitting in ['supervised_MSE_scipy', 'supervised_KL_scipy']:
        model.fit(X_train, y_train, method_fitting=method_fitting, parallel_Mext=parallel_Mext, 
                  options=options)
    elif method_fitting in ['supervised_KL_pytorch', 'supervised_CE_pytorch', 'supervised_MSE_pytorch']:
        model.fit(X_train, y_train, method_fitting=method_fitting, parallel_Mext=parallel_Mext, 
                  options=options, X_test=X_val, y_test=y_val) #val data provided: needed for the stopping criterion used in Pytorch fitting (stable validation loss)
    elif method_fitting == 'unsupervised_learning_rule':
        model.fit(X_train, method_fitting=method_fitting,
                  options=options,
                  parallel_Mext=parallel_Mext
                 ) #unsupervised learning --> no need for y
    elif method_fitting == 'unsupervised':
        model.fit(X_train, method_fitting=method_fitting, parallel_Mext=parallel_Mext,
                  options=options) #unsupervised learning --> no need for y
    else:
        print("method_fitting is not an available option")
        sys.exit()
    
    #Dealing with no convergence of the validation loss (bad training?)
    if hasattr(model, 'no_convergence') and model.no_convergence == True:
        print("inside fit_model function - trying to plot")
        loss_all_epochs = model.loss_all_epochs
        loss_val_all_epochs = model.loss_test_all_epochs
        plt.plot(loss_all_epochs, label='train')
        plt.plot(loss_val_all_epochs, label='val')
        plt.legend()
        plt.title("loss - try #2 (but shouldn't work)")
        plt.xlabel("iteration")
        plt.show()
        res['no_convergence'] = True
        res['loss_all_epochs'] = loss_all_epochs
        res['loss_val_all_epochs'] = loss_val_all_epochs
    
    #Compute the error on the training, validation, and test sets
    d_error_all_sets = {}
    dict_Xy = {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [X_test, y_test]}
    list_sets = ['train', 'val', 'test'] #if method_fitting != 'unsupervised_learning_rule' else ['val', 'set'] #careful: if unsupervised_learning_rule, then another X_train is taken
    for which_set in list_sets:
        X_select, y_select = dict_Xy[which_set]
        y_predict = model.predict(X_select, parallel_Mext=parallel_Mext, log=log)
    #     print("which_set = {}".format(which_set))
    #     print("y_predict[55]", y_predict[55])
    #     print("y_predict", pd.DataFrame(y_predict).to_numpy())
    #     print("y_select[55]", y_select.to_numpy()[55])
    #     print("len(y_predict) = {}, len(y_select) = {}".format(len(y_predict), len(y_select)))
        error_set = accuracy_score(y_predict, y_select, list_which_error=list_which_error, log=log)
        if print_index is None:
            print("{} set: {} error ({}) = {}".format(which_set, which_error_print, which_CI, error_set[which_error_print]))
        else:
            print("index = {} - {} set: {} error ({}) = {}".format(print_index, which_set, which_error_print, which_CI, error_set[which_error_print]))
        d_error_all_sets[which_set] = error_set
        
    #Compute the error on the val set during training (--> history of the error)
    if hasattr(model, 'alpha_history'):
        alpha_history = model.alpha_history
        print("I couldn't find function get_history_error --> implement if necessary")
        error_history = get_history_error(X_val, y_val, alpha_history) #add argument list_which_error
        x = list(error_history.keys())
        y = list(error_history.values())
        plt.plot(x, y, label='learning')
        plt.xlabel("iteration")
        plt.ylabel("error (val set)")
        plt.axhline(y=error_BP, label='BP', linestyle='--', color='black')
        if print_index is not None:
            plt.title(print_index)
        plt.legend()
        plt.show()

    #Save the results into variable
    if method_fitting != 'unsupervised_learning_rule':
        res['error_train'] = d_error_all_sets['train']
    res['error_val'] = d_error_all_sets['val']
    res['error_test'] = d_error_all_sets['test']
    res['alpha_fitted'] = model.alpha
    res['w_fitted'] = model.w
    res['w_input_fitted'] = model.w_input
    if hasattr(model, 'alpha_history'): #i.e., keep_history_learning == True and which_CI != 'BP'
        res['error_val_history'] = error_history
        res['alpha_fitted_history'] = alpha_history
        res['w_fitted_history'] = w_history
        res['w_input_fitted_history'] = w_input_history
#     print("alpha", model.alpha)
#     print("w", model.w)
#     print("w_input", model.w_input)
        
    return res
    

def starting_point_least_squares(graph, which_CI, which_alpha, which_w=None, which_Mext=None, k_max=None):
    """
    Determines:
    - x0 (starting point for least_squares): a vector which size depends on the model used, and represents alpha
    - the bounds for the least-square minimization
    """
    assert xor(k_max is None, which_CI == 'CIbeliefs') #k_max is provided for and only for which_CI == 'CIbeliefs'
    assert not(which_w is None and which_alpha in ['0','1'] and which_Mext is None) #otherwise there is nothing to fit
    
    #starting point for alpha
    if which_alpha not in ['0', '1']:
        if which_CI != 'CIbeliefs':
            if which_CI == 'CIbeliefs2':
                x0_alpha_val = {'val': 0} #x0_alpha_val = 0
                bounds_alpha = (-2, 2) #because the alpha_ij from CIbeliefs2 is in fact (with alpha_ij from CI) alpha_ij*J_ji (where J_ji can be negative)
    #         elif which_CI == 'CIpower': #alpha_ij = K_nodes[i] / K_edges[i,j], also written = alpha_i / K_ij
    #             x0_alpha_val = 1
    #             if which_alpha in ['uniform', 'nodal']: #alpha_ij = alpha_i, which should be in [-2,4]
    #                 bounds_alpha = (-2, 4)
    #             elif which_alpha == 'undirected':
    #                 bounds_alpha = (1/5, np.inf) #so that alpha = 1 / K_ij is < 5 (here we prevent alpha from being <0, but I think it's not a problem). Anyway, if alpha is too big, then it can create numerical problems in arctanh, probably because the new factor has factors which are too big if K_ij is too close to 0, positive or negative - anyway the solutions <0 seem to be bad)
    #             elif which_alpha == 'directed_ratio': #we want alpha_ij = alpha_i / K_ij to be <5 (or at least not too big)
    #                 bounds_Knodes = (-2, 4)
    #                 bounds_Kedges = (1/5, np.inf)
    #                 #we concatenate the bounds, using that x0 indicates K_nodes and then K_edges
    #                 bounds_alpha = ([bounds_Knodes[0]]*len(list(graph.nodes)) + [bounds_Kedges[0]]*len(list(graph.edges)),
    #                           [bounds_Knodes[1]]*len(list(graph.nodes)) + [bounds_Kedges[1]]*len(list(graph.edges)))
    #                 #Note that an alternative would be to indicate the constraint that we shouldn't have max(K_nodes/K_edges) be >4, but I don't think that it is possible with function least_squares, and also it would also control for K_nodes[i]/K_edges[j,k] and not only for K_nodes[i]/K_edges[i,j]
    #             else:
    #                 print("This case is not dealt with - which_CI = {} and which_alpha = {}".format(which_CI, which_alpha))
            else:
                x0_alpha_val = {'K_nodes': 1, 'inverse_K_edges': 1, 'val': 1} #x0_alpha_val = 1 #alpha = 1 (BP)
                
                if which_alpha in ['directed_ratio', 'directed_ratio_directed']:
                    if which_CI in dict_get_stability_matrix.keys(): #dict_get_stability_matrix is defined in utils_stability_analysis.py
                        
                        #choosing (K_edges, K_nodes) such that the algo converges with K_edges = K_nodes  (file 'supervised2')
#                         K = get_Knodes_Kedges_val_convergence_algo(graph, which_CI)
#                         print("initial K = {} (K_nodes=K_edges=K)".format(K))
#                         x0_alpha_val['K_nodes'] = K
#                         x0_alpha_val['inverse_K_edges'] = 1/K
                        
                        #choosing (K_edges=1, K_nodes) such that the algo converges (file 'supervised3', + 'supervised4'  as well because it seems that some supervised3 files actually have a mistake: x0_alpha_val was = 1/K instead of 1...)
#                         K = get_Knodes_val_convergence_algo(graph, which_CI)
#                         print("initial K = {} (K_nodes=K and K_edges=1)".format(K))
#                         x0_alpha_val['K_nodes'] = K
#                         x0_alpha_val['inverse_K_edges'] = 1 #1/K
                        
                        #choosing (K_edges=1, K_nodes=1) + low beta such that the algo converges (file 'supervised4' for some models like eFBP)
                        print("initial K_nodes=1 and K_edges=1")
                        x0_alpha_val['K_nodes'] = 1
                        x0_alpha_val['inverse_K_edges'] = 1
                        
                if which_alpha in ['directed', '1_directed_ratio']:
                    x0_alpha = np.array([x0_alpha_val['val']] * len(list(graph.edges)) * 2) #alpha_{i \to j}
                    bounds_alpha = (-np.inf, +np.inf) #(-10, +np.inf) #(-10, 12)
                elif which_alpha == 'undirected':
                    x0_alpha = np.array([x0_alpha_val['inverse_K_edges']] * len(list(graph.edges))) #alpha_{ij}  (such that alpha_{ij} = alpha_{i \to j} = alpha_{j \to i})
                    bounds_alpha = (-np.inf, +np.inf) #(-4, np.inf)
                elif which_alpha in ['nodal', 'nodal_out']:
                    
                    #choosing (K_edges=1, K_nodes) such that the algo converges (file 'supervised2')
#                     K = get_Knodes_val_convergence_algo(graph, which_CI)
#                     print("initial K = {} (K_nodes=K and K_edges=1)".format(K))
#                     x0_alpha_val['K_nodes'] = K

                    #choosing (K_edges=1, K_nodes=1) (file 'supervised1' or 'supervised' for some models) + low beta ('supervised4')
                    print("initial K = {} (K_nodes=1 and K_edges=1)".format(1))
                    x0_alpha_val['K_nodes'] = 1
            
                    x0_alpha = np.array([x0_alpha_val['K_nodes']] * len(list(graph.nodes))) #alpha_i or alpha_j
                    bounds_alpha = (-np.inf, +np.inf) #(-2, +np.inf)
                elif which_alpha == 'uniform':
                    x0_alpha = np.array([x0_alpha_val['val']]) #alpha
                    bounds_alpha = (-2, 4) #(0,2) #for CI, alpha is often fitted >2. I allow it to be <0 for the point 1 to be at the center of the bounds (and because sometimes alpha are fitted at ~0, which might be because it's trying to compensate for the loops for which alpha_ij~2 (and not more...), but maybe not only --> allow it and see. (I hope it won't often fit alpha<0 or >2...)
                elif which_alpha == 'directed_ratio': #default
                    x0_alpha = np.array(([x0_alpha_val['K_nodes']] * len(list(graph.nodes))) + ([x0_alpha_val['inverse_K_edges']] * len(list(graph.edges)))) #K_i and inverse_K_ij (such that alpha_ij = K_i / K_ij) #x0_alpha = np.array([x0_alpha_val] * (len(list(graph.edges)) + len(list(graph.nodes)) )) #K_i and inverse_K_ij (such that alpha_ij = K_i / K_ij)
                    bounds_Knodes = (-np.inf, +np.inf) #(-2, +np.inf)
                    bounds_inverse_Kedges = (-np.inf, +np.inf) #actually 1/Kedges, not K_edges #(-2, np.inf)
                    #we concatenate the bounds, using that x0 indicates K_nodes and then K_edges
                    bounds_alpha = ([bounds_Knodes[0]]*len(list(graph.nodes)) + 
                                    [bounds_inverse_Kedges[0]]*len(list(graph.edges)),
                                    [bounds_Knodes[1]]*len(list(graph.nodes)) + 
                                    [bounds_inverse_Kedges[1]]*len(list(graph.edges)))
                elif which_alpha == 'directed_ratio_directed':
                    x0_alpha = np.array(([x0_alpha_val['K_nodes']] * len(list(graph.nodes))) + ([x0_alpha_val['inverse_K_edges']] * (len(list(graph.edges)) * 2))) #K_i and K_ij (such that alpha_ij = K_i / K_ij)
                    bounds_Knodes = (-np.inf, +np.inf)
                    bounds_inverse_Kedges = (-np.inf, np.inf)
                    bounds_alpha = ([bounds_Knodes[0]]*len(list(graph.nodes)) + 
                                    [bounds_inverse_Kedges[0]]*len(list(graph.edges))*2,
                                    [bounds_Knodes[1]]*len(list(graph.nodes)) + 
                                    [bounds_inverse_Kedges[1]]*len(list(graph.edges))*2)
                else:
                    raise NotImplemented #it shouldn't appear (we covered all cases)
                    
        else: #which_CI == 'CIbeliefs':
            x0_alpha_val = {'val': 0} #x0_alpha_val = 0
            bounds_alpha = (-1, 1)
            if which_alpha == 'nodal_temporal':
                x0_alpha = np.array([x0_alpha_val['val']] * len(list(graph.nodes)) * (k_max - 1)) #alpha = 0  #TODO: have a better starting point: remove the linear effect for k=2, i.e. alpha_{j,2}=sum_i J_ij*J_ji (note that for now we used J_ij=J_ji)
            elif which_alpha == 'temporal': #alpha_{j, k} = alpha_k, i.e., alpha[j] are all identical (but are vectors still)
                x0_alpha = np.array([x0_alpha_val['val']] * (k_max - 1))
            elif which_alpha == 'nodal': #alpha_{j, k} = alpha_j, i.e., alpha[j] = [alpha_j, alpha_j, alpha_j, ...etc]
                x0_alpha = np.array([x0_alpha_val['val']] * len(list(graph.nodes)))
            elif which_alpha == 'uniform':
                x0_alpha = np.array([x0_alpha_val['val']])
            else:
                print("which_alpha = {} is not supported in function from_vec_to_obj".format(which_alpha))
                sys.exit()
    else: #which_alpha in ['0', '1']
        x0_alpha = np.array([])
        bounds_alpha = ([], []) #None
    
    #Starting point for w (if which_w is not None) - keep in mind that w is not a multiplicative coefficient but directly the graph weights
    if which_w is not None:
        assert which_w in ['undirected', 'directed']
        if which_w == 'undirected':
            x0_w = np.array([graph.edges[edge]['weight'] for edge in graph.edges]) #taking weights for graph as starting point
        elif which_w == 'directed':
            x0_w = np.array([graph.edges[edge]['weight'] for edge in get_all_oriented_edges(graph)]) #taking weights for graph as starting point
        if which_alpha in ['0', '1', 'nodal', 'undirected', 'directed_ratio', 'directed_ratio_directed']:
            #choosing (K_edges=1, K_nodes) such that the algo converges (--> reduces the amplitude of weights)
            beta_val = get_beta_val_convergence_algo(graph, which_CI)
            print("initial beta = {}".format(beta_val)) #'initial beta = {} (K_nodes=1 and K_edges=1)'
            tanh_Jij = 2 * x0_w - 1
            tanh_Jtildeij = np.tanh(beta_val * tanh_inverse(tanh_Jij))
            x0_w = 1/2 + 1/2 * tanh_Jtildeij
        if which_CI == 'rate_network':
            bounds_w = (-np.inf, +np.inf)
        else: #default
            bounds_w = (0, 1)
        bounds_w = ([bounds_w[0]] * len(x0_w), [bounds_w[1]] * len(x0_w))
    else:
        x0_w = np.array([])
        bounds_w = ([], []) #None
    
    #Starting point for w_input (if which_Mext is not None)
    if which_Mext is not None:
        assert which_Mext == 'nodal'
        x0_w_input_val = 1  #taking 1 as starting point (as for BP)
        x0_w_input = np.array([x0_w_input_val] * len(list(graph.nodes)))
        bounds_w_input = (-np.inf, +np.inf)
        bounds_w_input = ([bounds_w_input[0]] * len(x0_w_input), [bounds_w_input[1]] * len(x0_w_input))
    else:
        x0_w_input = np.array([])
        bounds_w_input = ([], [])
        
    #Returning a concatenation (alpha, w, and w_input) depending on the case
    x0, bounds = concatenate_arr(x0_alpha, x0_w, x0_w_input,
                                 bounds_alpha, bounds_w, bounds_w_input
                                )
#     print("here", len(x0), len(bounds[0]), len(bounds[1]))
#     print("what it should be", len(graph) + len(graph.edges))
    return x0, bounds


def concatenate_arr(x1, x2, x3, bounds1, bounds2, bounds3):
#     if which_w is None:
#         return x0_alpha, bounds_alpha
#     elif which_alpha in ['0', '1']:
#         return x0_w, bounds_w
#     else:
#         #Concatenating both
#         x0 = np.concatenate([x0_alpha, x0_w])
#         assert type(bounds_alpha) == tuple
#         #create bounds = (list_bounds_inf, list_bounds_sup), by combining bounds_alpha and bounds_w
#         if type(bounds_alpha[0]) != list:
#             assert type(bounds_alpha[1]) != list
#             bounds_alpha = ([bounds_alpha[0]] * len(x0_alpha), [bounds_alpha[1]] * len(x0_alpha))
#         else:
#             assert len(bounds_alpha[0]) == len(x0_alpha)
#             assert len(bounds_alpha[1]) == len(x0_alpha)
#         bounds = (bounds_alpha[0] + bounds_w[0], bounds_alpha[1] + bounds_w[1]) #concatenating the bounds
#         return x0, bounds
    x = np.concatenate([x1, x2, x3])
    if type(bounds1[0]) != list: #bounds_alpha
        assert type(bounds1[1]) != list
        bounds1 = ([bounds1[0]] * len(x1), [bounds1[1]] * len(x1))
    else:
        assert len(bounds1[0]) in [0, len(x1)]
        assert len(bounds1[1]) in [0, len(x1)]
    #create bounds = (list_bounds_inf, list_bounds_sup), by combining bounds1, bounds2, bounds3
    #concatenating the bounds
#     bounds = (bounds1[0] + bounds2[0] + bounds3[0], bounds1[1] + bounds2[1] + bounds3[1]) 
    bounds = (np.concatenate([bounds1[0], bounds2[0], bounds3[0]]), np.concatenate([bounds1[1], bounds2[1], bounds3[1]]))
    return x, bounds
    
    
def split_arr(arr, n1, n2, n3):
    assert len(arr) == n1 + n2 + n3
    x1, x2, x3 = arr[:n1], arr[n1: n1 + n2], arr[n1 + n2:]
    x1 = x1 if len(x1) > 0 else None
    x2 = x2 if len(x2) > 0 else None
    x3 = x3 if len(x3) > 0 else None
    return x1, x2, x3
    
        
def from_vec_to_obj(vec_alpha_w, graph, 
                    which_CI, which_alpha='directed', which_w=None, which_Mext=None,
                    k_max=None, parallel_CI=False):
    """
    Returns the object alpha (containing information about alpha / dict_alpha_impaired / K_nodes, K_edges), based on vec_alpha_w (= vec_alpha and vec_w)
    
    We need K_nodes and K_edges for CIpower (because it determines the weights of the graph and M_ext when class Network is called by run_algo)
    """
#     print("len(vec_alpha_w) = {}".format(len(vec_alpha_w)))
    
    assert xor(k_max is None, which_CI == 'CIbeliefs') #k_max is provided for and only for which_CI == 'CIbeliefs'
    assert not(which_w is None and which_alpha in ['0', '1'] and which_Mext is None) #otherwise there is nothing to fit
    
    assert which_w in ['undirected', 'directed', None]
    if which_w == 'undirected':
        n_w_vec = len(graph.edges)
    elif which_w == 'directed':
        n_w_vec = len(graph.edges) * 2 #= len(get_all_oriented_edges(graph))
    elif which_w is None:
        n_w_vec = 0
        
    assert which_Mext in ['nodal', None]
    if which_Mext == 'nodal':
        n_w_input_vec = len(graph.nodes)
    elif which_Mext is None:
        n_w_input_vec = 0
    
    n_alpha_vec = len(vec_alpha_w) - n_w_vec - n_w_input_vec #instead of looking at which_alpha
    
    vec_alpha, vec_w, vec_w_input = split_arr(vec_alpha_w, n_alpha_vec, n_w_vec, n_w_input_vec)
#     print("n_alpha_vec = {}, n_w_vec = {}, n_w_input_vec = {}".format(n_alpha_vec, n_w_vec, n_w_input_vec))
#     def len_with_none(arr):
#         if arr is None:
#             return 0
#         else:
#             return len(arr)
#     print("len(vec_alpha) = {}, len(vec_w) = {}, len(vec_w_input) = {}".format(len_with_none(vec_alpha), len_with_none(vec_w), len_with_none(vec_w_input)))
        
#     if which_w is None:
#         vec_alpha = vec_alpha_w
#     elif which_alpha in ['0', '1']:
#         vec_w = vec_alpha_w
#     else:
#         assert which_w in ['undirected', 'directed']
#         if which_w == 'undirected':
#             n_w_vec = len(graph.edges)
#         elif which_w == 'directed':
#             n_w_vec = len(graph.edges) * 2 #= len(get_all_oriented_edges(graph))
#         vec_alpha, vec_w = vec_alpha_w[:-n_w_vec], vec_alpha_w[-n_w_vec:]
                
#     print("len(vec_alpha) = {}, len(vec_w) = {}".format(len(vec_alpha), len(vec_w)))
          
    if which_alpha not in ['0', '1']:
        #Initialize the variables
        dict_alpha_impaired = None
        K_nodes = None
        K_edges = None
        alpha = None

        if which_CI != 'CIbeliefs':
            if which_alpha == 'directed': #default
                assert 'power' not in which_CI #CIpower with unconstrained alpha is impossible - see CInew instead (no K_i and K_ij but only alpha_ij which can be unconstrained) #{alpha_ij} unconstrained doesn't work, because we need K_i and K_ij to modify M_ext and w_ij . So M_ij = 1/alpha_ji . F(B_i - alpha_ij . M_ji) does not work. (it is not CIpower. However, it corresponds to CInew!!). Or maybe we could take Kedges_{i \to j} in CIpower?
                #However, with CInew, alpha_ij can be unconstrained, because there are no changes of M_ext (resp w_ij) depending of K_nodes (resp K_edges): the alpha_ij only appears in the message update equation: M_ij = 1/alpha_ji . F(B_i - alpha_ij . M_ji) 
                dict_alpha_impaired = dict(zip(get_all_oriented_edges(graph), vec_alpha))
            elif which_alpha == 'undirected':
                K_edges = dict(zip(list(graph.edges), 1 / vec_alpha)) #careful: 1 / alpha (before: vec_alpha)
            elif which_alpha == 'nodal': #alpha_{i \to j} = alpha_i
                K_nodes = dict(zip(list(graph.nodes), vec_alpha))
            elif which_alpha == 'nodal_out': #alpha_{i \to j} = alpha_j
                assert which_CI not in ['CIpower', 'CIpower_approx', 'CInew'] #Impossible to have which_alpha = nodal_out: alpha_ij = K_i / K_ij. Maybe introduce a new model with alpha_ij = K_j / K_ij but K_j is still used for M_ext and K_ij is still used for the transformation of weights
                raise NotImplemented #For which_CI not in ['CIpower', 'CInew']. To implement it that, add a field K_nodes_out to Alpha_obj
    #             corresp = dict(zip(list(graph.nodes), vec_alpha))
    #             dict_alpha_impaired = {(node1, node2): corresp[node2] 
    #                                    for (node1, node2) in get_all_oriented_edges(graph)}
            elif which_alpha == 'uniform':
                assert len(vec_alpha) == 1
                if which_CI not in ['CIpower', 'CIpower_approx', 'CInew']:
                    alpha = vec_alpha[0] #vec_alpha[0] because vec_alpha is a numpy array with one element so we get a float, to be coherent with all other cases
                else:
                    #CIpower and CInew cannot deal with alpha uniform without knowing K_nodes and K_edges. Here we set the uniform case to be uniform K_nodes equal to alpha. On the line below you can find a alternative
                    K_nodes = dict(zip(list(graph.nodes), repeat(vec_alpha[0]))) #K_ij = 1 and alpha_{i to j} = K_i = alpha
    #                 K_edges = dict(zip(list(graph.edges), repeat(1 / vec_alpha[0]))) #K_i = 1 and alpha_{i to j} = 1 / K_ij = alpha  #this is an alternative to what's above
            elif which_alpha == 'directed_ratio':
                K_nodes = dict(zip(list(graph.nodes), vec_alpha[:len(graph.nodes)]))
                K_edges = dict(zip(list(graph.edges), 1 / vec_alpha[len(graph.nodes):])) #changed - before: vec_alpha[len(graph.nodes):]
            elif which_alpha == '1_directed_ratio':
                assert which_CI != 'CI'
                K_edges = dict(zip(get_all_oriented_edges(graph), 1 / vec_alpha))
            elif which_alpha == 'directed_ratio_directed':
                assert which_CI != 'CI'
                K_nodes = dict(zip(list(graph.nodes), vec_alpha[:len(graph.nodes)]))
                K_edges = dict(zip(get_all_oriented_edges(graph), 1 / vec_alpha[len(graph.nodes):]))
            else:
                print("(which_alpha = {}, which_CI = {}) is not supported in function from_vec_to_obj".format(which_alpha, which_CI))
                sys.exit()

        elif which_CI == 'CIbeliefs':
            if which_alpha == 'nodal_temporal':
    #             length_alpha = int(len(vec_alpha) / len(graph.nodes)) #we recovered k_max - 1 from vec_alpha (but we cannot do that for 'nodal' or 'uniform')
                # print(length_alpha)
                vec_alpha_all_nodes = list(vec_alpha.reshape(len(graph.nodes), k_max - 1))
                dict_alpha_impaired = dict(zip(list(graph.nodes), vec_alpha_all_nodes))
    #             # print("vec_alpha_optim", vec_alpha_optim)
    #             # print("of len = ", len(vec_alpha_optim))
            elif which_alpha == 'temporal': #alpha_{j, k} = alpha_k, i.e., alpha[j] are all identical (but are vectors still)
                dict_alpha_impaired = dict(zip(list(graph.nodes), repeat(vec_alpha)))
            elif which_alpha == 'nodal': #alpha_{j, k} = alpha_j, i.e., alpha[j] = [alpha_j, alpha_j, alpha_j, ...etc]
                dict_alpha_impaired = dict(zip(list(graph.nodes), np.tile(vec_alpha, (k_max - 1, 1)).T))
            elif which_alpha == 'uniform':
                dict_alpha_impaired = dict(zip(list(graph.nodes), vec_alpha * np.ones((len(list(graph.nodes)), k_max - 1))))
            else:
                print("(which_alpha = {}, which_CI = {}) is not supported in function from_vec_to_obj".format(which_alpha, which_CI))
                sys.exit()
            print("before", vec_alpha)
            print("after", dict_alpha_impaired)

    #     dict_alpha = create_alpha_dict(graph, 
    #                                    alpha=alpha)#maybe do it only in the cases where dict_alpha_impaired has not been computed? It could make the code less efficient
    #     return dict_alpha
        if parallel_CI == False:
            alpha_obj = Alpha_obj(
                {'alpha': alpha, 
                 'dict_alpha_impaired': dict_alpha_impaired,
                 'K_nodes': K_nodes, 'K_edges': K_edges
                })
        else: #parallel_CI = True
            if dict_alpha_impaired is not None:
                assert which_CI != 'CIbeliefs'
                alpha_matrix = from_dict_to_matrix(dict_alpha_impaired, list(graph.nodes))
                alpha_obj = Alpha_obj({'alpha_matrix': alpha_matrix})
            else:
                if K_nodes is not None:
                    K_nodes_vector = from_dict_to_vector(K_nodes, list(graph.nodes), default_value=1)
                else:
                    K_nodes_vector = None
                if K_edges is not None:
                    make_symmetrical = which_alpha not in ['1_directed_ratio', 'directed_ratio_directed']
                    K_edges_matrix = from_dict_to_matrix(K_edges, list(graph.nodes), default_value=1, 
                                                         make_symmetrical=make_symmetrical)
                else:
                    K_edges_matrix = None
                alpha_obj = Alpha_obj(
                    {'alpha': alpha,
                     'K_nodes_vector': K_nodes_vector, 'K_edges_matrix': K_edges_matrix
                    })
    else:
        if which_CI == 'BP':
            alpha_obj = Alpha_obj({'alpha': 1}) #BP --> by definition, alpha = 1  (note that (which_CI='BP',which_alpha='0') means indeed BP, with alpha=1!
        else:
            alpha_obj = Alpha_obj({'alpha': float(which_alpha)}) #for which_alpha = '1' or '0', for instance
    
    assert which_w in ['undirected', 'directed', None]
    if which_w == 'undirected':
#         print("vec_w = {}".format(vec_w))
        dict_w = dict(zip(list(graph.edges), vec_w))
        if parallel_CI == False:
            w_obj = Alpha_obj({'K_edges': dict_w}) #K_edges
        else:
            w_matrix = from_dict_to_matrix(dict_w, list(graph.nodes), default_value=0.5, make_symmetrical=True)
            w_obj = Alpha_obj({'alpha_matrix': w_matrix}) #'K_edges_matrix'
    elif which_w == 'directed':
        dict_w = dict(zip(get_all_oriented_edges(graph), vec_w))
        if parallel_CI == False:
            w_obj = Alpha_obj({'dict_alpha_impaired': dict_w})
        else:
            w_matrix = from_dict_to_matrix(dict_w, list(graph.nodes), default_value=0.5)
            w_obj = Alpha_obj({'alpha_matrix': w_matrix})
    elif which_w is None:
        w_obj = None
        
    assert which_Mext in ['nodal', None]
    if which_Mext == 'nodal':
        if parallel_CI == False:
            dict_w_input = dict(zip(list(graph.nodes), vec_w_input))
            w_input = dict_w_input
        else:
            w_input = vec_w_input
    elif which_Mext is None:
        w_input = None

    return alpha_obj, w_obj, w_input #some of them can be None
        

def diff_CI_true(vec_alpha_w, graph, M_ext, p_1_true, 
                 which_CI='CI', which_alpha='directed', which_w=None, which_Mext=None,
                 damping=0, k_max=None,
                 parallel_CI=False, parallel_Mext=False,
                 which_distance='diff'
                ):
    """
    Function used in the least square minimization
    Returns the vector p_estimate(X=1) - p(X=1), with a penalization if the model didn't converge
    
    vec_alpha has size:
    - 2*N_edges for CI, where N_edges is the number of unoriented edges of graph
    - N_edges + N_nodes for CIpower (because alpha_ij = K_i / K_ij)
    - N_nodes*(k_max-1) for CIbeliefs
    
    The output of the function has size N_nodes
    
    Using the function:
    #     diff = diff_CI_true(vec_alpha_w, graph, M_ext, p_1_true) #diff between p(X=1) and p_estimate(X=1)
    #     print("max abs diff (CI vs true):", np.max(np.abs(diff)))
    # #     error_CI = 0.5 * np.sum(y**2)
    # #     print("error_CI = {}".format(error_CI))
    #     p_1_CI = diff + np.array(list(p_1_true.values()))
    #     print("p_1_CI:", p_1_CI)
    """
    assert which_distance in ['diff', 'KL']
    alpha, w, w_input = from_vec_to_obj(vec_alpha_w, graph, 
                                        which_CI, which_alpha, which_w, which_Mext,
                                        k_max, 
                                        parallel_CI=parallel_CI)
    res = simulate_CI(graph, M_ext, 
                      alpha=alpha, w=w, w_input=w_input,
                      which_CI=which_CI, damping=damping,
                      transform_into_dict=True,
                      parallel_CI=parallel_CI, parallel_Mext=parallel_Mext
                     )
#     res = simulate_CI_with_history(graph, M_ext, alpha=alpha, dict_alpha_impaired=dict_alpha_impaired, which_CI=which_CI, damping=damping) #the history is needed to test convergence #be careful with this: indeed, the result might differ consequently from simulate_CI, because of the discrepancies in the indices (+ the random perturbations to the input, which might lead to convergence to different fixed points between functions simulate_CI_with_history and simulate_CI...)
    # print(res.B_CI)
    #test the convergence and return a high value if there is no convergence (here a vector of 1: the sign does not matter)
#     if test_convergence_dict(res.B_history_CI) == False:
#         penalisation = 1.5
#     else:
#         penalisation = 1
#     multiplication_factor = {key: 1 + 2*np.abs(val[-1] - val[-2]) for key,val in res.B_history_CI.items()}
#     assert list(res.B_CI.keys()) == list(multiplication_factor.keys())
#     multiplication_factor = np.array(list(multiplication_factor.values()))
#         return np.array([1]*len(p_1_true)) #try if this helps to fit to a alpha for which there is convergence. Alternative: have a penalization (otherwise the cost function has the same value most of the time)
    
    if parallel_Mext == False:
        assert list(p_1_true.keys()) == list(res.B_CI.keys())
#         print("p_1_true.keys()", p_1_true.keys())
        p_1_CI = sig(np.array(list(res.B_CI.values())))
        return p_1_CI - np.array(list(p_1_true.values())) #no penalization
#         return (p_1_CI - np.array(list(p_1_true.values()))) * multiplication_factor
#         return (p_1_CI - np.array(list(p_1_true.values()))) * penalisation
    
    else: #p_1_true is [{node:p_node for all nodes} for all examples] / B_CI is {node: [B_node for all examples] for all nodes}
        assert list(res.B_CI[0].keys()) == list(p_1_true[0].keys())
        if which_distance == 'diff':
            p_1_CI = [{node: sig(val) for node, val in B_CI_ex.items()} for B_CI_ex in res.B_CI] #p_1_CI is a list of dict: p_1_CI = [{node: p_1_CI_node for all nodes} for all examples] (just like p_1_true)
            diff = []
            for p_1_true_ex, p_1_CI_ex in zip(p_1_true, p_1_CI):
                diff.append({node: p_1_true_ex[node] - p_1_CI_ex[node] for node in graph.nodes})
            return [np.array(list(el.values())) for el in diff] #return diff
        elif which_distance == 'KL':
            #KL-divergence between the marginal distributions: KL(p_true_i(x_i)||p_approx_i(x_i))
            #where KL(p_true_i(x_i)||p_approx_i(x_i)) = sum_{x_i} p_true_i(x_i) . log(p_true_i(x_i) / p_approx_i(x_i))
            p_1_true_matrix = np.array([list(el.values()) for el in p_1_true])
            p_1_approx_matrix = sig(np.array([list(el.values()) for el in res.B_CI]))
#             print(p_1_true_matrix.shape, p_1_approx_matrix.shape, KL_divergence(p_1_true_matrix, p_1_approx_matrix).shape) #all are of size (n_examples, n_nodes)
            return KL_divergence(p_1_true_matrix, p_1_approx_matrix) #of size (n_examples, n_nodes)
        else:
            raise NotImplemented
        
        
def diff_CI_true_list(vec_alpha_w, graph, X_train, y_train, 
                      which_CI='CI', which_alpha='directed', which_w=None, which_Mext=None,
                      damping=0, k_max=None,
                      parallel_CI=True, parallel_Mext=True, which_distance='diff'):
    """
    Returns a vector
    
    Using the function: 
    # diff = diff_CI_true_list(res.x, graph, X, y)
    # error_CI = 0.5 * np.sum(diff**2)
    # vec_alpha_BP = np.array([1] * len(list(graph.edges)) * 2)
    # diff = diff_CI_true_list(vec_alpha_BP, graph, X, y, parallel_CI=False)
    # error_BP = 0.5 * np.sum(diff**2)
    """
#     alpha, w, w_input = from_vec_to_obj(vec_alpha_w, graph, 
#                                         which_CI, which_alpha, which_w, which_Mext,
#                                         k_max, 
#                                         parallel_CI=parallel_CI)
#     print("alpha = {}".format(alpha))
#     print("w = {}".format(w))
#     print("w_input = {}".format(w_input))
    
    assert which_distance in ['diff', 'KL']
    X_train, y_train = from_df_to_list_of_dict(X_train, y_train)
    if parallel_CI == False:
        y_all = []
        for i in range(len(X_train)):
            M_ext = X_train[i]
            p_1_true = y_train[i]
            y = diff_CI_true(vec_alpha_w, graph, M_ext, p_1_true, which_CI=which_CI, 
                             which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext,
                             damping=damping, k_max=k_max,
                             parallel_CI=parallel_CI, parallel_Mext=parallel_Mext
                            )
            y_all.append(y)
        return np.array(y_all).reshape((-1,))
    else: #parallel_CI = True
        if parallel_Mext == False: 
            with Pool(n_cpus) as p:
                y_all = p.starmap(diff_CI_true, 
                                  zip(repeat(vec_alpha_w), repeat(graph), X_train, y_train, repeat(which_CI),
                                      repeat(which_alpha), repeat(which_w), repeat(which_Mext),
                                      repeat(damping), repeat(k_max),
                                      repeat(parallel_CI), repeat(parallel_Mext)
                                     )
                                 )
#             print("y_all.shape = {}".format(np.array(y_all).shape))
            return np.array(y_all).reshape((-1,))
        else:
            y_all = diff_CI_true(vec_alpha_w, graph, X_train, y_train, which_CI,
                                 which_alpha, which_w, which_Mext,
                                 damping, k_max, parallel_CI=parallel_CI, parallel_Mext=parallel_Mext,
                                 which_distance=which_distance)
#             print("y_all.shape = {}".format(np.array(y_all).shape))
            y_all = np.array(y_all).reshape((-1,))
            return y_all
#             sys.exit()
  
    
def get_final_messages_list(vec_alpha_w, graph, list_M_ext, 
                            which_CI='CI', which_alpha='directed', which_w=None, which_Mext=None,
                            damping=0, k_max=None,
                            parallel_CI=True, parallel_Mext=True):
#     print("vec_alpha_w = {}".format(vec_alpha_w))
    list_M_ext = from_df_to_list_of_dict(list_M_ext)
    if parallel_CI == True:
        if parallel_Mext == False:
            with Pool(n_cpus) as p:
                alpha, w = from_vec_to_obj(vec_alpha_w, graph, which_CI, which_alpha, which_w, which_Mext,
                                           k_max, parallel_CI=parallel_CI) #should this line be inside the Pool or outside?
                M_final_all = p.starmap(get_final_messages,
                                        zip(repeat(graph), list_M_ext, repeat(alpha), repeat(w), repeat(which_CI), 
                                            repeat(damping),
                                            repeat(parallel_CI), repeat(parallel_Mext))
                                       )
        #     print("alpha = {}".format(alpha.to_matrix(graph)))
            return np.array(M_final_all).reshape((-1,))
        else:
            alpha, w = from_vec_to_obj(vec_alpha_w, graph, which_CI, which_alpha, which_w, which_Mext,
                                                   k_max, parallel_CI=parallel_CI)
            M_final_all = get_final_messages(graph, list_M_ext, alpha, w, which_CI, 
                                             damping=damping, parallel_CI=parallel_CI, parallel_Mext=parallel_Mext)
            return M_final_all.reshape((-1,))
    else:
        raise NotImplemented #TODO: implement the non-parallel version (not urgent)

        
def get_final_messages(graph, M_ext, 
                       alpha, w=None, w_input=None, which_CI='CI', 
                       damping=0,
                       parallel_CI=True, parallel_Mext=True):
    res = simulate_CI(graph, M_ext, alpha=alpha, w=w, w_input=w_input,
                      which_CI=which_CI, return_final_M=True, damping=damping, transform_into_dict=True,
                      parallel_CI=parallel_CI, parallel_Mext=parallel_Mext
                     )
    M_final = res.final_M
    return np.array(list(M_final.values()))

    
def learning_unsupervised(graph, X_train, 
                          which_CI, which_alpha='directed', which_w=None, which_Mext=None,
                          damping=0, eps=None, keep_history_learning=True,
                          X_val=None, y_val=None, stopping_criterion='stable_validation_loss'
                         ):
    """
    Uses a learning rule to minimize E = sum_ij M_ij^2 (or sum_ij (B_i - alpha_ij M_ji)^2)
    The learning rule is not exact, but we hope to get nice results anyway (--> check)
    
    Previous name for X_train: list_M_ext
    
    eps is the learning rate
    Minimizes E = sum_{ij} Mfinal_{ij}^2 (after convergence of CI) over {alpha_ij} where M_ij = F_ij(B_i - alpha_ij.Mji)
    --> dE/d(alpha_ij) = -2*M_ij*F'_ij(B_i-alpha_ij*M_ji)*M_ji
    --> dalpha_ij = - eps*dE/dalpha_ij (without the factor 2 from above) to make E decrease
    
    Note that another function to minimize could be E = sum_t sum_{ij} dM_{ij}^2  ---> in this case, 
    """
    assert which_CI in ['BP', 'CI', 'CIpower', 'CIpower_approx'] #TODO: write code for all other cases
    assert which_Mext is None #TODO: think of a formula to fit \hat{\gamma}, i.e. the multiplicative factor in front of M_ext
    if which_CI == 'CI':
        assert which_alpha in ['uniform', 'nodal', 'undirected', 'directed'] #TODO (?): implement directed_ratio
    if 'power' in which_CI:
        assert which_alpha in ['nodal', 'undirected', 'directed_ratio', '1_directed_ratio', 'directed_ratio_directed'] #TODO: implement uniform
    
    class Object(object):
        pass
    res = Object()
    
    if which_CI == 'BP':
        res.alpha = Alpha_obj({'alpha': 1})
        return res
    
    N = len(X_train)
    if eps is None:
        eps_edge = 0.03
        eps_nodal = 0.003 #0.0003
#         eps = 3 * 1/N
#     list_examples = X_train #TODO: shuffle + copy examples to have several presentations
#     list_examples = X_train*5 #duplicate the training set (without shuffling)
#     list_examples = list(chain.from_iterable([random.sample(X_train, N) for _ in range(5)])) #duplicate the training set (with shuffling = sampling N times with replacement) - presenting 5 times each example; 1 block contains each example exactly one time (but shuffled inside the block)
#     list_examples = [X_train]*5000
    list_examples = [random.sample(X_train, 50) for _ in range(2000)] #doing batch learning
    parallel_CI = (which_CI in ['BP', 'CI', 'CIpower', 'CIpower_approx', 'CInew']) and ('weight' in graph.edges[list(graph.edges)[0]].keys())
    
    if parallel_CI == True:
        w_matrix = get_w_matrix(graph)
        W_matrix = 2 * get_w_matrix(graph) - 1
    
    #initialize dict_alpha
    if which_alpha == 'directed':
        dict_alpha = {edge: np.random.uniform(0.8, 1.2) for edge in get_all_oriented_edges(graph)} #both (i,j) and (j,i)
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'dict_alpha_impaired': dict_alpha})
        else:
            alpha_matrix = from_dict_to_matrix(dict_alpha, list(graph.nodes))
            alpha_obj = Alpha_obj({'alpha_matrix': alpha_matrix})
    elif which_alpha == 'uniform':
        alpha = np.random.uniform(0.8, 1.2)
        alpha_obj = Alpha_obj({'alpha': alpha})
    elif which_alpha == 'undirected':
        K_edges = {edge: np.random.uniform(0.8, 1.2) for edge in graph.edges} #undirected edges (i,j)
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'K_edges': K_edges})
        else:
            K_edges_matrix = from_dict_to_matrix(K_edges, list(graph.nodes), make_symmetrical=True)
            alpha_obj = Alpha_obj({'K_edges_matrix': K_edges_matrix})
    elif which_alpha == 'nodal':
        K_nodes = {node: np.random.uniform(0.8, 1.2) for node in graph.nodes} #alpha is associated to a node: alpha_ij = alpha_i
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'K_nodes': K_nodes})
        else:
            K_nodes_vector = from_dict_to_vector(K_nodes, list(graph.nodes))
            alpha_obj = Alpha_obj({'K_nodes_vector': K_nodes_vector})
    elif which_alpha == 'directed_ratio':
        #choosing (K_edges, K_nodes) = (1, 1)  (never done until now)
#         K_edges = dict(zip(list(graph.edges), repeat(1)))
#         K_nodes = dict(zip(list(graph.nodes), repeat(1)))
        
        #choosing (K_edges, K_nodes) randomly between 0.8 and 1.2
#         K_edges = {edge: np.random.uniform(0.8, 1.2) for edge in graph.edges} #undirected edges (i,j)
#         K_nodes = {node: np.random.uniform(0.8, 1.2) for node in graph.nodes} #node i
        
        #choosing (K_edges, K_nodes) such that the algo converges with K_edges = K_nodes  (file 'supervised2')
        K = get_Knodes_Kedges_val_convergence_algo(graph, which_CI)
        print("initial K = {} (K_nodes=K_edges=K)".format(K))
        K_edges = {edge: K for edge in graph.edges} #undirected edges (i,j)
        K_nodes = {node: K for node in graph.nodes} #node i
            
#         #choosing (K_edges=1, K_nodes) such that the algo converges (file 'supervised3')
#         K = get_Knodes_val_convergence_algo(graph, which_CI)
#         print("initial K = {} (K_nodes=K and K_edges=1)".format(K))
#         K_edges = {edge: K for edge in graph.edges} #undirected edges (i,j)
#         K_nodes = {node: 1 for node in graph.nodes} #node i
        
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'K_nodes': K_nodes, 'K_edges': K_edges}) #alpha_ij = K_nodes[i] / K_edges[i,j]
        else:
            K_edges_matrix = from_dict_to_matrix(K_edges, list(graph.nodes), make_symmetrical=True)
            K_nodes_vector = from_dict_to_vector(K_nodes, list(graph.nodes))
            alpha_obj = Alpha_obj({'K_nodes_vector': K_nodes_vector, 'K_edges_matrix': K_edges_matrix})
    elif which_alpha == '1_directed_ratio':
        assert which_CI != 'CI'
        K_edges = {edge: np.random.uniform(0.8, 1.2) for edge in get_all_oriented_edges(graph)} #directed edges (i,j)
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'K_edges': K_edges}) #alpha_ij = 1 / K_edges[i,j]
        else:
            make_symmetrical = which_alpha not in ['1_directed_ratio', 'directed_ratio_directed']
            K_edges_matrix = from_dict_to_matrix(K_edges, list(graph.nodes), make_symmetrical=make_symmetrical)
            alpha_obj = Alpha_obj({'K_edges_matrix': K_edges_matrix})
    elif which_alpha == 'directed_ratio_directed':
        assert which_CI != 'CI'
        K_edges = {edge: np.random.uniform(0.8, 1.2) for edge in get_all_oriented_edges(graph)} #directed edges (i,j)
        K_nodes = {node: np.random.uniform(0.8, 1.2) for node in graph.nodes} #node i
        if parallel_CI == False:
            alpha_obj = Alpha_obj({'K_nodes': K_nodes, 'K_edges': K_edges}) #alpha_ij = K_nodes[i] / K_edges[i,j] where K_edges is not necessarily symmetrical
        else:
            make_symmetrical = which_alpha not in ['1_directed_ratio', 'directed_ratio_directed']
            K_edges_matrix = from_dict_to_matrix(K_edges, list(graph.nodes), make_symmetrical=make_symmetrical)
            K_nodes_vector = from_dict_to_vector(K_nodes, list(graph.nodes))
            alpha_obj = Alpha_obj({'K_nodes_vector': K_nodes_vector, 'K_edges_matrix': K_edges_matrix})
    elif which_alpha == 'nodal_out':
        raise NotImplemented 
    else:
        raise NotImplemented
        
    #initialize w
    assert which_w in [None, 'undirected', 'directed'] #also add low-rank?
    if which_w is None: #which_prod_w
        w = None #i.e. taking weights from graph
#         w = {(i,j): graph.edges[i,j]['weight'] if (i,j) in graph.edges.keys() else graph.edges[j,i]['weight'] for (i,j) in get_all_oriented_edges(graph)} #all weights (only if graph has weights and not factors)
    elif which_w == 'undirected':
        dict_w = {edge: np.random.uniform(0.2, 0.8) for edge in graph.edges} #undirected edges (i,j)
        if parallel_CI == False:
            w = Alpha_obj({'K_edges': dict_w}) #before: K_edges
        else:
            w_matrix = from_dict_to_matrix(dict_w, list(graph.nodes), default_value=0.5, make_symmetrical=True)
            w = Alpha_obj({'alpha_matrix': w_matrix}) #'K_edges_matrix'
    elif which_w == 'directed':
        dict_w = {edge: np.random.uniform(0.2, 0.8) for edge in get_all_oriented_edges(graph)} #both (i,j) and (j,i) #maybe instead take the real weights from graph as a starting point (although if I get interesting results, then it wouldn't be as powerful compared to if I took random weights initially...)?
        if parallel_CI == False:
            w = Alpha_obj({'dict_alpha_impaired': dict_w})
        else:
            w_matrix = from_dict_to_matrix(dict_w, list(graph.nodes), default_value=0.5)
            w = Alpha_obj({'alpha_matrix': w_matrix})
        
    print("which_CI = {}, which_alpha = {}, which_w = {}, which_Mext = {}".format(which_CI, which_alpha, which_w, which_Mext))
    
    #initialize w_input
    assert which_Mext is None
    w_input = None
    
    if keep_history_learning:
        alpha_history = [deepcopy(alpha_obj)] #.copy() is not defined for the Alpha_obj class. Deepcopy is necessary because for instance in the case of dict_alpha_impaired, the shallow copy will not provide saved dictionnaries (in alpha_history) to have their fields changed when the learning takes place
    
    if (X_val is not None) and (y_val is not None) and (stopping_criterion == 'stable_validation_loss'):
        list_error_val = []
        
    #learn {alpha_ij} with gradient descent
    for i_example, M_ext in enumerate(list_examples):
#         print("##############################################")
#         print("len(M_ext) = {}".format(len(M_ext)))
        print("len(X_train) = {}".format(len(X_train)))
        print("type(M_ext) = {}".format(type(M_ext)))
        ################ Inspecting the numerical problems #######
#         print("alpha = {}".format(alpha_obj))
#         K_edges_matrix = alpha_obj.get('K_edges_matrix')
#         print("min(K_edges) = {}, max(K_edges) = {}".format(np.min(K_edges_matrix), np.max(K_edges_matrix)))
#         print("min(1/K_edges) = {}, max(1/K_edges) = {}".format(np.min(1/K_edges_matrix), np.max(1/K_edges_matrix)))
        ##########################################################
        
#         start_time = time.time()
        parallel_Mext = (type(M_ext) == list) #batch learning requires parallel_Mext = True
        res = simulate_CI(graph, M_ext, alpha=alpha_obj, w=w, w_input=w_input,
                          which_CI=which_CI, return_final_M=True, damping=damping, 
                          parallel_CI=parallel_CI, parallel_Mext=parallel_Mext) 
#         end_time = time.time()
#         print("time elapsed: {}".format(end_time - start_time))
        M_final = res.final_M
        B_final = res.B_CI #exists because simulate_CI has save_last=True
#         #check that B is indeed the sum of messages
#         print("M_final", M_final)
        print("M_final.shape", M_final.shape)
#         print("B_final", B_final)
        print("B_final.shape", B_final.shape)
#         print("M_ext", res.M_ext)
#         print("edges", list(res.graph.edges))
#         print("sum_M", {i: [M_final[edge] for edge in M_final.keys() if edge[1]==i] for i in res.graph.nodes})
#         print("sum_M", {i: np.sum([M_final[edge] for edge in M_final.keys() if edge[1]==i]) for i in res.graph.nodes})
#         sys.exit()
#         print(list(graph.nodes))

        assert which_w is None #i.e., not fitting w (--> TODO: propose a formula for the update of w)
        if which_CI == 'CI':
            if which_alpha == 'directed':
                if parallel_CI == False: #dict_alpha is a dict, so are B and M
                    dict_alpha = alpha_obj.get('dict_alpha_impaired')
                    for (i,j) in dict_alpha.keys():
                        x_inside = B_final[i] - dict_alpha[i,j] * M_final[j,i]
            #             dict_alpha[i,j] += eps * M_final[i,j] * dF_w(x_inside, w[i,j]) * M_final[j,i] #proportional to - dE/dalpha_ij
                        dict_alpha[i,j] += eps * x_inside * M_final[j,i] #modified version (but with the same sign as - dE/dalpha_ij)
                    alpha_obj = Alpha_obj({'dict_alpha_impaired': dict_alpha})
                else: #parallel CI --> everything is arrays (B, M, alpha)
                    alpha_matrix = alpha_obj.get('alpha_matrix')
        #                 print("alpha_matrix = {}".format(alpha_matrix))
        #                 print(B_final, np.sum(M_final, axis=0) + np.array(list(res.M_ext.values())))
        #                 print(B_final.reshape((-1,1)).shape, alpha_matrix.shape)
                    alpha_matrix += eps * (B_final.reshape((-1,1)) - alpha_matrix * M_final.T) * M_final.T #Sophie: alpha = alpha + eps * M' * (B - alpha*M')
                    alpha_obj = Alpha_obj({'alpha_matrix': alpha_matrix})

            elif which_alpha == 'uniform':
                if parallel_CI == False:
                    alpha = alpha_obj.get('alpha')
                    alpha_new = alpha.copy() #shallow copy is fine here (one level)
                    for (i,j) in dict_alpha.keys():
                        x_inside = B_final[i] - alpha * M_final[j,i]
                        alpha_new += eps * x_inside * M_final[j,i] #E = sum_ij (B_i - alpha_ij M_ji)^2
#                         alpha_new += eps * M_final[i,j] * dF_w(x_inside, w[i,j]) * M_final[j,i] #E = sum_ij M_ji^2
                    alpha = alpha_new
                    alpha_obj = Alpha_obj({'alpha': alpha})
                else:
                    raise NotImplemented
                    
            elif which_alpha == 'undirected':
                if parallel_CI == False:
                    K_edges = alpha_obj.get('K_edges')
                    K_edges_new = K_edges.copy() #shallow copy is fine here (one level)
                    for (i,j) in M_final.keys():
                        edge_key = (i,j) if (i,j) in K_edges.keys() else (j,i)
                        K_edge = K_edges[edge_key]
                        x_inside = B_final[i] - 1 / K_edge * M_final[j,i]
                        K_edges_new[edge_key] -= eps * x_inside * M_final[j,i] #E = sum_ij (B_i - alpha_ij M_ji)^2
#                         K_edges_new[edge_key] -= eps * M_final[i,j] * dF_w(x_inside, w[i,j]) * M_final[j,i] #E = sum_ij M_ji^2
                    K_edges = K_edges_new
                    alpha_obj = Alpha_obj({'K_edges': K_edges})
                else:
                    raise NotImplemented
                    
            elif which_alpha == 'directed_ratio':
                if parallel_CI == False:
                    K_edges = alpha_obj.get('K_edges')
                    K_edges_new = K_edges.copy() #shallow copy is fine here (one level)
                    K_nodes = alpha_obj.get('K_nodes')
                    K_nodes_new = K_nodes.copy() #shallow copy is fine here (one level)
                    for (i,j) in M_final.keys():
                        edge_key = (i,j) if (i,j) in K_edges.keys() else (j,i)
                        K_edge = K_edges[edge_key]
                        x_inside = B_final[i] - K_nodes[i] / K_edge * M_final[j,i]
                        K_edges_new[edge_key] -= eps * x_inside * M_final[j,i] * K_nodes[i] #E = sum_ij (B_i - alpha_ij M_ji)^2
#                         K_edges_new[edge_key] -= eps * M_final[i,j] * dF_w(x_inside, w[i,j]) * M_final[j,i] * K_nodes[i] #E = sum_ij M_ji^2
                        K_nodes_new[i] += eps * x_inside * M_final[j,i] / K_edge #E = sum_ij (B_i - alpha_ij M_ji)^2
#                         K_nodes_new[i] += eps * M_final[i,j] * dF_w(x_inside, w[i,j]) * M_final[j,i] / K_edge #E = sum_ij M_ji^2
                    K_edges = K_edges_new
                    K_nodes = K_nodes_new
                    alpha_obj = Alpha_obj({'K_edges': K_edges, 'K_nodes': K_nodes})
                else:
                    raise NotImplemented

            elif which_alpha == 'nodal':
                if parallel_CI == False:
                    K_nodes = alpha_obj.get('K_nodes')
                    K_nodes_new = K_nodes.copy() #shallow copy is fine here (one level)
                    for (i,j) in M_final.keys():
                        K_nodes_new[i] += eps * (B_final[i]-K_nodes[i]*M_final[j,i]) * M_final[j,i] #E = sum_ij (B_i - alpha_ij M_ji)^2
                    K_nodes = K_nodes_new
                    alpha_obj = Alpha_obj({'K_nodes': K_nodes})
                else:
                    raise NotImplemented
                
            else:
                raise NotImplemented
           
        elif which_CI in ['CIpower', 'CIpower_approx']:
            assert parallel_CI == True
            
            #TODO: use matrices (even for K_ij... but how?)
#             raise NotImplemented #TODO: finish coding
                
            #Note that J_tilde = J / K = arctanh(2*w-1) / K_edges is used here  (maybe use 2*w_tilde-1 ?)
            #(F_{w_tilde}(x) with w_tilde transformed  is like F_{J_tilde}(x) with J_tilde = J / K where J = arctanh(2w-1))
            #w is transformed according to: w_tilde / (1-w_tilde) = (w/(1-w))^{1/K_edges}
            
            if which_alpha == 'directed_ratio':
#                 print("M_ext should be a vector")
#                 M_ext = np.array(list(M_ext.values()))
                M_ext = pd.DataFrame(M_ext).to_numpy().T
                print("M_ext.shape = {}".format(M_ext.shape))
                print("TODO: finish coding: the formulas should be using matrices with one additional dimension (number of examples)")
                sys.exit()
#                 print("M_ext = {}".format(M_ext))
                K_edges_matrix = alpha_obj.get('K_edges_matrix')
                K_nodes_vector = alpha_obj.get('K_nodes_vector')
                Mtilde_final = K_nodes_vector[...,np.newaxis].T * M_final #Mtilde_ij = K_j * M_ij such that B_i = 1/K_i*[sum_j M_ji + winput_i*M_{ext,i}]
#                 print("K_nodes_vector.shape = {}, K_edges_matrix.shape = {}, M_final.shape = {}, B_final.shape = {}".format(K_nodes_vector.shape, K_edges_matrix.shape, M_final.shape, B_final.shape))
                delta_alpha = eps_edge * Mtilde_final.T * (B_final[...,np.newaxis] - 1 / K_edges_matrix * Mtilde_final.T) #eps_edge * Mtilde_final.T * (B_final[...,np.newaxis] - 1 / K_edges_matrix * Mtilde_final.T) + eps_edge * Mtilde_final * (B_final[...,np.newaxis].T - 1 / K_edges_matrix * Mtilde_final) 
                delta_alpha = delta_alpha + delta_alpha.T #make the update symmetrical (because alpha_ij = alpha_ji)
                delta_kappa = - eps_nodal * M_ext * (B_final - M_ext)  #old (proposed by Sophie, arbitrary rule)
#                 delta_kappa = eps_nodal * B_final * (B_final - M_ext)  #old (proposed by Vincent, alternative arbitrary rule) --> it should work as well (in theory) as the arbitrary rule proposed by Sophie
#                 delta_kappa = - eps_nodal * K_nodes_vector * B_final * np.sum(B_final[...,np.newaxis] - 1 / K_edges_matrix * Mtilde_final.T, axis=1) #new (proposed by me, comes directly from the minimization of sum_{ij} [B_i - alpha_{ij}.M_{j->i}]^2) --> all kappa converge towards 0...
#                 delta_kappa = + eps_nodal * K_nodes_vector * B_final * np.sum(B_final[...,np.newaxis] - 1 / K_edges_matrix * Mtilde_final.T, axis=1) #new (proposed by me, comes directly from the maximization of sum_{ij} [B_i - alpha_{ij}.M_{j->i}]^2) --> explosion of beliefs
                
#                 print("min(M_final) = {}, max(M_final) = {}".format(np.min(M_final), np.max(M_final)))
#                 print("min(B_final - K_nodes_vector / K_edges_matrix * M_final.T) = {}, max(B_final - K_nodes_vector / K_edges_matrix * M_final.T) = {}".format(np.min(B_final - K_nodes_vector[...,np.newaxis] / K_edges_matrix * M_final.T), np.max(B_final - K_nodes_vector / K_edges_matrix * M_final.T)))
#                 print("min(delta_alpha) = {}, max(delta_alpha) = {}".format(np.min(delta_alpha), np.max(delta_alpha)))
                #normalize delta_alpha
                if np.max(np.abs(delta_alpha)) > 2:
                    delta_alpha = delta_alpha / np.max(np.abs(delta_alpha)) * 2 #prevents the update to be too violent (numerical problems if |alpha| is too high)
                alpha_new = 1 / K_edges_matrix + delta_alpha #alpha_ij = 1 / beta_ij (where beta_ij = Kedges_ij)
                #change the alpha so that alpha=1/K_edges is not too close to 0 (otherwise there are some numerical problems)
#                 print("min(alpha_new) = {}, max(alpha_new) = {}".format(np.min(alpha_new), np.max(alpha_new)))
#                 print("min(1/alpha_new) = {}, max(1/alpha_new) = {}".format(np.min(1/alpha_new), np.max(1/alpha_new)))
#                 alpha_new[1 / alpha_new > 20] = 1/20 #100
#                 alpha_new[1 / alpha_new < -20] = -1/20 #100
#                 print("min(alpha_new) = {}, max(alpha_new) = {}".format(np.min(alpha_new), np.max(alpha_new)))
#                 print("min(1/alpha_new) = {}, max(1/alpha_new) = {}".format(np.min(1/alpha_new), np.max(1/alpha_new)))
                kappa_new = 1 / K_nodes_vector + delta_kappa #kappa_i = 1 / gamma_i (where gamma_i = Knodes_i)
                if np.sum(np.isnan(alpha_new)) != 0: 
                    print("some alpha_new are nan!")
                    print("M_final = {}".format(M_final))
                    print("B_final = {}".format(B_final))
                    print("alpha_new = {}".format(alpha_new))
                    sys.exit()
                if np.sum(np.isnan(kappa_new)) != 0:
                    print("some kappa_new are nan!")
                    print("M_final = {}".format(M_final))
                    print("B_final = {}".format(B_final))
                    print("kappa_new = {}".format(kappa_new))
                    sys.exit()
                alpha_obj = Alpha_obj({'K_edges_matrix': 1 / alpha_new, 'K_nodes_vector': 1 / kappa_new})
#                 print("alpha_obj = {}".format(alpha_obj))
                if i_example % 1000 == 0:
                    print("i_example = {} (out of {})".format(i_example+1, len(list_examples)))
                    print("alpha_new = {}".format(alpha_new))
                    print("kappa_new = {}".format(kappa_new))

            elif which_alpha == '1_directed_ratio':
                raise NotImplemented #TODO: implement

            elif which_alpha == 'directed_ratio_directed':
                raise NotImplemented #TODO: implement
        else:
            raise NotImplemented
        
        if keep_history_learning:
            alpha_history.append(deepcopy(alpha_obj)) #shallow copy would be enough in the uniform case(copy(alpha) or alpha.copy()) as the object has only one level for which shallow copy is fine. But in all other cases, it doesn't work well and deepcopy is needed (2 levels). I checked in an example that indeed shallow copy didn't work in the case of dictionnaries inside the object
            
        #stopping criterion
        if (X_val is not None) and (y_val is not None): #and i_example % 10 == 0
            assert stopping_criterion == 'stable_validation_loss'
            model = Model_alpha_ij_all(graph, 
                                       which_CI=which_CI, which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext,
                                       damping=damping)
            y_predict = model.predict(X_val, alpha=alpha_obj, w=w, w_input=w_input)
            error_val = accuracy_score(y_val, y_predict, which_error='MSE')
            list_error_val.append(error_val)
            percent_satur = 0.05 #0.5
            n_going_back = 500
            if i_example >= 500 and is_saturating(list_error_val, percent=percent_satur, back=n_going_back):
                print("Early stopping - after {} epochs".format(i_example))
                break
        
    print("alpha_new = {}".format(alpha_new))
    print("kappa_new = {}".format(kappa_new))
        
    #save the history of the learning
    res.alpha_history = alpha_history
    
    #plot the evolution of the loss during learning
    if (X_val is not None) and (y_val is not None):
        plt.plot(list_error_val)
        plt.xlabel('iteration of the algorithm')
        plt.ylabel('error on the validation set')
        plt.show()
        
    #plot the evolution of alpha during learning
    if keep_history_learning:
        if which_alpha == 'directed':
            if parallel_CI == False:
                list_dict_alpha = [el.get('dict_alpha_impaired') for el in alpha_history]
                list_dict_alpha_modif = {edge: [el[edge] for el in list_dict_alpha] for edge in list_dict_alpha[0].keys()}
            else:
                list_alpha_matrix = [el.get('alpha_matrix') for el in alpha_history]
                list_dict_alpha_modif = {edge: [el[edge] for el in [from_matrix_to_dict(mat, list(graph.nodes)) 
                                                                    for mat in list_alpha_matrix]]
                                         for edge in get_all_oriented_edges(graph)}
            for key, val in list_dict_alpha_modif.items():
                plt.plot(val, label=key)
        elif which_alpha == 'uniform':
            list_alpha = [el.get('alpha') for el in alpha_history]
            plt.plot(list_alpha)
        elif which_alpha == 'undirected':
            list_K_edges = [el.get('K_edges_matrix') for el in alpha_history]
            list_K_edges_modif = {edge: [el[edge] for el in list_K_edges] for edge in range(len(list_K_edges[0]))} #in list_K_edges[0].keys()
            for key, val in list_K_edges_modif.items():
                plt.plot(val, label=key)
        elif which_alpha == 'nodal':
            list_K_nodes = [el.get('K_nodes_vector') for el in alpha_history] #K_nodes
            list_K_nodes_modif = {node: [el[node] for el in list_K_nodes] for node in range(len(list_K_nodes[0]))} #in list_K_nodes[0].keys()
            for key, val in list_K_nodes_modif.items():
                plt.plot(val, label=key)
        elif which_alpha == 'directed_ratio':
            list_K_nodes = [el.get('K_nodes_vector') for el in alpha_history]
            print(np.array(list_K_nodes).shape)
            list_K_nodes_modif = {node: [el[node] for el in list_K_nodes] for node in range(len(list_K_nodes[0]))} #list_K_nodes[0].keys()
            for key, val in list_K_nodes_modif.items():
                plt.plot(1/np.array(val), label=key)
            plt.title('kappa = 1/gamma = 1/K_nodes', size=15)
            plt.xlabel('training iteration', size=15)
            plt.ylabel('kappa', size=15)
            plt.show()
            list_K_edges = [el.get('K_edges_matrix') for el in alpha_history]
            print(np.array(list_K_edges).shape)
            list_K_edges_modif = {edge: [el[edge] for el in list_K_edges] for edge in range(len(list_K_edges[0]))} #list_K_edges[0].keys()
            for key, val in list_K_edges_modif.items():
                plt.plot(1/np.array(val), label=key)
            plt.title('alpha = 1/K_edges', size=15)
            plt.xlabel('training iteration', size=15)
            plt.ylabel('alpha', size=15)
            plt.show()
        elif which_alpha == '1_directed_ratio':
            raise NotImplemented #TODO: implement

        elif which_alpha == 'directed_ratio_directed':
            raise NotImplemented #TODO: implement
        else:
            raise NotImplemented
        if which_alpha != 'uniform':
            plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
        if which_alpha != 'directed_ratio':
            plt.xlabel('training iteration', size=15)
            plt.ylabel('alpha', size=15)
            plt.show()
        
    #Fitted {alpha_ij} (final)
    res.alpha = alpha_obj
#     if which_alpha == 'directed':
#         if parallel_CI == False:
#             res.alpha = Alpha_obj({'dict_alpha_impaired': dict_alpha})
#         else:
#             res.alpha = Alpha_obj({'alpha_matrix': alpha_matrix})
#     elif which_alpha == 'uniform':
#         res.alpha = Alpha_obj({'alpha': alpha})
#     elif which_alpha == 'undirected':
#         res.alpha = Alpha_obj({'K_edges': K_edges})
#     elif which_alpha == 'nodal':
#         res.alpha = Alpha_obj({'K_nodes': K_nodes})
#     elif which_alpha == 'directed_ratio':
#         res.alpha = alpha_obj #Alpha_obj({'K_edges_matrix': 1 / alpha_new, 'K_nodes_vector': 1 / kappa_new})
# #         res.alpha = Alpha_obj({'K_nodes_vector': K_nodes_vector, 'K_edges_matrix': K_edges_matrix})
#     elif which_alpha == '1_directed_ratio':
#         raise NotImplemented #TODO: implement
#     elif which_alpha == 'directed_ratio_directed':
#         raise NotImplemented #TODO: implement
#     else:
#         raise NotImplemented #why isn't directed_ratio implemented??
        
    #Fitting {w_ij} (final) - Do nothing if which_w = None (i.e., {w_ij} is not fitted)
#     if which_w == 'undirected':
#         if parallel_CI == False:
#             res.w = Alpha_obj({'K_edges': dict_w})
#         else:
#             res.w = Alpha_obj({'alpha_matrix': w_matrix})
#     elif which_w == 'directed':
#         if parallel_CI == False:
#             res.w = Alpha_obj({'dict_alpha_impaired': dict_w})
#         else:
#             res.w = Alpha_obj({'alpha_matrix': w_matrix})
    res.w = w
    
    #Fitting {winput_ij} (final) - Do nothing if which_w = None (i.e., {w_ij} is not fitted)
    res.w_input = w_input
    
    return res
        
    
class Model_alpha_ij_all:
    """
    Main class, to create a model
    """
    
    def __init__(self, graph, 
                 which_CI, which_alpha='directed', which_w=None, which_Mext=None,
                 damping=0, k_max=None):
        
#         print("which_alpha here = {}".format(which_alpha))
        assert xor(which_CI == 'CIbeliefs', k_max is None) #k_max is provided for and only for CIbeliefs
        assert which_alpha in ['directed', 'undirected', 'nodal', 'nodal_out', 'uniform', 
                               'directed_ratio', '1_directed_ratio', 'directed_ratio_directed',
                               'nodal_temporal', 'temporal', '1', '0', None]
        
        self.which_CI = which_CI
        self.graph = graph
        self.damping = damping
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        
#         if self.which_CI == 'CIbeliefs':
#             self.k_max = k_max
        self.k_max = k_max
        
#     def get_params(self): #trying to use sklearn code, but for now giving up
#         return {'k_max': self.k_max}
    
#     def set_params(self, k_max=None):
#         if k_max is None:
#             print("pb")
#             sys.exit()
#         self.k_max = k_max
        
    def fit(self, X_train, y_train=None, 
            method_fitting='supervised_MSE_scipy', 
            which_CI=None, which_alpha=None, which_w=None, which_Mext=None, #supposed to be given already in init (but they can be changed here)
            options=None,
            parallel_CI=None, parallel_Mext=None,
            print_res=False, X_test=None, y_test=None
           ):
        """
        Fitting the alpha_ij (for many examples X_train = M_ext) for CI to be close to exact inference 
        (by minimizing at the error between the marginal of CI and the one of exact inference)
        
        y_train is needed only for supervised learning (not for unsupervised)
        eps is the learning rate for unsupervised learning (with learning rule) --> in variable options (dict)
        
        Variable 'which_alpha' can be:
        1) For which_CI != 'CIbeliefs' (--> alpha_{i \to j} associated to a directed edge):
        - directed       : alpha_{i \to j}, without constraints
        - undirected     : alpha_{i \to j} = alpha_{ij}, meaning alpha_{i \to j} = alpha_{j \to i}
        - nodal          : alpha_{i \to j} = alpha_i  (OR ALPHA_J ???)
        - nodal_out      : alpha_{i \to j} = alpha_j
        - uniform        : alpha_{i \to j} = alpha
        - directed_ratio : alpha_{i \to j} = K_i / K_{ij} where K_{ij} is symmetrical
        - 1_directed_ratio : alpha_{i \to j} = 1 / K_{ij} where K_{ij} has no constraints
        - directed_ratio_directed : alpha_{i \to j} = K_i / K_{ij} where K_{ij} has no constraints
        2) For which_CI == 'CIbeliefs' (--> alpha_{i, k} associated to node j and time t-k)
        - nodal_temporal : alpha_{j, k}, without constraints
        - temporal       : alpha_{j, k} = alpha_k
        - nodal          : alpha_{j, k} = alpha_j
        - uniform        : alpha_{j, k} = alpha
        
        #diff_CI_true_list has a penalization if the model doesn't converge
        # x0 = np.array([1] * len(list(graph.edges)) * 2)
        # res = least_squares(diff_CI_true_list, x0, bounds=(0, 2), args=(graph, X, y), verbose=1)
        # # print(res)
        # vec_alpha_optim = res.x
        # alpha_optim = dict(zip(get_all_oriented_edges(graph), vec_alpha_optim))
        # print("dict_alpha_impaired (found for this M_ext): {}".format(alpha_optim))

        X_test and y_test are needed for the stopping criterion 'stable_validation_loss' of Pytorch
        """
        if which_CI is None:
            which_CI = self.which_CI
        if which_alpha is None:
            which_alpha = self.which_alpha
        if which_w is None: #careful: the default value of w is None, and it's one of the options for which_w... (maybe change in the options None into something else, e.g. "not"? Otherwise there could be pbs if we want w=None and self.w!=None)
            which_w = self.which_w
        if which_Mext is None: #careful: the default value of w is None, and it's one of the options for which_w... (maybe change in the options None into something else, e.g. "not"? Otherwise there could be pbs if we want w=None and self.w!=None)
            which_Mext = self.which_Mext
        assert which_w in [None, 'directed', 'undirected']
        assert which_Mext in [None, 'nodal']
#         assert not(which_alpha is None and which_CI != 'BP')
        if which_alpha is None:
            assert which_CI == 'BP'
            which_alpha = '1'
        assert which_alpha in ['directed', 'undirected', 'nodal', 'nodal_out', 'uniform', 
                               'directed_ratio', '1_directed_ratio', 'directed_ratio_directed',
                               'nodal_temporal', 'temporal', '1', '0']#, None] #if '1' or '0', then alpha is not fitted (useful in order to fit w_ij)
        assert not(which_CI == 'CIbeliefs' 
                   and which_alpha not in ['nodal_temporal', 'temporal', 'nodal', 'uniform'])
        assert not(which_CI != 'CIbeliefs' and which_alpha not in ['directed', 'undirected', 'nodal', 'nodal_out', 'uniform', 'directed_ratio', '1_directed_ratio', 'directed_ratio_directed', '1', '0'])
        assert not(which_alpha == '0' and which_CI in ['CInew', 'CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh']) #because they have 1/alpha in their formula
        assert not(which_CI in ['full_CIpower_approx', 'full_CIpower_approx_approx_tanh', 'rate_network'] and which_alpha not in ['nodal', 'uniform', '1']) #which_alpha='1' corresponds to the absence of subtraction but K_nodes=1 #The rest would be ok but only if I decide a convention for the formula with beta_ij, i.e., where to have beta_ji --> for now CIpower is defined as M_ij = beta_ji / gamma_j * arctanh( tanh(J_ij / beta_ij) * tanh(B_i - gamma_i / beta_ij * M_ji) ), but maybe I should define it with beta_ij / gamma_j instead?
#         assert not(which_CI == 'BP' and which_alpha in ['0'])
        #default value for which_alpha
#         if hasattr(self, 'which_alpha'):
#             which_alpha = self.which_alpha #careful: overwrites the argument which_alpha given into fit_CI
#         if which_alpha is None:
#             if which_CI != 'CIbeliefs':
#                 which_alpha = 'directed'
#             else:
#                 which_alpha = 'nodal_temporal'
        assert method_fitting in ['supervised_MSE_scipy', 'supervised_KL_scipy', 'supervised_KL_pytorch', 'supervised_CE_pytorch', 'supervised_MSE_pytorch', 'unsupervised', 'unsupervised_learning_rule'] #TODO: implement other methods (lowest real eigenvalue, correlations, etc.)
#         assert xor(eps is None, 'unsupervised_learning_rule' in method_fitting) #eps provided for and only for unsupervised fitting methods with learning rule
#         assert xor(keep_history_learning == False, method_fitting == 'unsupervised_learning_rule')
        assert xor(y_train is None, method_fitting in ['supervised_MSE_scipy', 'supervised_KL_scipy', 'supervised_KL_pytorch', 'supervised_CE_pytorch', 'supervised_MSE_pytorch']) #y_train provided for and only for supervised methods (only case when it's needed for fitting)
        
        #default values for parallel_CI and parallel_Mext
        if parallel_CI is None:
            parallel_CI = (which_CI in 
                           ['BP', 'CI', 
                            'CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh',
                            'CIpower_approx', 'CIpower_approx_approx_tanh', 'full_CIpower_approx', 'full_CIpower_approx_approx_tanh', 
                            'CInew', 'CIapprox_tanh', 'CIapprox_linear', 'rate_network']) and ('weight' in self.graph.edges[list(self.graph.edges)[0]].keys())
        if parallel_Mext is None:
            parallel_Mext = parallel_CI
        assert not(parallel_Mext == True and parallel_CI == False), "Change to parallel_Mext = False"
        
        #particular case: BP ---> no fitting
        if (which_CI == 'BP' or which_alpha == '1') and (which_w is None) and (which_Mext is None):
            self.alpha = Alpha_obj({'alpha': 1})
            self.w = None
            self.w_input = None
            return
        
#         print("which_CI = {}, which_alpha = {}, which_w = {}, which_Mext = {}".format(which_CI, which_alpha, which_w, which_Mext))
        
        #full CI ---> no fitting
        if which_alpha == '0' and (which_w is None) and (which_Mext is None):
            self.alpha = Alpha_obj({'alpha': 0})
            self.w = None
            self.w_input = None
            return
            
        #particular case: CIbeliefs with k_max = 0 ---> no fitting
        if which_CI == 'CIbeliefs' and self.k_max == 0 and which_w is None and which_Mext is None:
            dict_alpha = dict(zip(list(self.graph.nodes), 
                                  list(np.array([0] * len(list(self.graph.nodes))).reshape((-1,1)))
                                 ))
            self.alpha = Alpha_obj({'dict_alpha_impaired': dict_alpha})
            self.w = None
            self.w_input = None
            return
        
        if method_fitting in ['supervised_MSE_scipy', 'supervised_KL_scipy']: #supervised learning: y_train is given
            X_train, y_train = from_df_to_list_of_dict(X_train, y_train)
        elif 'pytorch' not in method_fitting: #--> y_train is not given
            X_train = from_df_to_list_of_dict(X_train)
        
        if 'pytorch' in method_fitting:
            assert X_test is not None and y_test is not None
            if (options is not None) and ('eps' in options.keys()):
                eps = options['eps']
            else:
                assert False
                #eps = 1e-3 #this value is a bit random
            if (options is not None) and ('max_nepochs' in options.keys()):
                max_nepochs = options['max_nepochs']
            else:
                max_nepochs = 150
#             print("which_loss = {}".format(which_loss))
#             print("options = {}".format(options))
            which_loss = method_fitting.replace('supervised_', '').replace('_pytorch', '')
            if (options is not None) and ('keep_history_learning' in options.keys()):
                keep_history_learning = options['keep_history_learning']
                if keep_history_learning:
                    raise NotImplemented #it doesn't appear in the function fit_pytorch
            else:
                assert False
            assert keep_history_learning == False
            res = fit_pytorch(self.graph, X_train, y_train,
                              which_CI, which_alpha, which_w, which_Mext,
                              damping=self.damping, k_max=self.k_max,
                              X_test_df=X_test, y_test_df=y_test,
                              learning_rate=eps, algo='Rprop', which_loss=which_loss, max_nepochs=max_nepochs,
                              stopping_criterion='stable_validation_loss'
                             )#eps, keep_history_learning
            #what's below is impossible to plot here, because we are inside a starmap --> actually it gets plotted
            if hasattr(res, 'no_convergence'):
#                 print("inside fit function - trying to plot")
                loss_all_epochs = res.loss_all_epochs
                loss_test_all_epochs = res.loss_test_all_epochs
                #plot the loss
#                 if which_loss == 'MSE':
#                     print("loss_all_epochs[:5] (training loss) = {}".format(loss_all_epochs[:5]))
#                     plt.plot(np.log(loss_all_epochs), label='train')
#                     plt.plot(np.log(loss_test_all_epochs), label='val') #val and not test
#                     plt.title("log(loss) - try #1") #(but shouldn't work)"
#                 else:
#                     print("loss_all_epochs[:5] (training loss) = {}".format(loss_all_epochs[:5]))
#                     plt.plot(loss_all_epochs, label='train')
#                     plt.plot(loss_test_all_epochs, label='val') #val and not test
#                     plt.title("loss - try #1") #(but shouldn't work)"
#                 plt.legend()
#                 plt.xlabel("iteration")
#                 plt.show()
                
            alpha, w, w_input = res.alpha_obj, res.w_obj, res.w_input
#             print("alpha", alpha)
#             print("w", w)
#             print("w_input", w_input)
        
        if method_fitting in ['supervised_MSE_scipy', 'supervised_KL_scipy']:
            x0, bounds = starting_point_least_squares(self.graph, which_CI, which_alpha, which_w, which_Mext)
            max_eval_fun = 150
#             print("len(x0) = {}".format(len(x0)))
#             print("x0 = {}".format(x0))
#             y0 = diff_CI_true_list(x0, self.graph, X_train, y_train, which_CI, 
#                                    which_alpha, which_w, which_Mext,
#                                    self.damping, self.k_max,
#                                    parallel_CI, parallel_Mext)
#             print(np.sum(y0**2))
            if method_fitting == 'supervised_MSE_scipy': #minimization of the mean-square error between p_1_true and p_1_approx
                which_distance = 'diff'
                res = least_squares(diff_CI_true_list, x0, bounds=bounds, 
                                    args=(self.graph, X_train, y_train, 
                                          which_CI, which_alpha, which_w, which_Mext,
                                          self.damping, self.k_max,
                                          parallel_CI, parallel_Mext, which_distance),
                                    max_nfev=max_eval_fun, #to prevent the optimization from being too long
                                    verbose=1) #make least_squares faster: look at https://stackoverflow.com/questions/61907318/speed-up-least-square-scipy
            elif method_fitting == 'supervised_KL_scipy': #minimization of the KL-divergence between p_true and p_approx
                #y_all is the KL, for all examples and all nodes, of KL(p_i(x_i) || \hat{p}_i(x_i))
                #in order to use least-squares (if direct minimization, then compute the sum or the mean of elements of y
                which_distance = 'KL'
                print("why sqrt_loss??")
                res = least_squares(diff_CI_true_list, x0, bounds=bounds, 
                                    args=(self.graph, X_train, y_train,
                                          which_CI, which_alpha, which_w, which_Mext,
                                          self.damping, self.k_max,
                                          parallel_CI, parallel_Mext, which_distance),
                                    loss=sqrt_loss, #I should probably use standard minimization then? (minimization of a value) - but maybe least-square is convenient here because it supposes that the elements of the vector are smooth functions of the input. Besides, the approach worked for error, so maybe it means that the distance between p_true and p_approx is indeed a smooth function of {alpha_ij}?   If least-squares doesn't work or takes too much time, look instead at alternatives in scipy (there are plenty)
                                    max_nfev=max_eval_fun, #to prevent the optimization from being too long
                                    verbose=1)
            if res.nfev == max_eval_fun:
                print("The fitting stopped automatically after {} evaluations (possibly no convergence)".format(max_eval_fun))
            vec_alpha_optim = res.x
            if print_res:
                print(res)
            alpha, w, w_input = from_vec_to_obj(vec_alpha_optim, self.graph, which_CI, which_alpha, which_w, which_Mext,
                                                parallel_CI=parallel_CI)
        
        elif method_fitting == 'unsupervised':
            x0, bounds = starting_point_least_squares(self.graph, which_CI, which_alpha, which_w, which_Mext)
            max_eval_fun = 150
#             print("x0 = {}, bounds = {}".format(x0, bounds))
#             y0 = get_final_messages_list(x0, self.graph, X_train, 
#                                          which_CI, which_alpha, which_w, which_Mext, #w
#                                          self.damping, self.k_max,
#                                          parallel_CI, parallel_Mext)
#             print("y0")
#             print(y0.shape)
#             print(y0)
            res = least_squares(get_final_messages_list, x0, bounds=bounds, 
                                args=(self.graph, X_train, 
                                      which_CI, which_alpha, which_w, which_Mext, #w
                                      self.damping, self.k_max,
                                      parallel_CI, parallel_Mext),
                                max_nfev=max_eval_fun, #to prevent the optimization from being too long
                                verbose=1)
            if res.nfev == max_eval_fun:
                print("The fitting stopped automatically after {} evaluations (possibly no convergence)".format(max_eval_fun))
            vec_alpha_optim = res.x
            if print_res:
                print(res)
            alpha, w, w_input = from_vec_to_obj(vec_alpha_optim, self.graph, which_CI, which_alpha, which_w, which_Mext,
                                                parallel_CI=parallel_CI)
        
        elif method_fitting == 'unsupervised_learning_rule':
            #Tries to minimize the same quantity as the unsupervised learning: E = sum_{ij} Mfinal_{ij}^2
            if 'eps' in options.keys():
                eps = options['eps']
            else:
                assert False
            if (options is not None) and ('keep_history_learning' in options.keys()):
                keep_history_learning = options['keep_history_learning']
            else:
                assert False, "Provide keep_history_learning in variable options"
            res = learning_unsupervised(self.graph, X_train, 
                                        which_CI, which_alpha, which_w, which_Mext,
                                        self.damping,
                                        eps=eps, 
                                        keep_history_learning=keep_history_learning, 
                                        X_val=X_test, y_val=y_test)
            alpha = res.alpha
            if hasattr(res, 'alpha_history'): #if keep_history_learning:
                alpha_history = res.alpha_history
                self.alpha_history = alpha_history
            assert which_w is None
            assert which_Mext is None
            w = None
            w_input = None
            
        self.alpha = alpha
#         print('alpha', self.alpha)
        self.w = w
        self.w_input = w_input
        
    def predict_one(self, M_ext, 
                    alpha=None, w=None, w_input=None,
                    which_CI=None, keep_history_beliefs=False,
                    parallel_CI=True, parallel_Mext=True, log=False
                   ):
        """
        This function is only called by function predict
        
        It deals with one example M_ext
        """
        if keep_history_beliefs == False:
            res = simulate_CI(self.graph, M_ext, 
                              alpha=alpha, w=w, w_input=w_input,
                              which_CI=which_CI, damping=self.damping, 
                              parallel_CI=parallel_CI, parallel_Mext=parallel_Mext)
            #belief at the last timestep
            if log == False:
                if type(res.B_CI) == dict:
                    p_1_CI = {node: sig(val) for node, val in res.B_CI.items()}
                else: #np array (parallel implementation of CI)
                    p_1_CI = dict(zip(list(res.graph.nodes), sig(res.B_CI))) #sig(res.B_CI)
            else:
                if type(res.B_CI) == dict:
                    p_1_CI = {node: val for node, val in res.B_CI.items()}
                else: #np array (parallel implementation of CI)
                    p_1_CI = dict(zip(list(res.graph.nodes), res.B_CI))
        else:
            res = simulate_CI_with_history(self.graph, M_ext, 
                                           alpha=alpha, w=w, w_input=w_input,
                                           which_CI=which_CI, damping=self.damping,
                                           parallel_CI=parallel_CI, parallel_Mext=parallel_Mext)
            #belief history
            if log == False:
                if type(res.B_CI) == dict: #why not type(res.B_history_CI)?
                    p_1_CI = {node: sig(val) for node, val in res.B_history_CI.items()}
                else: #np array (parallel implementation of CI)
                    p_1_CI = dict(zip(list(res.graph.nodes), list(sig(np.array(res.B_history_CI).T))))
            else:
                if type(res.B_CI) == dict: #why not type(res.B_history_CI)?
                    p_1_CI = {node: val for node, val in res.B_history_CI.items()}
                else: #np array (parallel implementation of CI)
                    print("Returning logs!!! (here it's ok I think)")
                    p_1_CI = dict(zip(list(res.graph.nodes), list(np.array(res.B_history_CI).T)))
#             print("shape with history", p_1_CI[list(p_1_CI.keys())[0]].shape)
#             p_1_CI_final = {node: sig(res.B_CI[node]) for node in res.B_CI} #belief at the last timestep
#             print("shape without history", p_1_CI_final[list(p_1_CI_final.keys())[0]].shape)
        return p_1_CI
    
    def predict_all(self, list_M_ext, alpha=None, w=None, w_input=None,
                    which_CI=None, keep_history_beliefs=False, parallel_CI=True, log=False):
        """
        This function is only called by function predict
        
        It deals with many examples M_ext
        """
        parallel_Mext = True
        assert parallel_CI == True #only possible option because parallel_Mext = True
        if keep_history_beliefs == False:
            res = simulate_CI(self.graph, list_M_ext, 
                              alpha=alpha, w=w, w_input=w_input,
                              which_CI=which_CI, damping=self.damping, 
                              parallel_CI=parallel_CI, parallel_Mext=parallel_Mext,
                              transform_into_dict=True
                             )
            if log == False:
                p_1_CI = [{node: sig(val) for node, val in B_CI_ex.items()} for B_CI_ex in res.B_CI]
            else:
                p_1_CI = [{node: val for node, val in B_CI_ex.items()} for B_CI_ex in res.B_CI]
#             #belief at the last timestep
#             if type(res.B_CI[0]) == dict:
#                 p_1_CI = [{node: sig(val) for node, val in B_CI_ex.items()} for B_CI_ex in res.B_CI]
#             else: #np array (parallel implementation of CI)
#                 p_1_CI = [dict(zip(list(res.graph.nodes), sig(B_CI_ex))) for B_CI_ex in res.B_CI] #sig(res.B_CI)
        else:
            res = simulate_CI_with_history(self.graph, list_M_ext, 
                                           alpha=alpha, w=w, w_input=w_input,
                                           which_CI=which_CI, damping=self.damping, 
                                           parallel_CI=parallel_CI, parallel_Mext=parallel_Mext,
                                           transform_into_dict=True
                                          )
#             print(type(res.B_history_CI))
#             print(list(res.B_history_CI.keys()))
#             #print(len(res.B_history_CI))
#             print(type(res.B_history_CI[0]))
#             print(res.B_history_CI[0].shape)
            #belief history
            if log == False:
                if type(res.B_CI[0]) == dict: #why not type(res.B_history_CI)?
                    p_1_CI = {node: sig(val) for node, val in res.B_history_CI.items()}
                else: #np array (parallel implementation of CI)
                    print("Returning logs!!! (probably a mistake")
                    p_1_CI = dict(zip(list(res.graph.nodes), list(transpose(np.array(res.B_history_CI))))) #transpose from utils_CI_BP.py (inverts on the first 2 dimensions)
            else:
                if type(res.B_CI[0]) == dict: #why not type(res.B_history_CI)?
                    p_1_CI = {node: val for node, val in res.B_history_CI.items()}
                else: #np array (parallel implementation of CI)
                    print("Returning logs!!! (here it's ok I think")
                    p_1_CI = dict(zip(list(res.graph.nodes), list(transpose(np.array(res.B_history_CI))))) #transpose from utils_CI_BP.py (inverts on the first 2 dimensions)
#             print("shape with history", p_1_CI[list(p_1_CI.keys())[0]].shape)
#             p_1_CI_final = {node: sig(res.B_CI[node]) for node in res.B_CI} #belief at the last timestep
#             print("shape without history", p_1_CI_final[list(p_1_CI_final.keys())[0]].shape)
        return p_1_CI
    
    def predict(self, list_M_ext, 
                alpha=None, w=None, w_input=None,
                which_CI=None, keep_history_beliefs=False, 
                parallel_CI=None, parallel_Mext=None, log=False):
        #If alpha_ij / which_CI / {w_ij} / {w_input_i} is not provided, then use the ones in self (coming from the fitting)
        if which_CI is None:
            which_CI = self.which_CI
        if alpha is None:
            alpha = self.alpha
        if w is None:
            if hasattr(self, 'w'):
                w = self.w #careful: the default value of w is None, and it's one of the options for which_w... (maybe change in the options None into something else, e.g. "not"? Otherwise there could be pbs if we want w=None and self.w!=None)
            else:
                w = None
        if w_input is None:
            if hasattr(self, 'w_input'):
                w_input = self.w_input
            else:
                w_input = None
              
        if parallel_CI is None:
            parallel_CI = (which_CI in 
                           ['BP', 'CI',
                            'CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh',
                            'CIpower_approx', 'CIpower_approx_approx_tanh', 'full_CIpower_approx', 'full_CIpower_approx_approx_tanh', 
                            'CInew', 'CIapprox_tanh', 'CIapprox_linear', 'rate_network']) and ('weight' in self.graph.edges[list(self.graph.edges)[0]].keys())
        if parallel_Mext is None:
            parallel_Mext = parallel_CI #parallel_Mext can be True only if parallel_CI = True
            if len(list_M_ext) == 1:
                parallel_Mext = False
        list_M_ext = from_df_to_list_of_dict(list_M_ext)
#         fun_predict_one = lambda M_ext: self.predict_one(M_ext, dict_alpha=dict_alpha, which_CI=which_CI) #a lambda function cannot be used with multiprocessing.Pool.map()
#         def fun_predict_one(M_ext):
#             return self.predict_one(M_ext, dict_alpha=dict_alpha, which_CI=which_CI)
        if parallel_Mext == False:
            with Pool(n_cpus) as p:
                list_p_1_CI = p.starmap(self.predict_one, 
                                        zip(list_M_ext, 
                                            repeat(alpha), repeat(w), repeat(w_input),
                                            repeat(which_CI), repeat(keep_history_beliefs), 
                                            repeat(parallel_CI), repeat(parallel_Mext), repeat(log)))
#             list_p_1_CI = Pool(n_cpus).starmap(self.predict_one, zip(list_M_ext, repeat(dict_alpha), repeat(which_CI))) #Pool(n_cpus).map(fun_predict_one, list_M_ext) ---> doesn't work (cannot pickle locally defined functions) #map(fun_predict_one, list_M_ext) #[self.predict_one(M_ext, dict_alpha=dict_alpha, which_CI=which_CI) for M_ext in list_M_ext]
        else:
            list_p_1_CI = self.predict_all(list_M_ext, alpha=alpha, w=w, w_input=w_input, which_CI=which_CI,
                                           keep_history_beliefs=keep_history_beliefs,
                                           parallel_CI=parallel_CI, log=log)
        
        #transforming list of dict into dataframe (if necessary)
        if type(list_M_ext) == pandas.core.frame.DataFrame:
            list_p_1_CI = pd.DataFrame(list_p_1_CI)
        return list_p_1_CI
    
    def predict_CI(self, list_M_ext, 
                   alpha=None, w=None, w_input=None,
                   which_CI='CI', keep_history_beliefs=False, 
                   parallel_CI=None, parallel_Mext=None, log=False):
        """
        CI (message-passing, not linearized or approximated)
        """
        return self.predict(list_M_ext, 
                            alpha=alpha, w=w, w_input=w_input,
                            which_CI=which_CI, keep_history_beliefs=keep_history_beliefs,
                            parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def predict_CIbeliefs(self, list_M_ext, 
                          alpha=None, w=None, w_input=None,
                          which_CI='CIbeliefs', keep_history_beliefs=False, 
                          parallel_CI=None, parallel_Mext=None, log=False):
        """
        CIbeliefs (message-passing)
        """
        return self.predict(list_M_ext, 
                            alpha=alpha, w=w, w_input=w_input,
                            which_CI=which_CI, keep_history_beliefs=keep_history_beliefs,
                            parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def predict_BP(self, list_M_ext, 
                   w=None, w_input=None,
                   keep_history_beliefs=False, 
                   parallel_CI=None, parallel_Mext=None, log=False):
        """
        BP (message-passing, not linearized or approximated)
        """
#         vec_alpha = np.array([1] * len(list(self.graph.edges)) * 2) #BP
#         dict_alpha = dict(zip(get_all_oriented_edges(self.graph), vec_alpha))
#         return self.predict_CI(list_M_ext, dict_alpha, keep_history_beliefs=keep_history_beliefs)
        return self.predict_CI(list_M_ext, alpha=Alpha_obj({'alpha': 1}), w=w, w_input=w_input,
                               keep_history_beliefs=keep_history_beliefs,
                               parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)

    def predict_CI_uniform_alpha(self, list_M_ext, 
                                 alpha_val, w=None, w_input=None,
                                 keep_history_beliefs=False, 
                                 parallel_CI=None, parallel_Mext=None, log=False):
        """
        CI with uniform alpha
        """
#         vec_alpha = np.array([alpha] * len(list(self.graph.edges)) * 2)
#         dict_alpha = dict(zip(get_all_oriented_edges(self.graph), vec_alpha))
#         return self.predict_CI(list_M_ext, dict_alpha, keep_history_beliefs=keep_history_beliefs)
        return self.predict_CI(list_M_ext, 
                               alpha=Alpha_obj({'alpha': alpha_val}), w=w, w_input=w_input,
                               keep_history_beliefs=keep_history_beliefs,
                               parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def predict_full_CI(self, list_M_ext, w=None, w_input=None, keep_history_beliefs=False, 
                        parallel_CI=None, parallel_Mext=None, log=False):
        """
        CI with alpha=0 (message-passing, not linearized or approximated)
        Note that it also corresponds to CIbeliefs with alpha = 0, and to CIbeliefs2 with alpha = 0
        """
#         vec_alpha = np.array([0] * len(list(self.graph.edges)) * 2) #full CI
#         dict_alpha = dict(zip(get_all_oriented_edges(self.graph), vec_alpha))
#         return self.predict_CI(list_M_ext, dict_alpha, keep_history_beliefs=keep_history_beliefs)
        return self.predict_CI(list_M_ext, alpha=0, w=w, w_input=w_input,
                               keep_history_beliefs=keep_history_beliefs,
                               parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def predict_full_CIbeliefs(self, list_M_ext, 
                               w=None, w_input=None,
                               keep_history_beliefs=False, 
                               parallel_CI=None, parallel_Mext=None, log=False):
        """
        CIbeliefs with alpha=0 (also corresponds to CI with alpha=0 - TODO=check)
        """
#         k_max = 2 #self.k_max
#         vec_alpha = np.array([0] * len(list(self.graph.nodes)) * (k_max - 1))
# #         length_alpha = int(len(vec_alpha) / len(self.graph.nodes)) #= k_max - 1
#         vec_alpha_all_nodes = list(vec_alpha.reshape(len(self.graph.nodes), self.k_max - 1)) #length_alpha instead of self.k_max-1 (before)
#         dict_alpha = dict(zip(list(self.graph.nodes), vec_alpha_all_nodes))
#         return self.predict_CIbeliefs(list_M_ext, dict_alpha=dict_alpha, keep_history_beliefs=keep_history_beliefs)
        return self.predict_full_CI(list_M_ext, w=w, w_input=w_input, keep_history_beliefs=keep_history_beliefs,
                                    parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def predict_full_CIbeliefs2(self, list_M_ext, w=None, w_input=None,
                                keep_history_beliefs=False, 
                                parallel_CI=None, parallel_Mext=None, log=False):
        """
        CIbeliefs2 with alpha=0 (also corresponds to CI with alpha=0 - TODO=check)
        """
#         vec_alpha = np.array([0] * len(list(self.graph.edges)) * 2) #full CI
#         dict_alpha = dict(zip(get_all_oriented_edges(self.graph), vec_alpha))
#         return self.predict_CI(list_M_ext, dict_alpha, which_CI='CIbeliefs2', keep_history_beliefs=keep_history_beliefs)
        return self.predict_full_CI(list_M_ext, 
                                    w=w, w_input=w_input, keep_history_beliefs=keep_history_beliefs,
                                    parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)

    def predict_full_CI_or_CIbeliefs(self, list_M_ext, w=None, w_input=None, keep_history_beliefs=False, 
                                     parallel_CI=None, parallel_Mext=None, log=False):
        """
        The model needs to be already fitted (i.e. model.dict_alpha needs to exist)
        """
        return self.predict(X, {key: [0] for key in model.dict_alpha.keys()}, w=w, w_input=w_input, 
                            keep_history_beliefs=keep_history_beliefs,
                            parallel_CI=parallel_CI, parallel_Mext=parallel_Mext, log=log)
    
    def score(self, list_p_1_true, list_p_1_predict, log=False):
        """
        Gives the R2 score
        # print("R2 score (BP) = {}".format(model.score(y, y_predict)))
        """
        assert log == False
        
        list_p_1_true, list_p_1_predict = from_df_to_list_of_dict(list_p_1_true, list_p_1_predict)
        #score (not cost function)
        y_true = np.array([list(el.values()) for el in list_p_1_true])
        y_pred = np.array([list(el.values()) for el in list_p_1_predict])
#         print(y_true.shape, y_pred.shape)
        r2 = r2_score(y_true, y_pred) #, multioutput='raw_values') #raw_values, uniform_average, variance_weighted
        return r2

    
def Model_alpha_ij_CI(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CI', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping)
    
def Model_alpha_ij_CIbeliefs(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0, k_max=None):
    return Model_alpha_ij_all(graph, which_CI='CIbeliefs', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping, k_max=k_max)

def Model_alpha_ij_CIbeliefs2(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CIbeliefs2', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping)

def Model_alpha_ij_CIpower(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CIpower', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping)

def Model_alpha_ij_CIpower_approx(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CIpower_approx', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping)

def Model_alpha_ij_CInew(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CInew', which_alpha=which_alpha, which_w=which_w, 
                              which_Mext=which_Mext, damping=damping)

def Model_alpha_ij_CIapprox(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CIapprox', which_alpha=which_alpha, which_w=which_w,
                              which_Mext=which_Mext, damping=damping)

def Model_alpha_ij_CIapprox_tanh(graph, which_alpha=None, which_w=None, which_Mext=None, damping=0):
    return Model_alpha_ij_all(graph, which_CI='CIapprox_tanh', which_alpha=which_alpha, which_w=which_w,
                              which_Mext=which_Mext, damping=damping)



    
def residual(list_dict1, list_dict2):
    """
    returns the list of differences between the 2 dictionaries
    """
    assert len(list_dict1) == len(list_dict2)
    list_diff_dict = []
    for i in range(len(list_dict1)):
        dict1 = list_dict1[i]
        dict2 = list_dict2[i]
        assert list(dict1.keys()) == list(dict2.keys())
        dict_diff = {key: dict1[key] - dict2[key] for key in dict1.keys()}
        list_diff_dict.append(dict_diff)
    return list_diff_dict


def logsig_scalar(t):
    """
    Stable log-sigmoid function (instead of the unstable one: np.log(1 / (1 + np.exp(-t))))
    See logsig_log1pexp in http://fa.bianp.net/blog/2019/evaluate_logistic/
    
    I tested the function vs np.log(1 / (1 + np.exp(-t)))) --> ok
    """
    if t < -33.3:
        return t
    elif t <= -18:
        return t - np.exp(t)
    elif t <= 37:
        return -np.log1p(np.exp(-t))
    else:
        return -np.exp(-t)

def logsig(t):
    """
    Same as logsig_scalar buy operates on vectors
    
    Stable log-sigmoid function (instead of the unstable one: np.log(1 / (1 + np.exp(-t))))
    See logsig_log1pexp in http://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    y = np.zeros(shape=t.shape)
    y[t < -33.3] = t[t < -33.3]
    y[(t <= -18) * (t >= -33.3)] = (t - np.exp(t))[(t <= -18) * (t >= -33.3)]
    y[(t <= 37) * (t > -18)] = -np.log1p(np.exp(-t[(t <= 37) * (t > -18)]))
    y[t > 37] = -np.exp(-t[t > 37])
    return y


def accuracy_score(y_predict, y_true, which_error=None, list_which_error=None, log=False):
    """
    It is important that the 1st arg is the prediction, and the 2nd arg is the ground truth
    
    Error measure
    Mean square error (MSE)
    (Question: shouldn't it be penalized if the simulation didn't converge, like in diff_CI_true?)
    
    y_true and y_predict are pandas dataframes of size (n_examples, n_nodes)
    Once transformed via from_df_to_list_of_dict, they become a list of dict: [{p_1_node_val for all nodes} for all examples]
    
    # print("MSE (BP) = {}".format(accuracy_score(y_predict, y)))
    
    Both y_predict and y_true are expected to be probabilities (= neither or them should be a log-odds)
    
    But if log = True, it means y_predict is a log-odds
    """
    assert xor(which_error is None, list_which_error is None)
    
    #list_which_error is given
    if list_which_error is not None:
        assert type(list_which_error) == list
        d_error = {}
        for which_error in list_which_error:
            d_error[which_error] = accuracy_score(y_predict, y_true, which_error=which_error, log=log)
        return d_error
        
    #which_error is given
    if which_error == 'cross_entropy':
        print("which_error = 'cross_entropy' is deprecated - use 'CE' instead")
        which_error = 'CE'
    assert which_error in ['MSE', 'MSE_all', 'KL', 'CE'], "which_error = {} is not implemented".format(which_error) #other methods are not implemented
    
    if 'MSE' in which_error:
        y_true, y_predict = from_df_to_list_of_dict(y_true, y_predict)
        if log == True:
            y_predict = sig(y_predict) #transforming from log-odds to p(X=1)
        diff_CI_true = residual(y_predict, y_true) #list of dict
        diff = np.array([np.array(list(el.values())) for el in diff_CI_true])
    #     print(diff)
#         print(diff.shape)
        if which_error == 'MSE':
            MSE_CI = np.mean(diff**2)  #now I return the real MSE = mean of the squared errors, i.e., by dividing by both the number of nodes and the number of examples
            #There any division in least_squares function by scipy, except by 2 --> strange (not 2 which is something classic, but that there are no other divisions)
#             MSE_CI = 1 / diff.shape[0] * np.sum(diff**2) #0.5 * np.sum(diff**2) #0.5*np.sum(diff**2) is used in least_squares #I don't use this formula anymore because I didn't divide by the number of
#             print("MSE = {}".format(MSE_CI))
#             print("error in least-squares = {}".format(0.5*np.sum(diff**2)))
        elif which_error == 'MSE_all':
            #give one MSE for each example (but sum the error over nodes of a given graph)
            MSE_CI = np.sum(diff**2, axis=1)
        else:
            raise NotImplemented
        return MSE_CI
    #     return np.mean((y - y_true)**2)

    elif which_error == 'KL': #returns KL(p_true || p_predict)
        #By definition, KL(b||p) = sum_x [b(x) . log(b(x) / p(x))]
        #mean of the KL divergence between the factorized distributions (mean over the nodes and over examples)
        y_true, y_predict = from_list_of_dict_to_df(y_true, y_predict)
        y_true = y_true.to_numpy()
        y_predict = y_predict.to_numpy()
        if log == False:
            return np.mean(y_true * np.log(y_true / y_predict) + (1-y_true) * np.log((1-y_true) / (1-y_predict))) #KL shown on Fig 3b of Pitkow et al 2018
#             return np.mean(y_predict * np.log(y_predict / y_true) + (1-y_predict) * np.log((1-y_predict) / (1-y_true))) #KL which would seem more logical to me, because initially we want to minimize KL(b||p) and not KL(p||b) (at least it's what allows to derive the BP equations), and if we take b and p to be factorized (=prod of their marginals) then we get this formula
        else:
            return np.mean(y_true * (np.log(y_true) - logsig(y_predict)) + 
                           (1-y_true) * (np.log(1-y_true) - logsig(-y_predict)) 
                          )#because p(X=1) = sig(log_odds) and p(X=0) = 1 - sig(log_odds) = sig(-log_odds)
    
    elif which_error == 'CE': #returns CE(p_true || p_predict)
        #By definition, CE(p||q) = - sum_x p(x) log(q(x))           
        #mean of the cross-entropies between the marginals (mean over the nodes, over the examples, but not over x_i --> no / 2)
        y_true, y_predict = from_list_of_dict_to_df(y_true, y_predict)
        y_true = y_true.to_numpy()
        y_predict = y_predict.to_numpy()
        if log == False:
            return - np.mean(y_true * np.log(y_predict) + 
                             (1 - y_true) * np.log(1-y_predict)
                            )#/ 2 #if using df, need a double np.mean
        else:
            return - np.mean(y_true * logsig(y_predict) +
                             (1 - y_true) * logsig(-y_predict)
                            )
        
    else:
        raise NotImplemented


def get_fitted_models(data_name,
                      which_CI='CIpower', which_alpha='directed_ratio', which_w=None, which_Mext=None,
                      select_method_fitting=None,
                      data_loaded=None,
                      path_folder="../../results_code/simulations_CI_BP/Better_BP/",
                      path_folder_alpha="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/",
                      verbose=False
                     ):
    """
    Returns all the fitted models
    """
    data_name = data_name.replace('graph_and_local_fields_', '')
    
    print("data_name = {}, which_CI = {}, which_alpha = {}, which_w = {}, which_Mext = {}".format(data_name, which_CI, which_alpha, which_w, which_Mext))
    
    list_files_alpha = os.listdir(path_folder_alpha)
    list_files_alpha = [file for file in list_files_alpha if data_name in file] #potential models
#     print("list_files_alpha = {}".format(list_files_alpha))
    
    #filter the non-fitted files
    list_files_alpha_keep = []
    for file in list_files_alpha:
        if data_loaded is not None:
            assert file in data_loaded.keys()
            d_res = data_loaded[file]
        else: #loaded_data is None
            d_res = load_pickle(file, path_folder=path_folder_alpha)
        alpha_fitted = d_res['alpha_fitted']
#         print("alpha_fitted.keys() = {}".format(alpha_fitted.keys()))
        if (which_CI, which_alpha, which_w, which_Mext) not in alpha_fitted.keys():
            print("Strange - (which_CI, which_alpha, which_w, which_Mext) = ({}, {}, {}, {}) was not fitted (in file {})"
                  .format(which_CI, which_alpha, which_w, which_Mext, file))
#             sys.exit()
#             print("all fitted models: {}".format(alpha_fitted.keys()))
        else:
            list_files_alpha_keep.append(file)
    if len(list_files_alpha_keep) == 0:
        print("No fitting procedure worked")
        sys.exit()
    list_models = [el.replace('_' + data_name, '').replace('learntalpha_', '') for el in list_files_alpha_keep]
#     print("list_models = {}".format(list_models))
    
    #filtering further (for instance for structure = ladder, it keeps the circular_ladder files...) --> check that 'pytorch' or 'scipy' are at the end of the model name, as it should be
    list_models_filter = []
    for el in list_models:
        list_last = ['pytorch', 'scipy']
        for s in list_last:
            if el[-len(s):] == s:
                list_models_filter.append(el)
    list_models = list_models_filter
#     print("list_models = {}".format(list_models))
#     list_models = ['supervised_MSE_pytorch', 'supervised_MSE_scipy'] #, 'supervised_CE_pytorch']
#     print("list of models to compare between: {}".format(list_models))
    if select_method_fitting in ['scipy', 'pytorch']:
        list_models_filter = [model for model in list_models if select_method_fitting in model]
        list_models = list_models_filter
        print("list of models to compare between (after filtering): {}".format(list_models))
        assert len(list_models) != 0
    if verbose:
        print("list_models = {}".format(list_models)) #TODO: do not show if there are at least one scipy model and one pytorch model
    
    return list_models
        
        
def get_best_alpha_list_methods(data_file, which_error='MSE', 
                                which_CI='CIpower', which_alpha='directed_ratio', which_w=None, which_Mext=None,
                                i_graph='all', verbose=False, 
                                data=None, data_loaded=None, select_method_fitting=None,
                                path_folder="../../results_code/simulations_CI_BP/Better_BP/",
                                path_folder_alpha="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/",
                                return_errors=True, load_errors=True, return_dict=False, list_which_error=['MSE','KL','CE'],
                                ignore_error_nan=False
                               ):
    """
    Finds the best parameters for Circular BP (or other models), based on the error on the validation set
    (comparing between the different fitting methods: scipy/pytorch, MSE/CE/KL)
    
    One can provide data_name instead of data_file (it doesn't change anything), i.e., with or without "graph_and_local_fields_"
    
    Usage:
    1) alpha, w = get_best_alpha_list_methods(data_file, i_graph=0)
    2) list_alpha, list_w = get_best_alpha_list_methods(data_file, i_graph='all')
    """
    if return_dict:
        assert return_errors == True
    
    data_name = data_file.replace('graph_and_local_fields_', '')
    
    list_models = get_fitted_models(data_name,
                                    which_CI=which_CI, which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext,
                                    select_method_fitting=select_method_fitting,
                                    data_loaded=data_loaded,
                                    path_folder=path_folder, path_folder_alpha=path_folder_alpha)
#     print("which_CI = {}, which_alpha = {}, which_w = {}, which_Mext = {}, list_models = {}".format(which_CI, which_alpha, which_w, which_Mext, list_models))
        
    damping = 0
    k_max = None
    parallel_Mext = True
    
    data_part = 'val'
    print_train = True
    
    if load_errors == False:
#         print("computing the errors values from fitted parameters")
        dict_error_all_errors = compute_errors(data_name, list_models, 
                                               list_data_part=[data_part], list_errors=[which_error],
                                               which_CI=which_CI, which_alpha=which_alpha, which_w=which_w,
                                               which_Mext=which_Mext,
                                               damping=damping, k_max=k_max, parallel_Mext=parallel_Mext,
                                               data=data,
                                               i_graph=i_graph, path_folder=path_folder, path_folder_alpha=path_folder_alpha)
    #     print("errors (computed)", dict_error_all_errors)
    else: #this is faster (loading the errors previously computed)
#         print("loading the errors values (already computed and saved)")
        dict_error_all_errors = {}
        if print_train:
            dict_error_all_errors_train = {}
            dict_error_all_errors_test = {}
        for model in list_models:
            if data_loaded is not None:
                assert 'learntalpha_' + model + '_' + data_name in data_loaded.keys()
                d_res = data_loaded['learntalpha_' + model + '_' + data_name]
            else:
                d_res = load_pickle('learntalpha_' + model + '_' + data_name)
    #         print("here2")
    #         print(d_res['error_' + data_part].keys())
    #         print(d_res['error_' + data_part][which_error][which_CI, which_alpha, which_w, which_Mext][i_graph])
            errors = d_res['error_' + data_part][which_error][which_CI, which_alpha, which_w, which_Mext]
            if i_graph != 'all':
                errors = [errors[i_graph]]
            dict_error_all_errors[which_error, data_part, model] = errors
            if print_train:
                errors = d_res['error_' + 'train'][which_error][which_CI, which_alpha, which_w, which_Mext]
                if i_graph != 'all':
                    errors = [errors[i_graph]]
                dict_error_all_errors_train[which_error, 'train', model] = errors
                errors = d_res['error_' + 'test'][which_error][which_CI, which_alpha, which_w, which_Mext]
                if i_graph != 'all':
                    errors = [errors[i_graph]]
                dict_error_all_errors_test[which_error, 'test', model] = errors
    #     print("errors (loaded)", dict_error_all_errors_loaded)
    
#     n_graphs = len(dict_error_all_errors[list(dict_error_all_errors.keys())[0]])
#     print("n_graphs = {}".format(n_graphs))
    if i_graph == 'all':
        n_graphs = len(dict_error_all_errors[list(dict_error_all_errors.keys())[0]])
#         print("n_graphs", n_graphs)
        list_i_graph = range(n_graphs)
#         list_i_graph = range(len(list_graph))
    elif type(i_graph) == int:
        list_i_graph = [i_graph]
    else:
        assert (type(i_graph) == list) and type(i_graph[0]) == int
        list_i_graph = i_graph

    list_alpha, list_w, list_w_input = [], [], []
#     list_errors = []
    list_errors_train = {key: [] for key in list_which_error}
    list_errors_val = {key: [] for key in list_which_error}
    list_errors_test = {key: [] for key in list_which_error}
    #Select the model with the lowest error (with which_error) on the validation set (for each graph)
    if verbose:
        list_best_model = []
    for i, i_graph in enumerate(list_i_graph): #range(n_graphs):
        dict_errors = {model: dict_error_all_errors[which_error, data_part, model][i] for model in list_models}
#         print("dict_errors = {}".format(dict_errors))
        if verbose:
            if i == 0:
                print("labels: {}".format(list(dict_errors.keys())))
#             print("dict_errors = {}".format(dict_errors))
#             print("dict_errors = {}".format(list(dict_errors.values())))
            print("log(dict_errors) = {}".format(np.log(list(dict_errors.values()))/np.log(10))) #to plot
        if print_train:
            dict_errors_train = {model: dict_error_all_errors_train[which_error, 'train', model][i] for model in list_models}
            dict_errors_test = {model: dict_error_all_errors_test[which_error, 'test', model][i] for model in list_models}
#             print("dict_errors_train = {}".format(dict_errors_train))
#             print("dict_errors_train = {}".format(list(dict_errors_train.values())))
#             print("log(dict_errors_train) = {}".format(np.log(list(dict_errors_train.values()))/np.log(10)))
#             print("log(dict_errors_test) = {}".format(np.log(list(dict_errors_test.values()))/np.log(10)))
#             print("log(errors_train) - log(errors_val) = {}".format(np.log(list(dict_errors_train.values()))/np.log(10) - np.log(list(dict_errors.values()))/np.log(10)))
#             print("log(errors_train) - log(errors_test) = {}".format(np.log(list(dict_errors_train.values()))/np.log(10) - np.log(list(dict_errors_test.values()))/np.log(10)))
#             print(np.abs(dict_errors['supervised2_MSE_scipy'] - dict_errors['supervised_MSE_scipy']) < 1e-6)
        best_model = get_keys_min_val(dict_errors, ignore_error_nan=ignore_error_nan) #best model = model with the lowest error
        if ignore_error_nan and best_model == 'error_all_nan':
            print("Error: all errors are nan for this graph - because ignore_error_nan = True then picking the best model randomly")
            best_model = list(dict_errors.keys())[0]
        if verbose:
#             print("i_graph = {}: best model = {}".format(i_graph, best_model))
            list_best_model.append(best_model)
        if data_loaded is not None:
            d_res = data_loaded['learntalpha_' + str(best_model) + "_" + str(data_name)]
        else: #data_loaded is None
            d_res = load_pickle('learntalpha_' + str(best_model) + "_" + str(data_name), path_folder=path_folder_alpha)
        alpha_fitted = d_res['alpha_fitted'][which_CI, which_alpha, which_w, which_Mext][i_graph]
        w_fitted = d_res['w_fitted'][which_CI, which_alpha, which_w, which_Mext][i_graph]
        w_input_fitted = d_res['w_input_fitted'][which_CI, which_alpha, which_w, which_Mext][i_graph]
        error_train = {key: d_res['error_train'][key][which_CI, which_alpha, which_w, which_Mext][i_graph] 
                       for key in list_which_error}
        error_val = {key: d_res['error_val'][key][which_CI, which_alpha, which_w, which_Mext][i_graph] 
                     for key in list_which_error}
        error_test = {key: d_res['error_test'][key][which_CI, which_alpha, which_w, which_Mext][i_graph]
                      for key in list_which_error}
#         list_errors.append(dict_errors[best_model]) #validation errors for which_error
        list_alpha.append(alpha_fitted)
        list_w.append(w_fitted)
        list_w_input.append(w_input_fitted)
        for key in error_train.keys():
            list_errors_train[key].append(error_train[key])
            list_errors_val[key].append(error_val[key])
            list_errors_test[key].append(error_test[key])
    
    if verbose:
        if len(list_best_model) == 1: #only one graph fitted (I think)
            print("best_model = {}".format(list_best_model[0]))
        else:
            print("list_best_model = {}".format(list_best_model))
#     print("list_error_train = {}".format(list_errors_train['MSE']))
#     print("list_error_val = {}".format(list_errors_val['MSE']))
#     print("list_error_test = {}".format(list_errors_test['MSE']))
#     print("log(list_error_train) = {}".format(list(np.log(list_errors_train['MSE'])/np.log(10))))
#     print("log(list_error_val) = {}".format(list(np.log(list_errors_val['MSE'])/np.log(10))))
#     print("log(list_error_test) = {}".format(list(np.log(list_errors_test['MSE'])/np.log(10))))
        
    if len(list_alpha) == 1:
        assert len(list_w) == 1 and len(list_w_input) == 1
        if return_dict == False:
            if return_errors == False:
                return list_alpha[0], list_w[0], list_w_input[0]
            else:
                return list_errors_train[0], list_errors_val[0], list_errors_test[0], list_alpha[0], list_w[0], list_w_input[0]
        else:
            return {'error_train': list_errors_train[0], 'error_val': list_errors_val[0], 'error_test': list_errors_test[0], 
                    'alpha_fitted': list_alpha[0], 'w_fitted': list_w[0], 'w_input_fitted': list_w_input[0]}
        
    if return_dict == False:
        if return_errors == False:
            return list_alpha, list_w, list_w_input
        else:
            return list_errors_train, list_errors_val, list_errors_test, list_alpha, list_w, list_w_input
    else:
        return {'error_train': list_errors_train, 'error_val': list_errors_val, 'error_test': list_errors_test, 
                'alpha_fitted': list_alpha, 'w_fitted': list_w, 'w_input_fitted': list_w_input}
                
        
def get_d_res_best_val(data_name, which_error='MSE', list_models_to_load='all_fitted_models',
                       path_folder_alpha="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/",
                       verbose=False
                      ):
    """
    Returns a dict of keys alpha_fitted, w_fitted, w_input_fitted, error_train, error_val, error_test (where error_train values are itself dict with MSE,KL,CE)
    Each value of the dict is itself a dict with keys which_CI, which_alpha, which_w, which_Mext
    """
    list_which_error = ['MSE', 'KL', 'CE'] #types of errors on which to return the error values
    data_file = load_pickle('learntalpha_supervised_MSE_pytorch_' + data_name)
    d_res_model_best_val = {key: {} for key in data_file.keys()} #alpha_fitted, w_fitted, w_input_fitted, error_train, error_val, error_test
    for key in d_res_model_best_val.keys():
        if 'error' in key:
            d_res_model_best_val[key] = {key_error: {} for key_error in list_which_error}
    list_filenames = os.listdir(path_folder_alpha)
    list_filenames = [el for el in list_filenames if (data_name in el and 'learntalpha' in el)]
    data_files = {el: load_pickle(el) for el in list_filenames}
    if list_models_to_load == 'all_fitted_models':
        list_fitted_models = [list(data_files[el]['alpha_fitted'].keys()) for el in list_filenames] #[list(load_pickle(el)['alpha_fitted'].keys()) for el in list_filenames]
        list_fitted_models = list(set(list(itertools.chain.from_iterable(list_fitted_models))))
        list_fitted_models = list(data_file['alpha_fitted'].keys())
#         print(list_fitted_models)
        list_models_to_load = list_fitted_models
    for (which_CI, which_alpha, which_w, which_Mext) in list_models_to_load:
#         print(which_CI, which_alpha, which_w, which_Mext)
        d_res_model = get_best_alpha_list_methods(
            data_name, which_error=which_error, 
            which_CI=which_CI, which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext, 
            return_dict=True, data_loaded=data_files, verbose=verbose
        )
        for key, val in d_res_model.items():
#             print(key)
#             print(val)
            if 'error' in key:
                for key_error in val.keys():
#                     print(key_error)
                    d_res_model_best_val[key][key_error][which_CI, which_alpha, which_w, which_Mext] = val[key_error]
            else:
                d_res_model_best_val[key][which_CI, which_alpha, which_w, which_Mext] = val
    return d_res_model_best_val
        
    
def compute_errors(data_name, 
                   list_models=['supervised_MSE_pytorch', 'supervised_MSE_scipy'],
                   list_data_part=['train', 'val', 'test'], list_errors=['MSE', 'CE', 'KL'],
                   which_CI='CIpower', which_alpha='directed_ratio', which_w=None, which_Mext=None,
                   damping=0, k_max=None, parallel_Mext=True,
                   i_graph='all', verbose=False,
                   data=None,
                   path_folder="../../results_code/simulations_CI_BP/Better_BP/",
                   path_folder_alpha="../../results_code/simulations_CI_BP/Better_BP/learnt_alpha/"
                  ):
    
    #Load the data
    if data is None:
        data_file = 'graph_and_local_fields_' + str(data_name) #'graph_and_local_fields_random_connected_8nodes_p055_bimodalw'
        list_graph, list_X, list_y = load_from_mat_list(data_file, path_folder=path_folder)
        which_data = 'all'
    else:
        assert len(data) == 3
        if type(data[0]) == list:
            which_data = 'all'
            list_graph, list_X, list_y = data
        else:
            which_data = 'i_graph'
            graph, X, y = data
            
    if i_graph == 'all':
        list_i_graph = range(len(list_graph))
    elif type(i_graph) == int:
        list_i_graph = [i_graph]
    else:
        assert (type(i_graph) == list) and type(i_graph[0]) == int
        list_i_graph = i_graph
        
    #compute the errors
    dict_error_all_errors = {}
    
    for which_error, data_part, model_name in itertools.product(list_errors, list_data_part, list_models):
        
        if verbose:
            print("which_error = {}, data_part = {}, model_name = {}".format(which_error, data_part, model_name))
        
        d_res = load_pickle('learntalpha_' + str(model_name) + "_" + str(data_name), path_folder=path_folder_alpha)
        alpha_fitted = d_res['alpha_fitted']
        list_alpha = alpha_fitted[which_CI, which_alpha, which_w, which_Mext]
        w_fitted = d_res['w_fitted']
        list_w  = w_fitted[which_CI, which_alpha, which_w, which_Mext]
        w_input_fitted = d_res['w_input_fitted']
        list_w_input  = w_input_fitted[which_CI, which_alpha, which_w, which_Mext]
#         print("[1]:", list_alpha[1])
#         print("list_w[1] = {} (data_name = {}".format(list_w[1], data_name))
    
        for i, (alpha, w, w_input) in enumerate(list(zip(list_alpha, list_w, list_w_input))):
            if i not in list_i_graph:
                continue
            if which_data == 'all': #otherwise, graph, X and y are already defined
                graph, X, y = list_graph[i], list_X[i], list_y[i]
#             print("i = {}".format(i))
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(X, y, return_test=True)
            dict_Xy = {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [X_test, y_test]}
            X_select, y_select = dict_Xy[data_part]
            model = Model_alpha_ij_all(graph, which_CI, which_alpha=which_alpha, which_w=which_w, which_Mext=which_Mext, 
                                       damping=damping, k_max=k_max)
#             y_predict = Model_alpha_ij_CI(graph, damping=damping).predict_BP(X_select)
#             error_BP = accuracy_score(y_predict, y, which_error=which_error)

            y_predict = model.predict(X_select, alpha=alpha, w=w, w_input=w_input, 
                                      which_CI=which_CI, parallel_Mext=parallel_Mext)
#             if data_part == 'train':
#                 print("y_predict[55]", y_predict[55])
#                 print("y_train[55]", y_select.to_numpy()[55])
#                 print("len(y_predict) = {}".format(len(y_predict)))
            error_value = accuracy_score(y_predict, y_select, which_error=which_error)
            if verbose:
                print("{} error = {}".format(which_error, error_value))

            if (which_error, data_part, model_name) not in dict_error_all_errors.keys():
                dict_error_all_errors[which_error, data_part, model_name] = []
            dict_error_all_errors[which_error, data_part, model_name].append(error_value)

    #dict_error_all_errors = {'train': error_train_all_errors, 'val': error_val_all_errors, test': error_test_all_errors}
    #error_test_all_errors = {which_error: error_test_all}
    #where error_test_all = {(which_CI, which_alpha, which_w, which_Mext): [(error_train_model1, error_train_model2) for all graphs]}
    return dict_error_all_errors
        
        
def goodness_algo(B_true, B_approx):
    """
    Defines a goodness of approximation (result of CI vs true marginals)
    The inputs are dictionaries, representing the beliefs (log-odds of the marginal)
    
    In [Knoll, Mehta, Chen et al, 2017] (https://arxiv.org/pdf/1605.06451.pdf),
    the Goodness of approximation is defined as: 
    $$MSE = \frac{2}{N} \sum\limits_{i=1}^N |P(x_i = 1) - \tilde{P}(x_i = 1)|^2$$
    """
    assert list(B_approx.keys()) == list(B_true.keys())
    B_true = np.array(list(B_true.values()))
    B_approx = np.array(list(B_approx.values()))
    p_1_approx = sig(B_approx)
    p_1_true = sig(B_true)
    N = len(p_1_true)
    return 2/N * np.sum((p_1_approx - p_1_true)**2)


def get_compensating_alpha_dict(graph):
    """
    Getting alpha_dict which is supposed to improve BP (it's a very simplistic proposal and might not work)
    = hand-made proposal
    """
    def get_edges_list_from_path(path):
        return [(path[i], path[i+1]) for i in range(len(path)-1)]

    for node1, node2, d in graph.edges(data=True):
        d['J'] = 2 * d['weight'] - 1
    for node1, node2, d in graph.edges(data=True):
        d['minus_log_abs_J'] = - np.log(np.abs(d['J']))

    alpha_dict_compensating = {}
    for (node1, node2) in graph.edges:
        print("edge ({}, {})".format(node1, node2))
        
        #1st try: looking at cycles which include i->j. We are looking for the product of J_ij from j to i, not from j to j
#         graph_modif = graph.copy()
#         graph_modif.remove_edge(node1, node2)
#         shortest_path = nx.shortest_path(graph_modif, source=node1, target=node2, weight='minus_log_abs_J')
#         edges_shortest_path = get_edges_list_from_path(shortest_path)
#         prod_J_path = np.prod([graph.edges[node1_edge, node2_edge]['J'] for (node1_edge, node2_edge) in edges_shortest_path]) #product of J_ij from neighbors node1 to node2 (not by the direct path node1->node2)
#         eps_ij = - prod_J_path / graph.edges[node1, node2]['J'] #M_ij = F_ij(B_i - M_{ji} - [prod_{e in j->i} J_e] * J_{ji}^{-1}) i.e. alpha_ij = 1 + [prod_{e in j->i} J_e] * J_{ji}^{-1}
#         alpha_ij = 1 - eps_ij 
#         if alpha_ij != 1:
#             alpha_dict_compensating[node1, node2] = alpha_ij
#             alpha_dict_compensating[node2, node1] = alpha_ij
            
        #2nd try (looking at cycles i->...->i which don't go through j)
        graph_modif = graph.copy()
        graph_modif.remove_edge(node1, node2)
        cycle_basis = nx.algorithms.minimum_cycle_basis(graph_modif, weight='minus_log_abs_J') #the function is only defined for undirected graphs (thus impossible to use graphs for which J_ij != J_ji)
        for edge_dir in [(node1, node2), (node2, node1)]:
            i, j = edge_dir
            eps_ij = 0
            for cycle in cycle_basis:
                if i in cycle:
                    edges_cycle = get_edges_list_from_path(cycle) 
                    for edge in edges_cycle: #because minimum_cycle_basis might not return the path in the right order...
                        assert edge in graph_modif.edges 
                    prod_J_cycle = np.prod([graph.edges[edge[0], edge[1]]['J'] for edge in edges_cycle]) * 2 #*2 because the cycle can be taken in both directions (here we hypothesize that J_ij=J_ji)
                    eps_ij -= prod_J_cycle
            alpha_ij = 1 - eps_ij
            if alpha_ij != 1:
                alpha_dict_compensating[i, j] = alpha_ij
            
    return alpha_dict_compensating

    
def split_train_test(X, y, portions_dataset={'train': 0.5, 'val': 0.25, 'test': 0.25}, return_test=False):
    """
    Splitting between training and validation sets
    Potentially (if 
    """
    #split into training set and test set
    n = X.shape[0]
#     print(X.shape, y.shape)
    portion_train = portions_dataset['train']
    portion_val = portions_dataset['val']
    train_index = range(0, int(portion_train*n))
    val_index = range(int(portion_train*n), int((portion_train + portion_val)*n))
#     print(train_index, val_index)
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]
#     print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    if return_test == False: #default
        return X_train, X_val, y_train, y_val
    else:
        portion_test = portions_dataset['test']
        test_index = range(int((portion_train + portion_val)*n), n)
#         print(test_index)
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        return X_train, X_val, X_test, y_train, y_val, y_test


def generate_p_1_true_one(M_ext, graph):
    B_true = run_exact_inference(graph, M_ext) #log-odds
    p_1_true = {key: sig(val) for key, val in B_true.items()} #from log-odds to p(X=1)
    return p_1_true
    
    
def generate_p_1_true_all(X, graph, n_cpus=4):
    """
    Generates the true marginals, based on graph and a list of local beliefs (= unitary potentials)
    
    X has shape (N, n_nodes)
    The output y also has shape (N, n_nodes)
    """
    list_M_ext = from_df_to_list_of_dict(X)
    with Pool(n_cpus) as p:
        list_p_1_true = p.starmap(generate_p_1_true_one, zip(list_M_ext, repeat(graph)))
    #Transform into pandas dataframe
    y = pd.DataFrame(list_p_1_true)
    return y


# def generate_p_1_true_all_examples(list_X, list_graph, n_cpus=4):
#     """
#     Faster (probably) version of: "[generate_p_1_true_all(X, graph) for X, graph in zip(list_X, list_graph)]""
#     """
#     list_list_M_ext = []
#     for X in list_X:
#         if type(X) == pandas.core.frame.DataFrame:
#             list_M_ext = X.to_dict('records') #converts df into list of dict
#         else:
#             list_M_ext = X #list of dict
#         list_list_M_ext.append(list_M_ext)
#     #flatten the list of list into a list
#     TODO
    

def generate_data(graph, N=100, n_cpus=4):
    """
    Generates data = couples (M_ext, p_1_true)
    """
    np.random.seed() # np.random.seed(0) if I want reproducibility
#     list_M_ext = []
#     list_p_1_true = []
#     for i in range(N):
#         #Create the external input
#         values_constant_input = np.random.normal(loc=np.random.normal(),
#                                                  size=len(list(graph.nodes)))
#         M_ext = generate_M_ext('constant', graph, stimulated_nodes='all', values_constant_input=values_constant_input)
#     #     print("M_ext = {}".format(M_ext))
#         B_true = run_exact_inference(graph, M_ext)
#         p_1_true = {key: sig(val) for key,val in B_true.items()}
#         list_M_ext.append(M_ext)
#         list_p_1_true.append(p_1_true)
        
    with Pool(n_cpus) as p:
        list_M_ext = p.starmap(generate_M_ext_one, zip(range(N), repeat(graph)))
    with Pool(n_cpus) as p: #not sure whether I can remove it and follow directly with the line below (it seemed to work though...). Probably a bit slower with 2 "with Pool ..." than 1
        list_p_1_true = p.starmap(generate_p_1_true_one, zip(list_M_ext, repeat(graph)))
    
    #Transform into pandas dataframe
    X = pd.DataFrame(list_M_ext)
    y = pd.DataFrame(list_p_1_true)
    
    return X, y
      
    
def generate_examples_from_list_graph_G(list_graph, list_X=None, list_y_partial=None, N=300, which_error='KL', std_Mext=1/2,
                                        log=False, filter_bad_graphs=True):
    print("function generate_examples_from_list_graph_G is deprecated - use generate_examples_from_list_graph instead")
    sys.exit()
    
    
def generate_examples_from_list_graph(list_graph, list_X=None, list_y_partial=None, N=300, which_error='KL', std_Mext=1/2,
                                        log=False, filter_bad_graphs=True):
    """
    Generating examples (graph with weights, local fields)
    If n_nodes is kept None, then graphs have a random number of nodes (between 7 and 9)
    
    list_X can be provided (but it's not the default option)
    """
    if list_X is None: #default
        list_X_provided = False
        list_X = []
        nodes = max([len(graph) for graph in list_graph]) #10 #max number of nodes for the graph used later
        assert max([len(list(graph.nodes)) for graph in list_graph]) <= nodes, "Too many nodes in the provided graphs"
        X_all = generate_M_ext_all(nx.erdos_renyi_graph(n=nodes, p=1), 
                                   N=N, std_Mext=std_Mext
                                  ) #the 1st argument (graph) is used not later, it is just needed for the number of nodes
    else:
        list_X_provided = True
    list_y = []
    for i_graph, graph in enumerate(list_graph):

        print("########## i_graph = {} (out of {}) ##################".format(i_graph + 1, len(list_graph)))

        #checking whether BP converges for this graph
#         S = get_stability_matrix_CI(graph, Alpha_obj({'alpha':1}))
        S = get_A_CIpower_matrix(graph, alpha=Alpha_obj({'alpha':1}))
        eigenvals = np.linalg.eigvals(S)
    #     eigenvals_real = eigenvals.real
    #     eigenvals_imag = eigenvals.imag
        spectral_radius = np.max(np.abs(eigenvals))
        if spectral_radius >= 1:
            print("undamped BP might not converge (spectral radius >=1)") #rather than convergence, it means that 0 (fixed point of BP without local fields) is not stable - there could be convergence to other fixed points or frustration
#             continue
        
    #     plot_graph(graph, print_w_edges=False)

        #generate data
    #     X, y = generate_data(graph, N=N)
        if list_X_provided == False:
            X = X_all[list(graph.nodes)] #arbitrary choice (it could be something else...)
        else:
            X = list_X[i_graph]
        if (list_y_partial is not None) and (i_graph < len(list_y_partial)): 
            y = list_y_partial[i_graph]
        else: #default
            y = generate_p_1_true_all(X, graph)
            
        if filter_bad_graphs:
            #gives the performance of BP (and removes examples for which BP performs well)
            X_train, X_val, y_train, y_val = split_train_test(X, y)
            y_predict = Model_alpha_ij_CI(graph).predict_BP(X_val, log=log)
            error_BP = accuracy_score(y_predict, y_test, which_error=which_error, log=log)
            print("val set: {} error (BP) = {}".format(which_error, error_BP))
            if which_error == 'MSE' and error_BP < 0.005: #0.05
                print("BP performs well on this graph - skipping (actually not skipping with the new code")
    #             continue
            elif which_error != 'MSE':
                print("Find a criterion for the error ({})".format(which_error))
        
        if list_X_provided == False:
            list_X.append(X)
        list_y.append(y)

#         print("Number of examples up to now: {}".format(i_graph))
        
    return list_X, list_y



def generate_examples(N_examples, weighting_type, type_graph='random_connected', 
                      skipping_if_good_BP=True, skipping_if_BP_does_not_converge=True,
                      N=300, which_error='KL', log=False, **kwargs):
    """
    Generating examples (graph with weights, local fields)
    Example for weighting_type: 'bimodal_w' (see function create_random_weighted_graph in graph_generator.py)
    Examples for kwargs: n_nodes=None, proba=0.15  (resp. the number of graph nodes, and the connection proba)
    
    N_examples is the number of graphs generated 
    For each graph, (X, y) is generated, where N is the number of examples of X (= as well of y)
    X is composed of vectors of size Mext. It has N such vectors.
    y is composed of vectors of size Mext. It has N such vectors.
    """
    if 'n_nodes' in kwargs.keys():
        nodes_max = kwargs['n_nodes']
    else:
        nodes_max = 10 #max number of nodes for the graph used later
    if 'std_Mext' in kwargs.keys():
        std_Mext = kwargs['std_Mext']
        print('std_Mext = {}'.format(std_Mext))
        X_all = generate_M_ext_all(nx.erdos_renyi_graph(n=nodes_max, p=1), 
                                   N=N, std_Mext=std_Mext
                                  ) #the 1st argument (graph) is used not later, it is just needed for the number of nodes
    else: #use the default value for std_Mext
        X_all = generate_M_ext_all(nx.erdos_renyi_graph(n=nodes_max, p=1), 
                                   N=N) #the 1st argument (graph) is used not later, it is just needed for the number of nodes
    
    list_graph = []
    list_X = []
    list_y = []
    cpt = 0
    while cpt < N_examples: #for ind in range(50): #500

        print("#################################")

        ################  Create the weigthed graph  ################
        graph = create_random_weighted_graph(type_graph, weighting_type, **kwargs)

        #checking whether BP converges for this graph
#         S = get_stability_matrix_CI(graph, alpha=Alpha_obj({'alpha': 1}))
        S = get_A_CIpower_matrix(graph, alpha=Alpha_obj({'alpha':1}))
        eigenvals = np.linalg.eigvals(S)
    #     eigenvals_real = eigenvals.real
    #     eigenvals_imag = eigenvals.imag
        spectral_radius = np.max(np.abs(eigenvals))
        if skipping_if_BP_does_not_converge:
            if spectral_radius >= 1:
                print("undamped BP might not converge (spectral radius >=1)") #rather than convergence, it means that 0 (fixed point of BP without local fields) is not stable - there could be convergence to other fixed points or frustration
                continue
        
    #     plot_graph(graph, print_w_edges=False)

        #generate data
    #     X, y = generate_data(graph, N=N)
        X = X_all[list(graph.nodes)] #arbitrary choice (it could be something else...)
        y = generate_p_1_true_all(X, graph)
        
        #gives the performance of BP (and removes examples for which BP performs well)
        X_train, X_val, y_train, y_val = split_train_test(X, y)
        y_predict = Model_alpha_ij_CI(graph).predict_BP(X_val, log=log) #Model_alpha_ij_CIbeliefs(graph, k_max=None).predict_BP(X_test)
        error_BP = accuracy_score(y_predict, y_val, which_error=which_error, log=log)
        print("val set: {} error (BP) = {}".format(which_error, error_BP))
        if skipping_if_good_BP:
            if which_error == 'MSE' and error_BP < 0.005: #0.05
                print("BP performs well on this graph - skipping")
                continue
            elif which_error != 'MSE':
                print("Find a criterion for the error ({})".format(which_error))
                sys.exit()
                
        list_graph.append(graph)
        list_X.append(X)
        list_y.append(y)

        cpt += 1
        print("Number of examples up to now: {} (out of {})".format(cpt, N_examples))
        
    return list_graph, list_X, list_y


def get_MSE_and_beliefs_BP(list_graph, list_X, list_y, damping=0, plot_beliefs=False, which_error='MSE'):
    """
    list_beliefs_BP and MSE_BP
    + plot beliefs_BP vs true_marginals
    """
    list_MSE_BP = []
    list_beliefs_BP = []
    for graph, X, y in zip(list_graph, list_X, list_y):
    #     print("#################################")
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(X, y, return_test=True)
        ########################  fit ################################
        y_predict = Model_alpha_ij_CI(graph, damping=damping).predict_BP(X_test)
        MSE_BP = accuracy_score(y_test, y_predict, which_error=which_error)
    #     print("test set: MSE (BP) = {}".format(MSE_BP))
    #     if MSE_BP < 0.005: #0.05
    #         print("BP performs well on this graph - skipping")
    #         continue
        #checking whether BP converges for this graph
        S = get_stability_matrix_CI(graph, alpha=Alpha_obj({'alpha': 1}), damping=damping)
        eigenvals = np.linalg.eigvals(S)
        spectral_radius = np.max(np.abs(eigenvals))
#         if spectral_radius >= 1:
#             print("BP doesn't converge (spectral radius >=1)") #rather than convergence, it means that 0 (fixed point of BP without local fields) is not stable - there could be convergence to other fixed points or frustration
#             plot_eigenvalues(eig_values=eigenvals)
#             print("max(Re(eigenvalue)) =",np.max(eigenvals.real))
        list_beliefs_BP.append(y_predict)
        list_MSE_BP.append(MSE_BP)
#         plot_graph(graph, print_w_edges=True)
        #Show marginals computed by BP vs true marginals
#         print(y_test.to_numpy().shape, pd.DataFrame(y_predict).to_numpy().shape)
        if plot_beliefs:
            plt.scatter(y_test.to_numpy(), pd.DataFrame(y_predict).to_numpy())
            plt.xlabel('True marginals', size=15)
            plt.ylabel('Marginals from BP', size=15)
            x_lin = np.linspace(0,1,2)
            plt.plot(x_lin, x_lin, linestyle='--', color='black')
            plt.show()
            
    return list_MSE_BP, list_beliefs_BP