#! usr/bin/python
# -*- coding: ISO-8859-1 -*-

import torch
# print("version of torch (should be >=0.2):", torch.__version__)
version_torch = [int(el) for el in torch.__version__.split(".")][:2]
assert(version_torch >= [0, 2]) #version of torch should be >=0.2. If not, raises an AssertionError

if torch.cuda.is_available():
    print("Cuda is available")
    print("Cuda version:", torch.version.cuda)
#     print(torch.cuda.current_device())
else:
    print("Cuda is not available")
import torch.nn as nn


import matplotlib.pyplot as plt #%matplotlib inline  
# from utils import *
from torch.autograd import Variable
import torch
import random
import numpy as np
from scipy.special import expit as sigmoid
import itertools
from utils_compensating_alpha import *

    
def get_optimizer(model, learning_rate, algo='Rprop'):
    """
    Selecting the Pytorch optimizer
    See http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    
    Comments on optimizers (from best to worst for this problem and with learning rate = 2*1e-3):
    - Very fast and good quality optimization: Rprop (fast to converge to a (nearly) optimal value, and converges to the same parameters as the other algorithms)
    - Slow steps but good convergence in a few steps: LBFGS
    - Slow to converge (at least x3 wrt Rprop): SGD, Adagrad, Adadelta and Adam are really slow to converge and the optimized value is the same as for Rprop (up to the 6th or 7th decimal) and the optimized parameters are really similar as well
    - Very slow to converge: ASGD
    - The closure function is mandatory for LBFGS, but optional for other methods. For these other methods, it makes each step slower and does not seem to improve the convergence nore the number of steps needed to get convergence
    """
    if algo == 'Rprop':
        optimizer = torch.optim.Rprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif algo == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) #torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("optimization algorithm not known")
        sys.exit()
#     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) #torch.optim.SGD(model.parameters(), lr=learning_rate)
#     optimizer = torch.optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) #torch.optim.SGD(model.parameters(), lr=learning_rate)
#     optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#     optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#     optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) #fast to converge to a (nearly) optimal value (and converges to the same parameters as the other algorithms)
#     optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) #requires a closure (not done yet)
    return optimizer


def is_saturating(l, percent=1):
    """
    Early stopping criterion: 1% difference max in the last 20 iterations
    """
    converged = (100 * np.abs(l[-1] - l[-20]) / l[-1] < percent) #percent % difference at max
    return converged


def data_to_pytorch_variables(x):
    if type(x) == pd.DataFrame:
        x = x.to_numpy()
    dtype = torch.FloatTensor
    x_v = torch.from_numpy(x).type(dtype)
#     print("x_v.size()", x_v.size())
    return x_v


def cross_entropy(y_predict, y_true):
    """
    Using numpy variables
    """
    return - np.mean(y_true * np.log(y_predict) + (1-y_true) * np.log(1-y_predict))


def get_model(graph, which_CI, which_alpha, which_w, which_Mext, k_max=None, damping=0, 
              verbose=False, print_free_params=False):
    """
    damping=0 corresponds to BP/CI : M_new = (1-damping)*F(M_old) + damping*M_old
    Note that damping=1-dt if we write M_new = M_old + dt*(F(M_old) - M_old) i.e. dM/dt = - M_old + F(M_old)   (tau=1/dt)
    """
    if k_max is not None:
        assert which_CI == 'CIbeliefs'
        
    assert which_CI in list_available_models, "{} is not among the available models".format(which_CI)
    
    if which_CI == 'BP':
        which_CI = 'CI'
        which_alpha = '1'
        
    fixed_parameters = get_fixed_parameters_model(graph, which_alpha, which_w, which_Mext)
    if verbose:
        print("fixed_parameters = {}".format(list(fixed_parameters.keys())))
    
    if which_CI not in dict_models.keys():
        print("which_CI = {} is not a valid option".format(which_CI))
        sys.exit()
    model = dict_models[which_CI](fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=damping, 
                                  print_free_params=print_free_params)
    return model


def get_fixed_parameters_model(graph, which_alpha, which_w, which_Mext):
    """
    Get the list of fixed parameters in the model
    """
    fixed_parameters = {}
    
    if which_alpha in ['0', '1']:
        fixed_parameters['alpha'] = float(which_alpha)
    
    if which_w == None: #taking the weights from graph
        fixed_parameters['w'] = get_w_matrix(graph) #numpy array of size (n_nodes*n_nodes)
    
    if which_Mext == None:
        fixed_parameters['w_input'] = 1 #maybe build a vector instead?
    
    return fixed_parameters


loss_fn_CE = torch.nn.BCEWithLogitsLoss(reduction='mean') #LOGISTIC REGRESSION: log-loss function
loss_fn_MSE = torch.nn.MSELoss(reduction='mean') #Mean-square error
# loss_fn_KL = torch.nn.KLDivLoss(reduction='batchmean') #See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss #'mean' raises a warning  #batchmean doesn't work either as such
loss_fn_KL = torch.nn.KLDivLoss(reduction='mean') #See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss #'mean' raises a warning  #batchmean doesn't work either as such
#KL(p_i,q_i) = sum_{x_i} p_i(x_i) * log(p_i(x_i) / q_i(x_i))   (only depends on the unitary marginals) ---> KL(prod_i p_i(x_i), prod_i q_i(x_i)) = ... = sum_i sum_{x_i} p_i(x_i) * log(p_i(x_i) / q_i(x_i))
#Cross-Entropy: L(p,q) = - sum_{x_i} p_i log(q_i(x_i))

class RNN_utils:
    
    def initialize_parameter(self, param_name, val_init=None):
        """
        Creates self.param_name (where param_name is the name of parameter, of type string)
        
        Previous version: self.w = self.get_initial_value_parameter('w')
        but this is easier to use setattr(self,'w',self.get_initial_value_parameter('w', start, end))
        because we can treat any parameter name and create it
        """
        setattr(self, param_name, self.get_initial_value_parameter(param_name, val_init=val_init))
            
    def initialize_parameter_mask(self, param_name):
        """
        Creates the mask (allowing to fit the weights but not the graph structure, for instance)
        """
        setattr(self, param_name, nn.Parameter(self.mask_structure, requires_grad=False).unsqueeze(2))
            
#     def initialize_remaining_parameters(self):
#         """
#         Parameters which are common to all models
#         """
#         self.initialize_parameter('some_constant')
            
    def get_initial_value_parameter(self, param_name, val_init=None): #, start, end):
        """
        For alpha, taking initial value = 1 (can be changed later)
        For w, taking initial value = graph weights (can be changed later)
        For w_input, taking initial value = 1 (can be changed later)
        """
        ################### K_nodes / K_edges / alpha #########################
        if param_name in ['latent_inverse_K_edges', 'K_nodes', 'alpha', 'K_nodes_val']:
            if param_name in self.fixed_parameters: #same as "if self.which_alpha in ['0', '1']:"
#                 print("param_name = {} - fixed".format(param_name))
                assert val_init is None
                return nn.Parameter((torch.ones(1) * self.fixed_parameters[param_name]).unsqueeze(1).unsqueeze(2), requires_grad=False)
            assert self.which_alpha in ['uniform', 'nodal', 'undirected', 'directed_ratio', 'directed_ratio_directed', 'directed']#= not '0' or '1' (because in these cases we don't fit)
            if self.which_alpha == 'uniform':
                val = 1 if (val_init is None) else val_init
                return nn.Parameter((torch.ones(1) * val).unsqueeze(1).unsqueeze(2), requires_grad=True)
            elif self.which_alpha == 'nodal': #alpha_ij = alpha_i (depends only on the sender node)
                assert param_name == 'K_nodes'
                val = 1 if (val_init is None) else val_init
                return nn.Parameter((torch.ones(self.n_nodes) * val).unsqueeze(1).unsqueeze(2), requires_grad=True)
            elif self.which_alpha == 'undirected':
                assert param_name == 'latent_inverse_K_edges'
                val = 1 if (val_init is None) else val_init
                return nn.Parameter((torch.ones(size=(self.n_nodes, self.n_nodes)) * val).unsqueeze(2), requires_grad=True)
#                 size = self.n_edges
#                 edges = torch.tensor(np.array(list(self.graph.edges)).T) #torch.tensor(zip(*list_edges)) does not seem to work (where list_edges = list(graph.edges))
#                 alpha_values = torch.ones(size) #alpha = 1
#                 alpha_half = nn.Parameter(torch.sparse_coo_tensor(edges, alpha_values, size=(self.n_nodes, self.n_nodes)).unsqueeze(2), requires_grad=True)
#                 return alpha_half.to_dense() + torch.transpose(alpha_half.to_dense(), 0, 1)
#             elif self.which_alpha == 'directed_ratio':
#                 #size = self.n_edges + self.n_nodes
            elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
                assert param_name in ['K_nodes', 'latent_inverse_K_edges']
                val = 1 if (val_init is None) else val_init
                if param_name == 'K_nodes':
                    return nn.Parameter((torch.ones(self.n_nodes) * val).unsqueeze(1).unsqueeze(2), requires_grad=True)
                elif param_name == 'latent_inverse_K_edges':
                    return nn.Parameter((torch.ones(size=(self.n_nodes, self.n_nodes)) * val).unsqueeze(2), requires_grad=True)
            elif self.which_alpha == 'directed':
                assert param_name == 'alpha'
                val = 1 if (val_init is None) else val_init
#                 size = 2 * self.n_edges
#                 list_edges_all = torch.tensor(np.array(get_all_oriented_edges(graph)).T)
#                 alpha_values = torch.ones(size) #alpha = 1
# #                 alpha_all = nn.Parameter(torch.sparse_coo_tensor(list_edges_all, alpha_values, size=(self.n_nodes, self.n_nodes)).unsqueeze(2), requires_grad=True)
#                 alpha_all = nn.Parameter(torch.sparse_coo_tensor(list_edges_all, alpha_values, size=(self.n_nodes, self.n_nodes)).to_dense().unsqueeze(2), requires_grad=True)
#                 return alpha_all.to_dense()
#                 return alpha_all
                return nn.Parameter((torch.ones(size=(self.n_nodes, self.n_nodes)) * val).unsqueeze(2), requires_grad=True)
        
        ################### weights J (previously w) #########################
        if param_name in ['latent_J', 'J']: #['latent_w', 'w']
            assert val_init in [None, 'tanh_J'] #tanh_J = initialize with tanh(J) instead of J --> no constraints on tanh(J): can be >1 or <-1
            #recovering the weights of graph
            edges = torch.tensor(np.array(list(self.graph.edges)).T) #torch.tensor(zip(*list_edges)) does not seem to work (where list_edges = list(graph.edges))
            list_weights = [d['weight'] for node1, node2, d in self.graph.edges(data=True)]
            assert np.min(list_weights) != 0 and np.max(list_weights) != 1, "Possible numerical errors - some weights J_ij are seen as infinite (i.e., w_ij = 0 or 1)"
            weights = torch.tensor(list_weights, dtype=torch.float32)
            if self.which_w is None:
                assert param_name == 'J' #'w'
#                 weights_half = nn.Parameter(torch.sparse_coo_tensor(edges, weights, size=(self.n_nodes, self.n_nodes)).unsqueeze(2), requires_grad=False)
#                 return weights_half.to_dense() + torch.transpose(weights_half.to_dense(), 0, 1)
                weights = torch.tensor(get_w_matrix(self.graph)).unsqueeze(2)
                J = 1/2 * torch.log(weights / (1-weights)) #torch.atanh(2*weights-1) #w = sig(2.J), i.e., 2*w-1 = tanh(J)
                if val_init == 'tanh_J':
                    J = torch.tanh(J)
                return nn.Parameter(J, requires_grad=False) #nn.Parameter(weights, requires_grad=False)
            assert self.which_w in ['undirected', 'directed'] #= not None
            if self.which_w == 'undirected':
                assert param_name == 'latent_J' #'latent_w'
#                 weights_half = nn.Parameter(torch.sparse_coo_tensor(edges, weights, size=(self.n_nodes, self.n_nodes)).unsqueeze(2), requires_grad=True)
#                 return weights_half.to_dense() + torch.transpose(weights_half.to_dense(), 0, 1)
                weights = torch.tensor(get_w_matrix(self.graph)).unsqueeze(2)
                J = 1/2 * torch.log(weights / (1-weights)) #torch.atanh(2*weights-1) #w = sig(2.J), i.e., 2*w-1 = tanh(J)
                if val_init == 'tanh_J':
                    J = torch.tanh(J)
                return nn.Parameter(J, requires_grad=True) #nn.Parameter(weights, requires_grad=True)
            elif self.which_w == 'directed':
                assert param_name == 'J' #'w'
                #size = 2 * self.n_edges
#                 idx = torch.LongTensor([1, 0])
#                 edges_other_dir = edges.index_select(0, idx) #edges in the other direction (compared to edges)
#                 edges_all = torch.cat([edges, edges_other_dir], dim=1) #all directed edges
#                 weights_all = torch.cat([weights, weights], dim=0)
#                 weights_half = nn.Parameter(torch.sparse_coo_tensor(edges_all, weights_all, size=(self.n_nodes, self.n_nodes)).unsqueeze(2), requires_grad=True)
#                 return weights_all.to_dense()
                weights = torch.tensor(get_w_matrix(self.graph)).unsqueeze(2)
                J = 1/2 * torch.log(weights / (1-weights)) #torch.atanh(2*weights-1) #w = sig(2.J), i.e., 2*w-1 = tanh(J)
                if val_init == 'tanh_J':
                    J = torch.tanh(J)
                return nn.Parameter(J, requires_grad=True) #nn.Parameter(weights, requires_grad=True)
            
        ################### input weights w_input #########################
        if param_name == 'w_input':
            if self.which_Mext is None:
#                 print("param_name = {} - fixed".format(param_name))
                assert val_init is None
                return nn.Parameter((torch.ones(1) * self.fixed_parameters[param_name]).unsqueeze(1).unsqueeze(2),
                                    requires_grad=False)
            else:
                val = 1 if (val_init is None) else val_init
                return nn.Parameter((torch.ones(self.n_nodes) * val).unsqueeze(1).unsqueeze(2), requires_grad=True)
                
#         #custom initialization of the weights
#         #see how the parameters are initialized by default: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
#         if param_name not in self.fixed_parameters: #option by default
# #         if str(param_name)+' fixed=' not in self.model_name: #option by default
# #             return nn.Parameter(torch.FloatTensor(1, 1).uniform_(start, end), requires_grad=True)
#             return nn.Parameter(torch.FloatTensor(1).uniform_(start, end), requires_grad=True)
#         else:
# #             return nn.Parameter(torch.ones(1, 1) * self.fixed_parameters[param_name], requires_grad=False)
#             return nn.Parameter(torch.ones(1) * self.fixed_parameters[param_name], requires_grad=False)
    
#     def clamp_parameters(self, list_free_parameters_model, restricted_parameters): 
#         """
#         #TODO: improve this function (=not have these specific rules in the "if", in order to be adaptable to any new model
#         #Potentially clamp the bias_lateral to be between -pi/2 and pi/2 (or -pi/4 and pi/4)
#         #ISN'T THERE ANY OTHER METHOD TO HAVE RESTRICTED PARAMETERS? BY CONSTRAINING THE GRADIENT FOR INSTANCE?
#         """
#         if ('w_s' in list_free_parameters_model) and ('weighted_Bayes_Horga' not in self.model_name):
#             self.w_s.data.clamp_(min=0.5, max=1) #"clamp_" means clamp done in place 
    
    def get_list_param_values(self):
        list_param_values = [param.item() for param in self.parameters()]
        return list_param_values
    
    def get_list_free_param_values(self):
        list_free_param_values = [param.item() for param in self.parameters()\
                                  if param.requires_grad]
        return list_free_param_values
    
    def get_list_param_names(self):
        list_param_names = [paramname_and_value[0] for paramname_and_value in self.named_parameters()]
        return list_param_names
    
    def get_list_free_param_names(self):
        list_free_param_names = [paramname_and_value[0] for paramname_and_value in self.named_parameters()\
                                 if paramname_and_value[1].requires_grad]
        return list_free_param_names
    
#     def convert_list_param_into_param(self, list_param): #TODO: improve this function to take into account the param_name_4/_8/_12
        
#         #TODO (?): compute indices_parameters_model and indices_free_parameters_model once for all (= at the initialization of the model
#         indices_parameters_model = [list_parameter_names.index(param_name) for param_name in self.list_parameter_names_model]
#         indices_free_parameters_model = [list_parameter_names.index(param_name) for param_name in self.list_free_parameter_names_model]

# #         print("list_parameter_names_model", self.list_parameter_names_model)
# #         print("list_free_parameter_names_model", self.list_free_parameter_names_model)
# #         print("indices_parameters_model", indices_parameters_model)
# #         print("indices_free_parameters_model", indices_free_parameters_model)
        
#         if 'weighted_Bayes_Horga' in self.model_name and 'beta_depends_on_nsamples' in self.model_name and "lottery" not in self.model_name:
#             parameters = [1., 1., 0., 0., 1., 1., 1., 0., 0.] #default values for w_s,w_p,alpha_s,alpha_p,beta_4,beta_8,beta_12,bias,epsilon
#         elif "lottery" not in self.model_name:
#             parameters = [1., 1., 0., 0., 1., 0., 0.] #default values for w_s,w_p,alpha_s,alpha_p,beta,bias,epsilon
#         else:
#             parameters = [1., 1., 0., 0., 1., 0., 0., 1., 0., 0.] #default values for w_s,w_p,alpha_s,alpha_p,beta,bias,epsilon,beta_lottery,bias_lottery,espilon_lottery
#         for j,index in enumerate(indices_parameters_model):
#             parameters[index] = list_param[j]
#         return parameters
    
    #tentative of alternative
    def convert_list_param_into_param(self, list_param):
        """
        Uses the global variable default_values_parameters
        """
        all_params = default_values_parameters.keys() #in the same order as above
        parameters = []
        i = 0
        for param_name in all_params:
            if param_name in self.list_parameter_names_model:
                parameters.append(list_param[i])
                i = i + 1
            parameters.append(default_values_parameters[param_name])
        return parameters
    # print(list_parameters_model)
    # convert_list_param_into_param_v2(list_param, list_parameters_model, model_name)
    
    
    def to_obj(self, verbose=False):
        """
        Transforms model attributes model.latent_inverse_K_edges, model.alpha, model.K_nodes, model.J, into w_obj and alpha_obj
        """
    
        which_CI = self.which_CI
        which_alpha = self.which_alpha
        which_w = self.which_w
        which_Mext = self.which_Mext

        #################### alpha #########################
        if verbose:
            print("which_alpha = {}".format(which_alpha))
        if which_alpha in ['0', '1']:
            alpha_obj = Alpha_obj({'alpha': float(which_alpha)})
        elif which_alpha == 'uniform':
            if which_CI == 'CI':
                alpha_obj = Alpha_obj({'alpha': self.alpha.cpu().detach().numpy()[0,0,0]})
            else:
                alpha_obj = Alpha_obj({'K_nodes_vector': np.ones(self.n_nodes) * self.alpha.cpu().detach().numpy()[0,0,0]})
        elif which_alpha == 'nodal':
            alpha_obj = Alpha_obj({'K_nodes_vector': self.K_nodes.cpu().detach().numpy()[:,0,0]})
        elif which_alpha == 'undirected':
            alpha_obj = Alpha_obj({'K_edges_matrix': 1 / ((self.latent_inverse_K_edges + torch.transpose(self.latent_inverse_K_edges, 0, 1))/2).cpu().detach().numpy()[:,:,0]})
        elif which_alpha == 'directed':
            alpha_obj = Alpha_obj({'alpha_matrix': self.alpha.cpu().detach().numpy()[:,:,0]})
        elif which_alpha == 'directed_ratio':
        #     alpha_obj = Alpha_obj({'alpha_matrix': self.get_alpha().cpu().detach().numpy()[:,:,0]})
            alpha_obj = Alpha_obj({'K_nodes_vector': self.K_nodes.cpu().detach().numpy()[:,0,0],
                                   'K_edges_matrix': 1 / ((self.latent_inverse_K_edges + torch.transpose(self.latent_inverse_K_edges, 0, 1))/2).cpu().detach().numpy()[:,:,0]})
        elif which_alpha == 'directed_ratio_directed':
            alpha_obj = Alpha_obj({'K_nodes_vector': self.K_nodes.cpu().detach().numpy()[:,0,0],
                                   'K_edges_matrix': 1 / self.latent_inverse_K_edges.cpu().detach().numpy()[:,:,0]})
        else:
            print("which_alpha = {} is not implemented".format(which_alpha))
            raise NotImplemented
        if verbose:
            print("alpha_obj = {}".format(alpha_obj))

        #################### w #########################
        if verbose:
            print("which_w = {}".format(which_w))
        if which_w  == None:
            w_obj = None
        #     w_obj = Alpha_obj({'alpha_matrix': self.get_w().cpu().detach().numpy()[:,:,0]}) #alternatively, for which_w = None (only this case), we can give w=None in simulate_CI
        elif which_w in ['undirected', 'directed']: #in [None, 'undirected', 'directed']:
#             w_obj = Alpha_obj({'alpha_matrix': self.get_w().cpu().detach().numpy()[:,:,0]})
            w_obj = Alpha_obj({'alpha_matrix': torch.sigmoid(2*self.get_J()).cpu().detach().numpy()[:,:,0]}) #w = sig(2J)
        # elif which_w == 'undirected':
        #     w_obj = Alpha_obj({'alpha_matrix': ((self.latent_w + torch.transpose(self.latent_w, 0, 1))/2).cpu().detach().numpy()[:,:,0]})
        else:
            raise NotImplemented
        if verbose:
            print("w_obj = {}".format(w_obj))

        #################### w_input #########################
        if verbose:
            print("which_Mext = {}".format(which_Mext))
        if which_Mext is None:
            w_input = None
        elif which_Mext == 'nodal':
            w_input = self.w_input.cpu().detach().numpy()[:,0,0]
#             print("Result w_input.shape = {}".format(w_input.shape))
        else:
            print("which_Mext = {} is not implemented".format(which_Mext))
            raise NotImplemented
        if verbose:
            print("w_input = {}".format(w_input))
        
        return alpha_obj, w_obj, w_input

    
    def parameters_into_dict(self):
        fitted_parameters = {}
        for param_name, param_value in list(self.named_parameters()):
            fitted_parameters[param_name] = param_value.item()
        return fitted_parameters
    
#     def forward(self, Mext):
#         """
#         Simulating the model (from Ls=0 to Lfinal)
#         """
#         #1. Initialization of M
# #         n_examples = Mext.size()[1]
# #         size = 2 * self.n_edges
# #         list_edges_all = torch.tensor(np.array(get_all_oriented_edges(graph)).T)
# #         M_values = torch.zeros(size) #M = 0
# #         M = torch.sparse_coo_tensor(list_edges_all, M_values, size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
# #         M = torch.sparse_coo_tensor(edges, alpha_values, size=(self.n_nodes, self.n_nodes)).unsqueeze(2) #HEREHERE #torch.zeros(size=(self.n_nodes, self.n_nodes, n_examples))
#         M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
    
#         #2. Running the model through the samples
#         for i in range(100):
#             M = self.F_next(Mext, M)
# #             if i == 2:
# #                 sys.exit()
#         B = self.get_B(Mext, M)
#         return B
    

    def get_probaR_model(self, X, verbose=True):
        """
        Returning p(x=1) based on Mext
        """
        Lfinal = self(X)
        return self.get_proba_categorization(Lfinal)
    
    
    def get_proba_categorization(self, Lfinal):
        """
        Returning p(x=1) based on Lfinal
        """
        return torch.sigmoid(Lfinal)

    def get_loss_categorization(self, Lfinal, y, which_loss='CE'):
        """
        See definitions of loss functions above
        """
        dict_loss_fn = {'CE': loss_fn_CE, 'KL': loss_fn_KL, 'MSE': loss_fn_MSE}
        loss_fn = dict_loss_fn[which_loss]
        if which_loss != 'MSE':
            return loss_fn(Lfinal, y)
        else:
            return loss_fn(torch.sigmoid(Lfinal), y)
        
    def get_accuracy_categorization(self, Lfinal, y):
        output1 = self.get_output_categorization(Lfinal)
        sign_output1 = torch.sign(output1)
#         return y == (sign_output1 + 1) / 2 #returns a boolean array
        return np.mean((y == (sign_output1 + 1) / 2).numpy()) #returns the mean of the boolean array

    def eval_model(self, X, y, which_loss='CE', verbose=True):
        """
        Compute the accuracies and likelihoods
        """
        
        #Simulate the model
        Lfinal = self(X)

        #Generate fake responses
#         proba_categorization_cat = self.get_proba_categorization(Lfinal)
#         y_fake = torch.bernoulli(proba_categorization_cat)

        #Compute the loss based on Lfinal
        loss = self.get_loss_categorization(Lfinal, y, which_loss=which_loss)
#         #Compute the accuracies based on Lfinal
#         accuracy_categorization = self.get_accuracy_categorization(Lfinal, y)
        
#         if verbose:
#             print('Accuracies of categorization:', np.round(accuracy_categorization_all_nsamples, 3))
#         likelihood_categorization = np.exp(- loss_categorization.item())
#         if verbose:
#             print('Likelihood of categorization:', np.round(likelihood_categorization, 3))
#         likelihood = np.exp(- loss.item())
#         if verbose:
#             print("Overall Likelihood:", likelihood)
        
        return loss.detach().numpy()
#         return {'likelihood': likelihood} #, 'y_fake': y_fake #possibly return more information than the overall likelihood (e.g. + accuracies, ...etc)
        
#     def get_K_edges(self):
#         return (self.latent_K_edges + torch.transpose(self.latent_K_edges, 0, 1)) / 2
    
    def get_inverse_K_edges(self):
        assert self.which_alpha in ['undirected', 'directed_ratio', 'directed_ratio_directed']
        if self.which_alpha == 'directed_ratio_directed':
            return self.latent_inverse_K_edges
        else:
            return (self.latent_inverse_K_edges + torch.transpose(self.latent_inverse_K_edges, 0, 1)) / 2
    
    def get_alpha(self):
#         print("entering")
        if self.which_alpha in ['0', '1', 'directed']:
            return self.alpha
        elif self.which_alpha == 'uniform':
            if self.which_CI != 'CIpower':
                return self.alpha
            else:
                assert hasattr(self, 'K_nodes_val')
                return self.K_nodes_val #modify the size? (but maybe not necessary)
        elif self.which_alpha == 'nodal':
#             return self.K_nodes
            if hasattr(self, 'K_nodes'):
                return self.K_nodes
            else:
                raise NotImplemented
        elif self.which_alpha == 'undirected':
#             print("sizes of latent_K_edges and transp(latent_K_edges) = {}, {}".format(self.latent_K_edges.size(), torch.transpose(self.latent_K_edges, 0, 1).size()))
#             K_edges = self.get_K_edges()
#             return 1 / K_edges
            inverse_K_edges = self.get_inverse_K_edges()
            return inverse_K_edges
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
#             print("sizes of latent_K_edges and transp(latent_K_edges) = {}, {}".format(self.latent_K_edges.size(), torch.transpose(self.latent_K_edges, 0, 1).size()))
            inverse_K_edges = self.get_inverse_K_edges()
            if hasattr(self, 'K_nodes'):
                K_nodes = self.K_nodes
            elif hasattr(self, 'K_nodes_val'):
                K_nodes = self.K_nodes_val #modify the size? (but maybe not necessary)
#             print("K_nodes.shape = {}, inverse_K_edges.shape = {}".format(K_nodes.shape, inverse_K_edges.shape))
            return K_nodes * inverse_K_edges #K_nodes / K_edges
        else:
            print("Problem")
            raise NotImplemented
    
#     def get_w(self):
#         if self.which_w in [None, 'directed']:
#             return self.w
#         elif self.which_w == 'undirected':
# #             print("sizes of latent_w and transp(latent_w) = {}, {}".format(self.latent_w.size(), torch.transpose(self.latent_w, 0, 1).size()))
#             return (self.latent_w + torch.transpose(self.latent_w, 0, 1)) / 2
#         else:
#             print("Problem")
#             raise NotImplemented
    
    def get_J(self):
        if self.which_w in [None, 'directed']:
            return self.J
        elif self.which_w == 'undirected':
#             print("sizes of latent_J and transp(latent_J) = {}, {}".format(self.latent_J.size(), torch.transpose(self.latent_J, 0, 1).size()))
            return (self.latent_J + torch.transpose(self.latent_J, 0, 1)) / 2
        else:
            print("Problem")
            raise NotImplemented
            
            
class RNN_CI(nn.Module, RNN_utils):
    """
    Circular Belief Propagation (also called Circular Inference model)
    Equation: Mij_{t+1} = F(Bi_t - alpha_ij Mji_t)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        #alternative to putting this piece of code in __init__: have def reset_parameters(self)  (see Internet)
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'CI'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha in ['0', '1', 'uniform']:
            self.initialize_parameter('alpha') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            self.initialize_parameter('alpha') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
#             print("self.alpha = {}".format(self.alpha[:,:,0]))
#             print("self.alpha_mask = {}".format(self.alpha_mask[:,:,0]))
#             self.alpha = nn.Parameter(self.alpha * self.alpha_mask) #we still get gradients on the masked values... (even though they are small) --> instead changing the gradients by hand (see code below)
            
        #Initialize w
#         if self.which_w is None:
#             self.initialize_parameter('w')
#         elif self.which_w == 'undirected':
#             self.initialize_parameter('latent_w')
#             self.initialize_parameter_mask('w_mask')
#         elif self.which_w == 'directed':
#             self.initialize_parameter('w')
#             self.initialize_parameter_mask('w_mask')
#         self.J = torch.atanh(2*self.w - 1)
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
        
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
        
#         self.initialize_remaining_parameters()
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
#         print("alpha = {}".format(self.get_alpha()))
#         print("alpha.size() = {}".format(alpha.size()))
        dM = self.F(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1))
        return (1-self.damping) * dM + self.damping * M
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F(self, x):
        """
        Compute F_x = F(x, w_ij)
        """
#         #1st version. Fails if x>16.7 (or x<-17) and w=1
#         logistic_x = torch.sigmoid(x)
#         int_x = (2*logistic_x - 1).mm(self.w) + (1-logistic_x)
#         return torch.log(int_x / (1-int_x))
    
        #2nd version. Fails if x>89 (or x<-16.7) and w=1
        #Fails for x>45 or x<-89 (actually for x>89 or x>104?). It linearizes F wrt w, so that the gradient wrt w computed by Pytorch in w=1 is the true one: (exp(x)-1)*(exp(x)+1)/exp(x). i.e. F(x,w) = F(x,1) + (w-1)*gradf_w(x,w) The advantage of the formula is that the value of F for w=1 is indeed x, and the gradient for w=1 is indeed (exp_x-1) * (exp_x+1) / exp_x
#         print("w.size() = {}".format(w.size()))
#         exp_x = torch.exp(x)
#         return torch.log((self.w*exp_x + 1 - self.w) / ((1-self.w)*exp_x + self.w))
#         return 2 * torch.atanh( (2*self.w-1) * torch.tanh(x/2) )
#         return 2 * torch.atanh( torch.tanh(self.J) * torch.tanh(x/2) )
        exp_x = torch.exp(x)
        w = torch.sigmoid(2*self.J)
        return torch.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))
    
    def get_B(self, Mext, M):
        return Mext + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    
class RNN_CIpower(nn.Module, RNN_utils):
    """
    CIpower = extension of Fractional BP with nodal parameter
    Equation: Mij_{t+1} = K_ij / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t) where F is applied to J_ij / K_ij
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        #alternative to putting this piece of code in __init__: have def reset_parameters(self)  (see Internet)
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'CIpower'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
#             print("self.K_nodes = {}, self.latent_inverse_K_edges = {}".format(self.K_nodes, self.latent_inverse_K_edges))
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
#             self.initialize_parameter('alpha') #size: (n_nodes, n_nodes, 1)
#             self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
            
        #Initialize w
#         if self.which_w is None:
#             self.initialize_parameter('w')
#         elif self.which_w == 'undirected':
#             self.initialize_parameter('latent_w')
#             self.initialize_parameter_mask('w_mask')
#         elif self.which_w == 'directed':
#             self.initialize_parameter('w')
#             self.initialize_parameter_mask('w_mask')
#         self.J = torch.atanh(2*self.w - 1)
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
#         self.initialize_remaining_parameters()
         
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
#         print("alpha = {}".format(self.get_alpha()))
        dM = 1 / torch.transpose(self.alpha, 0, 1) * self.G(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1))
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def G(self, x):
        """
        Same as F but with different self.w (= for CIpower / Fractional BP / Power EP / ...)
        w_ij is replaced by wtilde_ij  where wtilde / (1-wtilde) = (w/(1-w))^{1/K_edges}
        """
#         exp_x = torch.exp(x)
#         return torch.log((self.w_eff*exp_x + 1 - self.w_eff) / ((1-self.w_eff)*exp_x + self.w_eff))
        exp_x = torch.exp(x)
        w_eff = torch.sigmoid(2*self.J_eff)
        return torch.log((w_eff*exp_x + 1 - w_eff) / ((1-w_eff)*exp_x + w_eff))
#         return 2 * torch.atanh( torch.tanh(self.J_eff) * torch.tanh(x/2) ) #the previous version (3 lines above) gave sometimes numerical issues when w_eff were too close to 0 or 1 (explosion of M with the iterations) --> actually with arctanh there are also numerical issues, even if they come slightly later in the iterations

#     def get_w_eff(self):
#         """
#         Find the wtilde such that wtilde / (1-wtilde) = (w / (1-w))^{1/K_edges}
#         """
#         if not hasattr(self, 'latent_K_edges'):
#             return self.w
#         K_edges = self.get_K_edges()
# #         w = self.get_w() #or even self.w= ...?
#         w_ratio_power = (self.w / (1-self.w))**(1 / K_edges)
#         return w_ratio_power / (1 + w_ratio_power)
    
    def get_J_eff(self):
        """
        Find the wtilde such that wtilde / (1-wtilde) = (w / (1-w))^{1/K_edges}
        i.e., Jtilde = J / K_edges  (where J = 1/2 . log(w/(1-w)) )
        """
        if not hasattr(self, 'latent_inverse_K_edges'):
            return self.J
        inverse_K_edges = self.get_inverse_K_edges() #K_edges = self.get_K_edges()
#         J = self.get_J() #or even self.J= ...?
        return self.J * inverse_K_edges #self.J / K_edges


class RNN_CIpower_approx_tanh(nn.Module, RNN_utils):
    """
    CIpower_approx_tanh: Mij_{t+1} = K_ij / alpha_j . F_w_approx(Bi_t - alpha_i / K_ij . Mji_t)
    CIpower: Mij_{t+1} = K_ij / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'CIpower_approx_tanh'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        dM = 1 / torch.transpose(self.alpha, 0, 1) * self.G_approx_tanh(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1))
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def G_approx_tanh(self, x):
        """
        F without the arctanh
        """
        return 2 * torch.tanh(self.J_eff) * torch.tanh(x/2)

    def get_J_eff(self):
        """
        Find the wtilde such that wtilde / (1-wtilde) = (w / (1-w))^{1/K_edges}
        i.e., Jtilde = J / K_edges  (where J = 1/2 . log(w/(1-w)) )
        """
        if not hasattr(self, 'latent_inverse_K_edges'):
            return self.J
        inverse_K_edges = self.get_inverse_K_edges() #K_edges = self.get_K_edges()
        return self.J * inverse_K_edges #self.J / K_edges

    
class RNN_full_CIpower(nn.Module, RNN_utils):
    """
    full_CIpower: Mij_{t+1} = K_ij / alpha_j . F(Bi_t)
    CIpower: Mij_{t+1} = K_ij / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'full_CIpower'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
#             print("self.K_nodes = {}, self.latent_inverse_K_edges = {}".format(self.K_nodes, self.latent_inverse_K_edges))
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
#             self.initialize_parameter('alpha') #size: (n_nodes, n_nodes, 1)
#             self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        dM = 1 / torch.transpose(self.alpha, 0, 1) * self.G(B.unsqueeze(1))
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def G(self, x):
        """
        Same as F but with different self.w (= for CIpower / Fractional BP / Power EP / ...)
        w_ij is replaced by wtilde_ij  where wtilde / (1-wtilde) = (w/(1-w))^{1/K_edges}
        """
        exp_x = torch.exp(x)
        w_eff = torch.sigmoid(2*self.J_eff)
        return torch.log((w_eff*exp_x + 1 - w_eff) / ((1-w_eff)*exp_x + w_eff))
#         return 2 * torch.atanh( torch.tanh(self.J_eff) * torch.tanh(x/2) ) #the previous version (3 lines above) gave sometimes numerical issues when w_eff were too close to 0 or 1 (explosion of M with the iterations) --> actually with arctanh there are also numerical issues, even if they come slightly later in the iterations

    def get_J_eff(self):
        """
        Find the wtilde such that wtilde / (1-wtilde) = (w / (1-w))^{1/K_edges}
        i.e., Jtilde = J / K_edges  (where J = 1/2 . log(w/(1-w)) )
        """
        if not hasattr(self, 'latent_inverse_K_edges'):
            return self.J
        inverse_K_edges = self.get_inverse_K_edges()
        return self.J * inverse_K_edges


class RNN_full_CIpower_approx_tanh(nn.Module, RNN_utils):
    """
    full_CIpower_approx_tanh: Mij_{t+1} = K_ij / alpha_j . F_w_approx(Bi_t)
    CIpower_approx_tanh: Mij_{t+1} = K_ij / alpha_j . F_w_approx(Bi_t - alpha_i / K_ij . Mji_t)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'full_CIpower_approx_tanh'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        dM = 1 / torch.transpose(self.alpha, 0, 1) * self.G_approx_tanh(B.unsqueeze(1))
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def G_approx_tanh(self, x):
        """
        F without the arctanh
        """
        return 2 * torch.tanh(self.J_eff) * torch.tanh(x/2)

    def get_J_eff(self):
        """
        Find the wtilde such that wtilde / (1-wtilde) = (w / (1-w))^{1/K_edges}
        i.e., Jtilde = J / K_edges  (where J = 1/2 . log(w/(1-w)) )
        """
        if not hasattr(self, 'latent_inverse_K_edges'):
            return self.J
        inverse_K_edges = self.get_inverse_K_edges()
        return self.J * inverse_K_edges


class RNN_CIpower_approx(nn.Module, RNN_utils):
    """
    CIpower_approx = extension of Circular BP with a nodal parameter
    This extension is defined as an approximation of the extension of Fractional BP (= extension of CIpower)
    Equation: Mij_{t+1} = 1 / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t) where F is applied to J_ij (and not J_ij / K_edges as in CIpower)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        #alternative to putting this piece of code in __init__: have def reset_parameters(self)  (see Internet)
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'CIpower_approx'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
#             print("self.K_nodes = {}, self.latent_inverse_K_edges = {}".format(self.K_nodes, self.latent_inverse_K_edges))
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
#             self.initialize_parameter('alpha') #size: (n_nodes, n_nodes, 1)
#             self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
            
        #Initialize w
#         if self.which_w is None:
#             self.initialize_parameter('w')
#         elif self.which_w == 'undirected':
#             self.initialize_parameter('latent_w')
#             self.initialize_parameter_mask('w_mask')
#         elif self.which_w == 'directed':
#             self.initialize_parameter('w')
#             self.initialize_parameter_mask('w_mask')
#         self.J = torch.atanh(2*self.w - 1)
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
#         print("self.w_input.shape = {}".format(self.w_input.shape))
            
#         self.initialize_remaining_parameters()
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
#         print("alpha = {}".format(self.get_alpha()))
#         print("self.alpha.shape = {}".format(self.alpha.shape))
#         print("torch.transpose(self.alpha, 0, 1).shape = {}".format(torch.transpose(self.alpha, 0, 1).shape))
#         print("self.K_nodes.shape = {}".format(self.K_nodes.shape))
#         print("torch.transpose(self.K_nodes, 0, 1).shape = {}".format(torch.transpose(self.K_nodes, 0, 1).shape))
        if hasattr(self, 'K_nodes'):
            dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1))
        elif hasattr(self, 'K_nodes_val'): #uniform
            raise NotImplemented
        else:
            dM = self.F(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1)) #K_nodes = 1 (--> this model = CIpower_approx with K_nodes = 1 is equivalent to CI)
#         dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1)) #1 / self.K_nodes   #does not work if self.K_nodes is not defined, e.g., for which_alpha = undirected
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
#         print("self.w_input.shape = {}, Mext.shape = {}".format(self.w_input.shape, Mext.shape))
#         print("self.K_nodes.squeeze(1).shape = {}".format(self.K_nodes.squeeze(1).shape))
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
#         print("Mext_eff.shape = {}".format(Mext_eff.shape))
        #3. Running the model
        for i in range(100):
#             print("i = {}".format(i))
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F(self, x):
        """
        Compute F_x = F(x, w_ij)
        Defined as in the class RNN_CI
        """
        exp_x = torch.exp(x)
        w = torch.sigmoid(2*self.J)
        return torch.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))
    
    
class RNN_CIpower_approx_approx_tanh(nn.Module, RNN_utils):
    """
    CIpower_approx_approx_tanh: Mij_{t+1} = 1 / alpha_j . F_approx_tanh(Bi_t - alpha_i / K_ij . Mji_t)
    CIpower_approx: Mij_{t+1} = 1 / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t) where F is applied to J_ij (and not J_ij / K_edges as in CIpower)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'CIpower_approx_approx_tanh'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == '0':
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        elif self.which_alpha == 'undirected':
            self.initialize_parameter('latent_inverse_K_edges') #size: (n_nodes, n_nodes, 1)
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha in ['directed_ratio', 'directed_ratio_directed']:
            K = get_Knodes_Kedges_val_convergence_algo(graph, self.which_CI)
            print("initial K = {}".format(K))
            self.initialize_parameter('K_nodes', val_init=K) #size: (n_nodes, 1, 1)
            self.initialize_parameter('latent_inverse_K_edges', val_init=1/K) #size: (n_nodes, n_nodes, 1)
#             print("self.K_nodes = {}, self.latent_inverse_K_edges = {}".format(self.K_nodes, self.latent_inverse_K_edges))
            self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
        elif self.which_alpha == 'directed':
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
#             self.initialize_parameter('alpha') #size: (n_nodes, n_nodes, 1)
#             self.initialize_parameter_mask('alpha_mask') #size: (n_nodes, n_nodes, 1)
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        if hasattr(self, 'K_nodes'):
            dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F_approx_tanh(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1))
        elif hasattr(self, 'K_nodes_val'): #uniform
            raise NotImplemented
        else:
            dM = self.F_approx_tanh(B.unsqueeze(1) - self.alpha * torch.transpose(M, 0, 1)) #K_nodes = 1 (--> this model = CIpower_approx with K_nodes = 1 is equivalent to CI)
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F_approx_tanh(self, x):
        """
        Function which approximates F(x) defined in other classes
        """
        w = torch.sigmoid(2*self.J)
        return 2 * (2*w-1) * torch.tanh(x/2)
    

class RNN_full_CIpower_approx(nn.Module, RNN_utils):
    """
    full_CIpower_approx: Mij_{t+1} = 1 / alpha_j . F(Bi_t)
    CIpower_approx: Mij_{t+1} = 1 / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t) where F is applied to J_ij (and not J_ij / K_edges as in CIpower)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'full_CIpower_approx'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha in ['0', 'undirected', 'directed_ratio', 'directed_ratio_directed', 'directed']:
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        elif self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        elif self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        else:
            print("self.which_alpha = {}: not implemented".format(self.which_alpha))
            raise NotImplemented
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
            
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        if hasattr(self, 'K_nodes'):
            dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F(B.unsqueeze(1))
        elif hasattr(self, 'K_nodes_val'): #uniform
            raise NotImplemented
        else:
            dM = self.F(B.unsqueeze(1)) #K_nodes = 1 (--> this model = full_CIpower_approx with K_nodes = 1 is equivalent to full_CI)
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F(self, x):
        """
        Compute F_x = F(x, w_ij)
        Defined as in the class RNN_CI
        """
        exp_x = torch.exp(x)
        w = torch.sigmoid(2*self.J)
        return torch.log((w*exp_x + 1 - w) / ((1-w)*exp_x + w))
    
    
class RNN_full_CIpower_approx_approx_tanh(nn.Module, RNN_utils):
    """
    full_CIpower_approx_approx_tanh: Mij_{t+1} = 1 / alpha_j . F_approx_tanh(Bi_t)
    CIpower_approx_approx_tanh: Mij_{t+1} = 1 / alpha_j . F_approx_tanh(Bi_t - alpha_i / K_ij . Mji_t)
    CIpower_approx: Mij_{t+1} = 1 / alpha_j . F(Bi_t - alpha_i / K_ij . Mji_t) where F is applied to J_ij (and not J_ij / K_edges as in CIpower)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'full_CIpower_approx_approx_tanh'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == ['0', 'undirected', 'directed_ratio', 'directed_ratio_directed', 'directed']:
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        else:
            print("self.which_alpha = {}: not implemented".format(self.which_alpha))
            raise NotImplemented
            
        #Initialize w
        if self.which_w is None:
            self.initialize_parameter('J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
        
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        if hasattr(self, 'K_nodes'):
            dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F_approx_tanh(B.unsqueeze(1))
        elif hasattr(self, 'K_nodes_val'): #uniform
            raise NotImplemented
        else:
            dM = self.F_approx_tanh(B.unsqueeze(1)) #K_nodes = 1 (--> this model = full_CIpower_approx_approx_tanh with K_nodes = 1 is equivalent to full_CI_approx_tanh)
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F_approx_tanh(self, x):
        """
        Function which approximates F(x) defined in other classes
        """
        w = torch.sigmoid(2*self.J)
        return 2 * (2*w-1) * torch.tanh(x/2)
    

class RNN_rate_network(nn.Module, RNN_utils):
    """
    Rate network = just like full_CIpower_approx_approx_tanh, unless the connection weight is not bounded between -1 and 1
    
    rate_network: Mij_{t+1} = 1 / alpha_j . F_approx_tanh_no_constraints(Bi_t) where F_approx_tanh_no_constraints(x) = 2 * J * torch.tanh(x/2)
    full_CIpower_approx_approx_tanh: Mij_{t+1} = 1 / alpha_j . F_approx_tanh(Bi_t) where F_approx_tanh(x) = 2 * (2*w-1) * torch.tanh(x/2) for w = sig(2*J), i.e. (2*w-1) = tanh(J) -->  F_approx_tanh(x) = 2 * torch.tanh(J) * torch.tanh(x/2)
    """
    def __init__(self, fixed_parameters, graph, which_alpha, which_w, which_Mext, damping=0, print_free_params=False):
        super().__init__()
        
        self.fixed_parameters = fixed_parameters
        self.graph = graph
        self.which_CI = 'rate_network'
        self.which_alpha = which_alpha
        self.which_w = which_w
        self.which_Mext = which_Mext
        self.damping = damping
        self.n_nodes = len(graph)
        self.n_edges = graph.size() #number of unoriented edges

        #Useful variables for the mask
        w_structure = torch.Tensor(get_w_matrix(graph))
        self.mask_structure = torch.where(w_structure != 0.5, torch.ones_like(w_structure), torch.zeros_like(w_structure))
        
        #Initializes parameters, to fixed_parameters[param] if the parameter is fixed, or to a particular value (Before: at random uniformly in the given interval)
        
        #Initialize alpha
        if self.which_alpha == ['0', 'undirected', 'directed_ratio', 'directed_ratio_directed', 'directed']:
            print("Impossible model - not defined")
            raise NotImplemented #not defined in this case - I haven't thought yet about what could be proposed
        if self.which_alpha == '1':
            print("BP model - call which_CI = 'BP' instead of {}".format(self.which_CI))
            raise NotImplemented #not defined in this case - just have K_nodes_val = 1 with requires_grad = False
        if self.which_alpha == 'uniform':
            self.initialize_parameter('K_nodes_val') #size: (1, 1, 1)
        elif self.which_alpha == 'nodal':
            self.initialize_parameter('K_nodes') #size: (n_nodes, 1, 1)
        else:
            print("self.which_alpha = {}: not implemented".format(self.which_alpha))
            raise NotImplemented
            
        #Initialize w WITH INITIAL VALUE TANH(J), NOT J
        if self.which_w is None:
            self.initialize_parameter('J', 'tanh_J')
        elif self.which_w == 'undirected':
            self.initialize_parameter('latent_J', 'tanh_J')
            self.initialize_parameter_mask('J_mask')
        elif self.which_w == 'directed':
            self.initialize_parameter('J', 'tanh_J')
            self.initialize_parameter_mask('J_mask')
            
        #Initialize w_input
        self.initialize_parameter('w_input') #size: (n_nodes, 1, 1)
        
        self.list_parameter_names_model = self.get_list_param_names()
        self.list_free_parameter_names_model = self.get_list_free_param_names()
        if print_free_params:
            print("List of free parameters: {}".format(self.list_free_parameter_names_model))
        
    def F_next(self, Mext, M):
        """
        Get M_{t+1}, based on Mext_{t} and M_{t}
        """
        B = self.get_B(Mext, M)
        if hasattr(self, 'K_nodes'):
            dM = 1 / torch.transpose(self.K_nodes, 0, 1) * self.F_approx_tanh_no_constraints(B.unsqueeze(1))
        elif hasattr(self, 'K_nodes_val'): #uniform
            raise NotImplemented
        else:
            dM = self.F_approx_tanh_no_constraints(B.unsqueeze(1)) #K_nodes = 1 (--> this model = full_CIpower_approx_approx_tanh with K_nodes = 1 is equivalent to full_CI_approx_tanh)
        return (1-self.damping) * dM + self.damping * M
    
    def get_B(self, Mext, M):
        if not hasattr(self, 'K_nodes'):
            return Mext + torch.sum(M, dim=0)
        return Mext / self.K_nodes.squeeze(1) + torch.sum(M, dim=0) #Mext + torch.sparse.sum(M, dim=0)
    
    def forward(self, Mext):
        """
        Simulating the model (from Ls=0 to Lfinal)
        """
        #1. Initialization of M
        M = torch.zeros(size=(self.n_nodes, self.n_nodes)).unsqueeze(2)
        #2. Change Mext into w_input * M_ext
        Mext_eff = self.w_input.squeeze(1) * Mext #effective M_ext
        #3. Running the model
        for i in range(100):
            M = self.F_next(Mext_eff, M)
        B = self.get_B(Mext_eff, M)
        return B
    
    def F_approx_tanh_no_constraints(self, x):
        """
        Function which approximates F(x) defined in other classes
        
        No constraints on the connectivity (= not bounded)
        It is equivalent to approximating tanh(J) with J in the formula
        """
#         w = torch.sigmoid(2*self.J)
#         return 2 * (2*w-1) * torch.tanh(x/2)
        return 2 * self.J * torch.tanh(x/2) #because 2*w-1 = 2*sig(2*J)-1 = tanh(J) ~ J
    

dict_models = {'CI': RNN_CI,
               'CIpower': RNN_CIpower,
               'CIpower_approx_tanh': RNN_CIpower_approx_tanh,
               'full_CIpower': RNN_full_CIpower,
               'full_CIpower_approx_tanh': RNN_full_CIpower_approx_tanh,
               'CIpower_approx': RNN_CIpower_approx,
               'CIpower_approx_approx_tanh': RNN_CIpower_approx_approx_tanh,
               'full_CIpower_approx': RNN_full_CIpower_approx,
               'full_CIpower_approx_approx_tanh': RNN_full_CIpower_approx_approx_tanh,
               'rate_network': RNN_rate_network
              }
list_available_models = ['BP'] + list(dict_models.keys())

    
    
def fit_pytorch(graph, X_train_df, y_train_df,
                which_CI, which_alpha, which_w, which_Mext,
                damping=0, k_max=None,
                X_test_df=None, y_test_df=None,
                verbose=False, run_all_computations=False,
                learning_rate=1e-3, algo='Rprop', max_nepochs=1000, which_loss='CE',
                stopping_criterion='stable_validation_loss'
               ):
    """
    Fit with Pytorch
    
    If run_all_computations = False, then just fit (and do not compute anything else during the fitting). In this case, do not provide X_test_df and y_test_df
    
    Hyper-parameters: learning_rate, algo, nepochs, which_loss (+ stopping_criterion)
    """
#     print(learning_rate, algo, max_nepochs, which_loss)
    
    assert stopping_criterion in [None, 'stable_validation_loss'] #None means that we use all the max_nepochs
    
    run_all_computations = run_all_computations or stopping_criterion == 'stable_validation_loss'
    
    assert which_CI in list_available_models, str(which_CI) + " is not among the available models"
    assert k_max is None #not implemented
    
    if run_all_computations == False:
        assert X_test_df is None and y_test_df is None, "Do not provide X_test_df and y_test_df as input: they are not needed"
    
    if run_all_computations:
        assert (X_test_df is not None) and (y_test_df is not None), "Provide X_test_df and y_test_df as input to fit_pytorch" #X_test_df and y_test_df are needed for stopping_criterion = 'stable_validation_loss'
    
    if verbose:
        print("learning_rate =", learning_rate)
    assert algo in ['Rprop', 'Adam']
    
    do_evaluation = run_all_computations
    save_all_evaluations_and_parameters = False #False by default. Useful mostly to inspect why loss = nan
#     return_accuracy = True #plots the accuracy (computed once in a while on the whole session)
    
    #Transforming the data into Python variables
    X_train = data_to_pytorch_variables(X_train_df.T) #M_ext
    y_train = data_to_pytorch_variables(y_train_df.T) #p_1_true
    if which_loss in ['KL', 'MSE']:
        X_train = X_train.double()
        y_train = y_train.double()
    if run_all_computations:
        X_test = data_to_pytorch_variables(X_test_df.T) #M_ext
        y_test = data_to_pytorch_variables(y_test_df.T) #p_1_true
        if which_loss in ['KL', 'MSE']:
            X_test = X_test.double()
            y_test = y_test.double()
    if verbose:
        print("X_train.size = {}".format(X_train.size()))
        print("y_train.size = {}".format(y_train.size()))

    if verbose:
        print("which_CI = {}, which_alpha = {}, which_w = {}, which_Mext = {}".format(which_CI, which_alpha, which_w,
                                                                                      which_Mext))

    #Choosing the right model (and initializing the parameters)
    model = get_model(graph, which_CI, which_alpha, which_w, which_Mext, damping=damping, verbose=verbose)
    if verbose:
        print(model.get_alpha().cpu().detach().numpy()[:,:,0])

    optimizer = get_optimizer(model, learning_rate, algo=algo) #'Rprop' #'Adam'
        
    list_parameter_names_model = model.get_list_param_names()
    list_free_parameter_names_model = model.get_list_free_param_names()
    if verbose:
        print("model", model)
        print("list_parameter_names_model", list_parameter_names_model)
        print("list_free_parameter_names_model", list_free_parameter_names_model)
        print("sizes of parameters: {}".format({param_name: val.size() for (param_name, val) in model.named_parameters()}))

    #Initialization of arrays
    if run_all_computations:
        if save_all_evaluations_and_parameters:
            evaluations_tried_sesssion, parameters_tried_session = [], []
        likelihood_eval_all = [] #likelihood of the full dataset (evaluated once in a while)
#         if return_accuracy:
#             accuracy_all = [] #accuracy on the full dataset (evaluated once in a while)
        list_param_all = [] #parameters (not all, only once in a while)

    ########### Optimisation #####################
    if verbose:
        print("Starting optimization...")
    # list_param = [param for param in model.parameters()] #[param.item() for param in model.parameters()]
    # print("Initial parameters: "+str(np.round(list_param,3)))
    model.train()
    if run_all_computations:
        gradients_all_epochs = []
        loss_all_epochs = []
        loss_test_all_epochs = []

    start_time = time.time()
    no_convergence = False
    
    for epoch in range(max_nepochs):
#         print("epoch = {} out of {}".format(epoch, max_nepochs))
        
        model.J = model.get_J() #model.w = model.get_w()
        if which_CI in ['CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh']:
            model.J_eff = model.get_J_eff() #model.w_eff = model.get_w_eff()
        model.alpha = model.get_alpha()

        #Simulate the model
        Lfinal = model(X_train)

        #Compute the loss based on Lfinal
        loss = model.get_loss_categorization(Lfinal, y_train, which_loss=which_loss)
        if run_all_computations:
#             print("epoch = {}: training loss = {}".format(epoch, loss.detach().item()))
            loss_all_epochs.append(loss.detach().item()) #training loss
    
        if save_all_evaluations_and_parameters:
            evaluations_tried_session.append(loss.detach().item())
    #         list_param = []
    #         for param in model.parameters():
    #             list_param.append(param.item())
            parameters_tried_session.append(list_param)

        if torch.isnan(loss.data):
            print("Optimisation failed: loss=nan (for epoch = {})".format(epoch))
            #plot the loss
#             plt.plot(loss_all_epochs, label='train')
#             plt.plot(loss_test_all_epochs, label='test')
#             plt.legend()
#             plt.title("loss")
#             plt.xlabel("iteration")
#             plt.show()
            params = {param_name: param_val.detach().numpy()[:,:,0].copy() for (param_name, param_val) in model.named_parameters() if param_val.requires_grad}
            print(params)
#             sys.exit() #stops the whole program
#             break #stops the for loop (= goes to another session)
            print("---> running again with less epochs (stop just before breaking)") #from what I observed, loss=nan because M explodes, creating numerical errors; it comes from the effective weights being too strong, i.e., |K_edges| being too low (or equivalently, |inverse_K_edges| being too big. Instead of stopping the fitting just before the nan, an alternative would be to start with different initial conditions or even to constraint K_edges to be not too big, but I prefer not to go into that: lots of fine tuning, and it's not even sure that the problems would disappear...
            max_nepochs = epoch - 2 #I think epoch - 1 would be enough; it doesn't matter too much anyway
            return fit_pytorch(graph, X_train_df, y_train_df, 
                               which_CI, which_alpha, which_w, which_Mext,
                               damping=damping, k_max=k_max,
                               X_test_df=X_test_df, y_test_df=y_test_df, 
                               verbose=verbose, run_all_computations=run_all_computations,
                               learning_rate=learning_rate, algo=algo, max_nepochs=max_nepochs, which_loss=which_loss,
                               stopping_criterion=stopping_criterion
                              )
            
        ######## Zero gradients, perform a backward pass, and update the weights ###########
        optimizer.zero_grad() #no difference with model.zero_grad()
        loss.backward() #doesn't work if multiprocessing (in function fit_models_list + Cuda available)

        #### Multiply the gradients by masks, to prevent indices which don't correspond to an edge to be trained #####
        #### (remark: nothing to do with w_input: only with alpha and J) #############################################
        if which_alpha in ['undirected', 'directed_ratio', 'directed_ratio_directed']:
            model.latent_inverse_K_edges.grad = model.latent_inverse_K_edges.grad * model.alpha_mask #model.latent_K_edges.grad = model.latent_K_edges.grad * model.alpha_mask
        elif which_alpha in ['directed']:
    #         print("model.alpha.grad.size() = {}, model.alpha_mask.size() = {}".format(model.alpha.grad.size(), model.alpha_mask.size()))
    #         print("--> (model.alpha.grad * model.alpha_mask).size() = {}".format((model.alpha.grad * model.alpha_mask).size()))
            model.alpha.grad = model.alpha.grad * model.alpha_mask
        if which_w in ['undirected']:
            model.latent_J.grad = model.latent_J.grad * model.J_mask #model.latent_w.grad = model.latent_w.grad * model.w_mask
        elif which_w == 'directed':
            model.J.grad = model.J.grad * model.J_mask #model.w.grad = model.w.grad * model.w_mask

        if run_all_computations:
            #Save the gradients with respect to the parameters of the model
#             for param_name, param_val in model.named_parameters():
#                 if param_val.requires_grad:
#                     print(param_name, param_val.grad.detach().numpy())
#             gradients_all = [param.grad.data.item() for param in model.parameters() if param.requires_grad] #i.e. for all parameters in list_free_parameters_model
            gradients_all = {param_name: param_val.grad.detach().numpy()[:,:,0].copy() for (param_name, param_val) in model.named_parameters() if param_val.requires_grad} #i.e. for all parameters in list_free_parameters_model
#         #     print(np.round(gradients_all,3))
#             print("gradients sizes:", {key: val.shape for key, val in gradients_all.items()})
            gradients_all = {key: (val[:,0] if key in ['K_nodes'] else val) for key, val in gradients_all.items()}
            gradients_all_epochs.append(gradients_all)

            #save the current parameters of the model
            params = {param_name: param_val.detach().numpy().copy() for (param_name, param_val) in model.named_parameters()} #maybe add "if param_val.requires_grad" (to keep only the fitted parameters)
            list_param_all.append(params)

        optimizer.step()

        loss.detach_()
        
        #stopping criterion
        if run_all_computations:
            with torch.no_grad():
                Lfinal_test = model(X_test)
                loss_test = model.get_loss_categorization(Lfinal_test, y_test, which_loss=which_loss)
                loss_test_all_epochs.append(loss_test.detach().item())
                loss_test.detach_()
                if stopping_criterion == 'stable_validation_loss':
                    percent_satur = 0.05 if which_loss != 'MSE' else 0.5
                    if epoch >= 50 and is_saturating(loss_test_all_epochs, percent=percent_satur):
                        print("Early stopping - after {} epochs".format(epoch))
                        break
        
        if epoch == max_nepochs - 1 and stopping_criterion == 'stable_validation_loss':
            print("No early stopping - the fitting took {} epochs and the validation loss didn't converge".format(max_nepochs))
            #plots do not show up because of multiprocessing
#             plt.plot(loss_all_epochs, label='train')
#             plt.plot(loss_test_all_epochs, label='test')
#             plt.legend()
#             plt.title("loss")
#             plt.xlabel("iteration")
#             plt.show()
            no_convergence = True #convergence of the validation loss
            
    if verbose:
        print("time elapsed", time.time() - start_time)

    class Object(object):
        pass
    res = Object()
        
    alpha_obj, w_obj, w_input = model.to_obj()
    res.alpha_obj = alpha_obj
    res.w_obj = w_obj
    res.w_input = w_input
    
    if no_convergence:
        res.no_convergence = True
        #saving also the history of losses (on the train and test set) in order to plot it (impossible to do here as it's in a starmap)
        res.loss_all_epochs = loss_all_epochs
        res.loss_test_all_epochs = loss_test_all_epochs
    
    if run_all_computations:
        #evaluate the model
        model.eval()

        model.J = model.get_J() #model.w = model.get_w()
        if which_CI in ['CIpower', 'CIpower_approx_tanh', 'full_CIpower', 'full_CIpower_approx_tanh']:
            model.J_eff = model.get_J_eff() #model.w_eff = model.get_w_eff()
        model.alpha = model.get_alpha()

        #training set
        Lfinal_train = model(X_train)
        y_predict_train_np = sigmoid(Lfinal.detach().numpy())
        y_true_train_np = y_train.numpy()
    #     loss_train_hand = cross_entropy(y_predict_train_np, y_true_train_np)
    #     print(loss_train_hand)
        loss_train = model.get_loss_categorization(Lfinal_train, y_train, which_loss=which_loss).item()

        #test set
        Lfinal_test = model(X_test)
        y_predict_test_np = sigmoid(Lfinal_test.detach().numpy())
        y_true_test_np = y_test.numpy()
    #     loss_test_hand = cross_entropy(y_predict_test_np, y_true_test_np)
    #     print(loss_test_hand)
        loss_test = model.get_loss_categorization(Lfinal_test, y_test, which_loss=which_loss).item()

        if verbose:
            print("loss (train) = {}, loss (test) = {}".format(loss_train, loss_test))
            
            plt.scatter(y_true_train_np, y_predict_train_np)
            plt.xlabel('p(x=1) (true)')
            plt.xlabel('p(x=1) (predict)')
            plt.title("training set")
            plt.show()

            plt.scatter(y_true_test_np, y_predict_test_np)
            plt.xlabel('p(x=1) (true)')
            plt.xlabel('p(x=1) (predict)')
            plt.title("test set")
            plt.show()
        
        res.loss_train = loss_train
        res.loss_test = loss_test
        res.y_predict_train_np = y_predict_train_np
        res.y_predict_test_np = y_predict_test_np
        res.gradients_all_epochs = gradients_all_epochs
        res.loss_all_epochs = loss_all_epochs
        res.loss_test_all_epochs = loss_test_all_epochs
        res.list_param_all = list_param_all
        res.model = model
        
    return res
