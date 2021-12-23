from compute_effects_CI_vs_BP import *
from graph_generator import generate_graph
import numpy as np
from utils_CI_BP import *
# from utils_graph_rendering import *
from pprint import pprint
import networkx as nx
import bct



class Simulate_old:
    """
    I think that this is an old version of Simulate (see simulate.py)
    """
    def __init__(self, type_graph, type_M_ext,
                 alpha_c=None, alpha_d=None,
                 list_alphac_alphad=None,
                 run_also_BP=True,
                 begin=0, keep_history=True):
        super().__init__()
        
        #checking that some (alpha_c,alpha_d) is defined (or list_alphac_alphad)
        assert (list_alphac_alphad is not None) or ((alpha_c is not None) and (alpha_d is not None))
        
        #checking that (alpha_c,alpha_d) and list_alphac_alphad are not all defined
        assert not((list_alphac_alphad is not None) and ((alpha_c is not None) and (alpha_d is not None)))
        
        #checking that if list_alphac_alphad is given, then run_also_BP=False (i.e. run_also_BP is not True)
        assert not((list_alphac_alphad is not None) and (run_also_BP==True))
        
        
        ############################  Define the graph  ####################################
        #1. Generate an unoriented graph
        G, graph_array = generate_graph(type_graph)

        #2. Assign weights to the graph
        #Set the strength of the connections
        w = 0.65 if not('realistic_' in type_graph) else 0.55 #w = 0.65
        #Orient the edges randomly
        graph = {key : (np.random.choice([w,1-w]) , numpy.random.choice(['up', 'down'])) for key in G.edges}
        ######################################################################################################

        ##################  Generate external messages ###################################################
        list_nodes = list(G.nodes())
        n_periods = 4 if type_graph == 'realistic_connectome_AAL' else 4 if '_SW' in type_graph else sys.exit()
        print("n_periods = {}".format(n_periods))
        n_stimulated_nodes = int(len(G) / 2) #for each period (they are not necessarily the same across periods)
        M_ext = generate_M_ext(type_M_ext, list_nodes, n_stimulated_nodes=n_stimulated_nodes, 
                               n_periods=n_periods, type_graph=type_graph)
        ######################################################################################################

        
        if list_alphac_alphad is None:
        
            #Run CI
            B_CI, B_history_CI = run_CI(graph, M_ext, alpha_c, alpha_d, keep_history=keep_history)
            # updates_CI = {key:val[1:]-val[:-1] for key,val in B_history_CI.items()}
            # average_B_CI = {key: np.mean(np.abs(val)) for key,val in B_history_CI.items()}
    #         activations_history_CI = get_activations(B_history_CI, method='leaky_belief', k=k)
    #         total_activation_CI = get_total_activation(activations_history_CI, begin=begin)

            #Run BP
            if run_also_BP:
                B_BP, B_history_BP = run_BP(graph, M_ext, keep_history=keep_history)
                # updates_BP = {key:val[1:]-val[:-1] for key,val in B_history_BP.items()}
                # average_B_BP = {key: np.mean(np.abs(val)) for key,val in B_history_BP.items()}
    #             activations_history_BP = get_activations(B_history_BP, method='leaky_belief', k=k)
    #             total_activation_BP = get_total_activation(activations_history_BP, begin=begin)

        else:
            B_history_CI_all = {}
            for (alpha_c, alpha_d) in list_alphac_alphad: #the product should contain BP (alpha_c=alpha_d=1)
                #CI
                B_CI, B_history_CI = run_CI(graph, M_ext, alpha_c, alpha_d, keep_history=keep_history)
                B_history_CI_all[alpha_c, alpha_d] = B_history_CI

        
        
        self.G = G
        self.graph = graph
        self.M_ext = M_ext
        
        if list_alphac_alphad is None:
            self.B_history_CI = B_history_CI
#             self.activations_history_CI = activations_history_CI
#             self.total_activation_CI = total_activation_CI
            if run_also_BP:
                self.B_history_BP = B_history_BP
#                 self.activations_history_BP = activations_history_BP
#                 self.total_activation_BP = total_activation_BP
        else:
            self.B_history_CI_all = B_history_CI_all

    def compute_updates_CI(self, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        updates_CI = {key:val[1:]-val[:-1] for key,val in B_history_CI.items()}
        return updates_CI

    def compute_updates_BP(self):
        return self.compute_updates_CI(which='BP')
    
    def compute_average_B_CI(self, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        average_B_CI = {key: np.mean(np.abs(val)) for key,val in B_history_CI.items()}        
        return average_B_CI

    def compute_average_B_BP(self):
        return self.compute_average_B_CI(which='BP')
    
    def compute_activations_history_CI(self, method='leaky_belief', k=0.05, which='CI'):
        if which == 'CI':
            B_history_CI = self.B_history_CI
        elif which == 'BP':
            B_history_CI = self.B_history_BP
        activations_history_CI = get_activations(B_history_CI, method=method, k=k)
        if which == 'CI':
            self.activations_history_CI = activations_history_CI
        elif which == 'BP':
            self.activations_history_BP = activations_history_CI
        return activations_history_CI
    
    def compute_activations_history_BP(self, method='leaky_belief', k=0.05):
        return self.compute_activations_history_CI(method=method, k=k, which='BP')
    
    def compute_total_activation_CI(self, method='leaky_belief', k=0.05, begin=0, which='CI'):
        if not(hasattr(self, 'activations_history_'+str(which))):
            self.compute_activations_history_CI(method=method, k=k, which=which)
            if not(hasattr(self, 'activations_history_'+str(which))):
                print("pb")
        if which == 'CI':
            activations_history_CI = self.activations_history_CI
        elif which == 'BP':
            activations_history_CI = self.activations_history_BP
        total_activation_CI = get_total_activation(activations_history_CI, begin=begin)     
        if which == 'CI':
            self.total_activation_CI = total_activation_CI
        elif which == 'BP':
            self.total_activation_BP = total_activation_CI
        return total_activation_CI
        
    def compute_total_activation_BP(self, method='leaky_belief', k=0.05, begin=0):
        return self.get_total_activation_CI(method=method, k=k, begin=begin, which='BP')
    
    def check_frustration_CI(self):
        B_history_CI = self.B_history_CI
        #Filtering files for which there is frustration
        squared_updates_CI = {node:(val[1:] - val[:-1])**2 for node,val in B_history_CI.items()}
        if np.max(np.array(list(squared_updates_CI.values()))) > 3:
            print("Frustration detected (alpha_c = {}, alpha_d = {})".format(alpha_c, alpha_d))
            
    
####### Plotting functions (do not put them in the class Simulation) ##########
    
def plot_M_ext(M_ext):
    #showing M_ext for one of the nodes
    some_node = list(M_ext.keys())[0]
    plt.plot(M_ext[some_node])
    plt.show()

    #showing M_ext for all nodes
    for node, M_ext_node in M_ext.items():
        plt.plot(M_ext_node, label='node ={}'.format(node))
    plt.ylabel('M_ext', size=15)
    plt.xlabel('time (iteration of BP/CI)', size=15)
    # plt.title('M_ext (for each node - if not plotted then 0)', size=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})
    plt.show()

def plot_G(graph):
    G_directed = G_from_graph(graph)  #in order to plot the edge directions; use plot_graph_old with method_pos='directed' or 'undirected' 
    plot_graph_old(G_directed, 'orange', method_pos='directed')
    plt.show()
    plot_graph_old(G_directed, 'orange', method_pos='undirected')
    plt.show()


def plot_G_2(G):
    #structural connectivity (representation of "graph")
    plot_graph_old(G, 'orange') #possible additionnal argument: method_pos='directed' or 'undirected' (if the graph is oriented)
    plt.show()
