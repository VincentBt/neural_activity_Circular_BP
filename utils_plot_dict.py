from compute_effects_CI_vs_BP import *
from graph_generator import generate_graph
import numpy as np
from utils_CI_BP import *
# from utils_graph_rendering import *
from pprint import pprint
import networkx as nx
import bct

def plot_dict(d):
    for node,val in d.items():
        plt.plot(val)
    plt.show()

def plot_M_ext(M_ext, show_one_node=True, show_legend=False):
    """
    Note that the keys of M_ext might not include all graph nodes. If the node is represented as a key, it means that the external signal arriving to this node is zero.
    """
    if show_one_node: #showing M_ext for one of the nodes
        some_node = list(M_ext.keys())[0]
        plt.plot(M_ext[some_node])
        plt.title('M_ext (for an example node)')
        plt.show()

    #showing M_ext for all nodes
    plt.figure(figsize=(14,6))
    for node, M_ext_node in M_ext.items():
        plt.plot(M_ext_node, label='node ={}'.format(node))
#     plt.axhline(y=0, linestyle='--', color='black')
    plt.xlabel('time (iteration of BP/CI)', size=20)
    plt.ylabel('M_ext', size=20)
    plt.title('M_ext (for all nodes)', size=15)
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})
    plt.show()

def plot_B_history_CI(B_history_CI, show_legend=False):
    plt.figure(figsize=(14,6))
    for node, val in B_history_CI.items():
        plt.plot(val, label='node = {}'.format(node))
    plt.xlabel('time (iteration of BP/CI)', size=20)
    plt.ylabel('Belief (log-odds)', size=20)
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 15})
    plt.show()
    
def plot_B_history_BP_B_history_CI(B_history_BP, B_history_CI):
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for node,val in B_history_BP.items():
        plt.plot(val)
    plt.xlabel('time (iteration of BP)', size=15)
    plt.ylabel('Belief (log-odds)')
    plt.title('BP')
    plt.subplot(1,2,2)
    for node,val in B_history_CI.items():
        plt.plot(val)
    plt.xlabel('time (iteration of CI)', size=15)
    plt.ylabel('Belief (log-odds)')
    plt.title('CI')
    plt.show()
    
def plot_activations_history_CI(activations_history_CI):
    #plotting the activation history
    for node,val in activations_history_CI.items():
    #     plt.plot(val[times_hallu_all])
        plt.plot(val)
    #     plt.plot(val[times_hallu_all].reshape(-1,5).T)
    plt.title('activation history (CI)')
    plt.show()