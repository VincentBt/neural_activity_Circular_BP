# Graph rendering
# With hierarchical structure (as defined with 'up' and 'down') and colors corresponding to the difference between the beliefs under CI and BP
# See graph_rendering.ipynb

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import bct
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def plot_directed_graph(graph):
    G_directed = G_from_graph(graph)  #in order to plot the edge directions; use plot_graph_old with method_pos='directed' or 'undirected' 
    plot_graph_old(G_directed, 'orange', method_pos='directed')
    plt.show()
    plot_graph_old(G_directed, 'orange', method_pos='undirected')
    plt.show()


def plot_G(G):
    #structural connectivity (representation of "graph")
    plot_graph_old(G, 'orange') #possible additionnal argument: method_pos='directed' or 'undirected' (if the graph is oriented)
    plt.show()

    
def plot_graph_old(G, node_color, method_pos='undirected', title=None, path_save_file=None):
    """
    Plots the graph, without the external messages
    If node_color is a list: colors nodes with the corresponding colors
    If node_color is a string: colors nodes with this color (same for all nodes)
    method_pos (directed or undirected) is the way nodes of the directed graph are represented: so that clusters can be seen (=undirected method) or so that nodes at the top of the hierarchy are at the top of the figure (=durected method).
    
    TODO: merge plot_graph_old with plot_graph_old2
    """
    plt.figure(figsize=(20,10))
    
    pos_nodes = graphviz_layout(G, prog='dot') if method_pos=='directed' else graphviz_layout(G.to_undirected(), prog='dot')
    node_size = 1200
    
    if isinstance(node_color, str):
        nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size)
        
    else: #node_color is a list
    
        if np.sum(node_color < 0) == 0: #all node_color are positive
            vmin, vmax = 0, np.max(node_color)
#             cmap = plt.cm.Reds
            cmap = plt.cm.autumn_r
        else:
            vmin, vmax = -np.max(np.abs(node_color)), np.max(np.abs(node_color))
            cmap = plt.cm.RdYlGn_r
    #     print(vmin, vmax)
        if vmin == vmax: #both are 0
            vmin, vmax = -0.1, 0.1

        nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size, vmin=vmin, vmax=vmax, cmap=cmap, width=1.5)
    
        #plot the colorbar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)#, ticks=range(len(G)))#, boundaries=np.arange(-0.05,2.1,.1))

    if title != None:
        plt.title(title)
    if path_save_file != None:
        if  ('.png' not in path_save_file) and ('.jpg' not in path_save_file) and ('.svg' not in path_save_file):
            print("adding .png to path_save_file")
            path_save_file = path_save_file + '.png'
        plt.savefig(path_save_file)
    plt.draw() #plt.show()
    
    

    

def plot_graph_old2(G, node_color, method_pos='undirected', 
                   title=None, path_save_file=None, node_sizes=None, pos_nodes=None,
                   plot_colorbar=False,
                   label_node_color='betweenness centrality'
                  ):
    """
    Plots the graph, without the external messages
    If node_color is a list: colors nodes with the corresponding colors
    If node_color is a string: colors nodes with this color (same for all nodes)
    method_pos (directed or undirected) is the way nodes of the directed graph are represented: so that clusters can be seen (=undirected method) or so that nodes at the top of the hierarchy are at the top of the figure (=directed method).
    
    TODO: merge plot_graph_old with plot_graph_old2
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111)
    if all([isinstance(x, int) for x in list(G.nodes)]):
        G = nx.relabel_nodes(G, lambda x: x + 1)
#     pos_nodes = graphviz_layout(G, prog='dot') if method_pos=='directed' else graphviz_layout(G.to_undirected(), prog='dot')
#     pos_nodes = graphviz_layout(G, prog='dot',args='-Gnodesep=3.6') if method_pos=='directed' else graphviz_layout(G.to_undirected(), prog='dot',args='-Gnodesep=3.6')
    k = 0.6 #distance between nodes
    if pos_nodes is None:
        pos_nodes = nx.drawing.layout.spring_layout(G, k=k) if method_pos=='directed' else nx.drawing.layout.spring_layout(G, k=k)
    else:
        pos_nodes = pos_nodes
    if node_sizes is None:
        node_size = 4000 if len(G) < 50 else 2000
    else:
#         node_size = node_sizes
        node_sizes_rescaled = (node_sizes - np.min(node_sizes)) / (np.max(node_sizes) - np.min(node_sizes))
        node_size = 2000 + (6500-2000) * node_sizes_rescaled #so that the size of nodes goes from 1000 to 4000
        legend_elements = [
            Line2D([0], [0], marker='o', color='black', markerfacecolor='grey', markersize=markersize)
#             Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='grey', markersize=markersize),
            for markersize in [21,26,30,33,36]
        ]
        leg = plt.legend(
            handles=legend_elements, ncol=len(legend_elements),
#             loc='lower right', bbox_to_anchor=(1., 0.2),
#             title='degree', frameon=False
            borderpad=2
        )
#         leg.get_title().set_fontsize('20')
        plt.text(0.978,0.165,'Degree of a node',
                 horizontalalignment='right', verticalalignment='bottom', 
                 transform = ax.transAxes, fontsize=25)
    font_size = 35 if len(G) < 50 else 15
    
    if isinstance(node_color, str):
        nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size, font_size=font_size)
        
    else: #node_color is a list
    
        if np.sum(node_color < 0) == 0: #all node_color are positive
            vmin, vmax = 0, np.max(node_color)
#             cmap = plt.cm.Reds
            cmap = plt.cm.autumn_r
        else:
            vmin, vmax = -np.max(np.abs(node_color)), np.max(np.abs(node_color))
            cmap = plt.cm.RdYlGn_r
    #     print(vmin, vmax)
        if vmin == vmax: #both are 0
            vmin, vmax = -0.1, 0.1

        if set(node_color) == set([1,2,3,4]): #show the modules using circles
            print("4 colors - creating colormap")
            colors_4 = ['#A8E10C', '#8A6FDF', '#FFDB15', "dodgerblue"] #green,purple,yellow,blue  (colorblind-friendly)
#             colors_4 = ['#A8E10C', '#8A6FDF', '#FFDB15', "#FF5765"] #green,purple,yellow,red #https://www.oberlo.com/blog/color-combinations-cheat-sheet - see https://wp-en.oberlo.com/wp-content/uploads/2019/07/image8-4.png
            cmap = mcolors.ListedColormap(colors_4)
#             nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=[colors_4[el-1] for el in node_color], node_size=node_size, width=1.5, font_size=font_size) #color the interior of each node
#             nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size, cmap=cmap, width=1.5, font_size=font_size) #color the interior of each node
            nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color='white', edgecolors=[colors_4[el-1] for el in node_color], node_size=node_size, width=1.5, linewidths=3, font_size=font_size) #color the outline of each node
            legend_elements = [
                Line2D([0], [0], marker='o', color='black', markerfacecolor=colors_4[i], markersize=28, label='Module '+str(i+1))
#                 Line2D([0], [0], marker='o', color='black', markerfacecolor='grey', markersize=28)
                for i in range(4)
            ]
            plt.legend(
                handles=legend_elements,
#                 loc='lower right', bbox_to_anchor=(1, 0),
                borderpad=1.6,
                prop={'size':25}, fontsize=40
            )
        elif set(node_color) == set([1,2,3]): #show the node type (connector hub, local hub, other node) using circles
            #corresp = {'connector_hub': 1, 'local_hub': 2, 'other_node': 3} #the numbers 1, 2 and 3 should be defined using corresp
            corresp_invert = {1: 'connector hub', 2:'local hub', 3:'other node'} #{1: 'connector_hub', 2:'local_hub', 3:'other_node'}
            print("3 colors - creating colormap")
            import seaborn as sns
            colors_3 = sns.color_palette('Set2')[:3]
#             colors_3 = ['#A8E10C', '#8A6FDF', '#FFDB15']#, "dodgerblue"] #green,purple,yellow,blue  (colorblind-friendly)
#             colors_3 = ['#A8E10C', '#8A6FDF', '#FFDB15', "#FF5765"] #green,purple,yellow,red #https://www.oberlo.com/blog/color-combinations-cheat-sheet - see https://wp-en.oberlo.com/wp-content/uploads/2019/07/image8-4.png
            cmap = mcolors.ListedColormap(colors_3)
#             nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=[colors_3[el-1] for el in node_color], node_size=node_size, width=1.5, font_size=font_size) #color the interior of each node
#             nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size, cmap=cmap, width=1.5, font_size=font_size) #color the interior of each node
            nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color='white', edgecolors=[colors_3[el-1] for el in node_color], node_size=node_size, width=1.5, linewidths=3, font_size=font_size) #color the outline of each node
            legend_elements = [
                Line2D([0], [0], marker='o', color='black', markerfacecolor=colors_3[i], markersize=28, label=corresp_invert[i+1])
#                 Line2D([0], [0], marker='o', color='black', markerfacecolor='grey', markersize=28)
                for i in range(3)
            ]
            plt.legend(
                handles=legend_elements,
#                 loc='lower right', 
                bbox_to_anchor=(1.25, 0.68),
                borderpad=1.6,
                prop={'size':25}, fontsize=40
                      )
        
        else:
            nx.draw(G, pos=pos_nodes, with_labels=True, arrows=True, node_color=node_color, node_size=node_size, vmin=vmin, vmax=vmax, cmap=cmap, width=1.5, font_size=font_size)    
        #plot the colorbar
        if plot_colorbar:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
#             plt.colorbar(sm)#, ticks=range(len(G)))#, boundaries=np.arange(-0.05,2.1,.1))

            axins = inset_axes(ax,
                   width="23.3%",  # width = 5% of parent_bbox width
                   height="8%",  # height : 50%
                   loc='lower right',
                   bbox_to_anchor=(-0.005, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
            cbar = plt.colorbar(sm, cax=axins, orientation='horizontal')#, ticks=[1, 2, 3])
            cbar.set_ticks([]) #cbar.set_ticks([0,0.05,0.1,0.15])
            cbar.set_label(label_node_color, rotation=0, size=25)
    if title != None:
        plt.title(title)
#     plt.draw() #plt.show()
    return pos_nodes



    
def plot_graph_with_Mext(G, M_ext, node_color, title=None, path_save_file=None, verbose=True):
    """
    Plots the graph, with arrows corresponding to external messages
    """
    
    ex_Mext = M_ext[list(M_ext.keys())[0]]
    if not(isinstance(ex_Mext, int) or isinstance(ex_Mext, float)): #case where M_ext varies with time
        M_ext_time_varying = True
    else:
        M_ext_time_varying = False
        
#     if M_ext_time_varying == True:
#         plot_graph_old(G, node_color, title=title, path_save_file=path_save_file) #without the external messages
#         return
    
    G = G.copy()
    
    #add edges for external messages
    for key, value in M_ext.items():
        G.add_edge('ext_'+str(key), key, M_ext_val=value)
    e_normal = [(u, v) for (u, v, d) in G.edges(data=True) if 'M_ext_val' not in list(d.keys())]
    e_ext_mes = [(u, v) for (u, v, d) in G.edges(data=True) if 'M_ext_val' in list(d.keys())]
    print("here", [el[0] for el in e_ext_mes])
    print()
    print("e_ext_mes")
    print(e_ext_mes)
    print("finish")
    
    #AXES
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    
    #DRAW NODE LABELS
    dict_node_labels = {node: node if 'ext_' not in str(node) else '' for node in G.nodes}
    nx.draw_networkx_labels(G, pos=graphviz_layout(G, prog='dot'), labels=dict_node_labels)

    #DRAW NODES
    node_size = 600 #1200
    max_abs = np.max(np.abs(node_color))
    vmin, vmax = -max_abs, max_abs
    if vmin == vmax: #both are 0
        vmin, vmax = -0.1, 0.1
    cmap = plt.cm.RdYlGn_r
    nx.draw_networkx_nodes(G, pos=graphviz_layout(G, prog='dot'), 
                           nodelist=[node for node in G.nodes if 'ext_' not in str(node)], 
                           node_color=node_color, node_size=node_size, vmin=vmin, vmax=vmax, cmap=cmap)

    #DRAW EDGES
    if verbose:
        print(list(M_ext.keys()))
        print(e_ext_mes)
    for edge in G.edges:
        if edge in e_ext_mes:
            print(edge)
            assert edge[1] in M_ext.keys()
    list_edges_colors = [M_ext[edge[1]] for edge in G.edges if edge in e_ext_mes]
    nx.draw_networkx_edges(G, pos=graphviz_layout(G, prog='dot'), 
                           edgelist=e_normal, node_size=node_size) #normal edges (between nodes)
#     nx.draw_networkx_edges(G, pos=graphviz_layout(G, prog='dot'), edgelist=e_ext_mes, style='dashed', width=4, arrows=True, edge_color=list_edges_colors, node_size=node_size) #external messages
    if M_ext_time_varying == False:
        nx.draw_networkx_edges(G, pos=graphviz_layout(G, prog='dot'), edgelist=e_ext_mes, width=4, edge_color=list_edges_colors, node_size=node_size, edge_vmin=-np.max(np.abs(list_edges_colors)), edge_vmax=np.max(np.abs(list_edges_colors)), edge_cmap=cmap) #external messages #style='dashed' #arrows=True
    else: #time-varying external messages: showing the external messages in black
        nx.draw_networkx_edges(G, pos=graphviz_layout(G, prog='dot'), 
                               edgelist=e_ext_mes, width=4, edge_color='black', node_size=node_size) 
        
    #plot the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)#, ticks=range(len(G)))#, boundaries=np.arange(-0.05,2.1,.1))

    if title != None:
        plt.title(title)
    if path_save_file != None:
        if  ('.png' not in path_save_file) and ('.jpg' not in path_save_file) and ('.svg' not in path_save_file):
            print("adding .png to path_save_file")
            path_save_file = path_save_file + '.png'
        plt.savefig(path_save_file)
    ax.set_axis_off() #remove the box
    plt.draw() #plt.show()


def plot_graph_graph_measure(G, measure_name, title=None, path_save_file=None):
    """
    Plots the graph, without the external messages
    The color of the nodes corresponds to the color of the measure
    """
    
    from utils_graph_measures import all_fun_measures
    
    assert measure_name in list(all_fun_measures.keys()), "{} is not among the possible graph measures".format(measure_name)
    fun_graph_measure = all_fun_measures[measure_name]
    
    dict_degree = dict(fun_graph_measure(G))
        
    assert list(G.nodes) == list(dict_degree.keys())
    list_degree = list(dict_degree.values())
    node_color = list_degree
    
    vmin, vmax = np.min(node_color), np.max(node_color)
    cmap = plt.cm.autumn_r
    nx.draw(G, pos=graphviz_layout(G, prog='dot'), 
            with_labels=True, arrows=True, node_color=node_color, node_size=800, 
            vmin=vmin, vmax=vmax, cmap=cmap)

    #plot the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)#, ticks=range(len(G)))#, boundaries=np.arange(-0.05,2.1,.1))

    if title != None:
        plt.title(title)
    if path_save_file != None:
        print(path_save_file + '.png')
        plt.savefig(path_save_file + '.png')
    plt.draw() #plt.show()
    
    
# def visualize_network(struct):
#     return