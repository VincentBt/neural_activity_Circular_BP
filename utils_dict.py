import matplotlib.pyplot as plt
import numpy as np
# from utils_save_fig import save_fig

def sort_dict_by_value(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    
    
def plot_dict_ordered(d, d_order=None, ylabel=None, title_fig=None, ylim=None, savefig=False, 
                      type_graph=None, colored_keys=None):
    """
    colored_keys is used to make a list of keys (for instance stimulated nodes here) having a different color
    """
    if savefig != False:
        assert type_graph is not None
    plt.figure(figsize=(20,10))
    width = 0.35
    if d_order == None:
        #sort the keys of d by value
        d = sort_dict_by_value(d) #sort
    else:
        #sort the keys of d using the same order as d_order (where keys of d_order are first sorted by value)
        list_keys = list(sort_dict_by_value(d_order).keys())
        d = {key: d[key] for key in list_keys}
    x_pos = np.array(range(len(d)))
    if colored_keys is None:
        plt.bar(x_pos, list(d.values()), width=width,
        #         yerr=list(d_error_BP.values()), 
                align='center', alpha=0.5, ecolor='black', capsize=10)
    else: #color a few nodes in red instead of blue
#         list_colors = ['red']*10 + ['black']*(len(x_pos)-10)
        list_colors = ['red' if key in colored_keys else 'black' for key in d.keys()]
        plt.bar(x_pos, list(d.values()), width=width,
        #         yerr=list(d_error_BP.values()), 
                align='center', alpha=0.5, ecolor='black', capsize=10, color=list_colors)
    plt.xticks(x_pos, labels=list(d.keys()), rotation=90, size=12) #add ticks on the x-axis
    if ylabel != None:
        plt.ylabel(ylabel, size=15) #'% overactivation'
    if title_fig != None:
        plt.title(title_fig, size=15)
    if ylim != None:
        #remove some of the highest bars
        n_remove = len(list_regions_trigger_hallu)
#         print(sorted(list(d.values())))
        maxi_others = sorted(list(d.values()))[-n_remove-1]
        mini = np.min(list(d.values()))
#         plt.ylim(ylim[0], ylim[1]) #useful if some values of the dict are very high compared to others (e.g. because of pulse on some regions)
        plt.ylim(mini - 1, maxi_others + 1) #useful if some values of the dict are very high compared to others (e.g. because of pulse on some regions)
    if savefig != False:
        if savefig == True:
            print("give a name of the fig to be saved (variable savefig)")
        if isinstance(savefig, str):
            save_fig(savefig, type_graph) #save_fig("FigS6", type_graph)
    plt.show()
    

def plot_dict_ordered_with_additional_curve(d, graph, ylabel=None, list_measures=None, 
                                            savefig=False, type_graph=None, colored_keys=None):
    """
    Plots the overactivation (bars) and graph measures (lines) on top of each other
    Uses fun_measures (dict of functions) and function get_legend_measures
    
    #Matplotlib example: how to have 2 y-axis
    # Generate data
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    """
    if savefig != False:
        assert type_graph is not None
    from utils_graph_measures import fun_measures, get_legend_measures
    d = sort_dict_by_value(d) #sort 
    
    fig, ax1 = plt.subplots(figsize=(18,8)) # plt.figure()
    width = 0.35
    x_pos = np.array(range(len(d)))
    if colored_keys is None:
        color = 'black'
    else:
        color = ['red' if key in colored_keys else 'black' for key in d.keys()] #list of colors
    ax1.bar(x_pos, list(d.values()), width=width,
    #         yerr=list(d_error_BP.values()), 
            align='center', alpha=0.5, ecolor='black', color=color, capsize=10, label='% overactivation')
    plt.ylabel(ylabel, size=15) #plt.ylabel('Overactivation (%)', size=15)
    plt.xticks(x_pos, labels=list(d.keys()), rotation=90, size=12) #add ticks on the x-axis
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for name_measure in list_measures:
        fun_measure = fun_measures[name_measure]
    #     measure_nodes = {mapping_nodes[node]: graph.fun_measure(node) for node in graph.nodes} #with region names
        measure_nodes = fun_measure(graph)
        measure_nodes = {key: val for key,val in measure_nodes.items()} #{mapping_nodes[key]: val for key,val in measure_nodes.items()}
    #     print(measure_nodes)
        measure_nodes_good_order = {key:measure_nodes[key] for key in d.keys()} #to have the same order of keys
    #     plt.plot(x_pos, list(measure_nodes_good_order.values()), label=name_measure)
    #     ax2.plot(x_pos, np.array(list(measure_nodes_good_order.values())) / list(measure_nodes_good_order.values())[0], label=get_legend_measures(name_measure), linewidth=3) #"normalized" so that the most overactive region has measure 1 (normalization)
        y_measures = np.array(list(measure_nodes_good_order.values()))
        ax2.plot(x_pos, (y_measures - np.min(y_measures)) / (np.max(y_measures) - np.min(y_measures)), label=get_legend_measures(name_measure), linewidth=3) #normalized so that each metric goes between 0 and 1
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel('node metrics (normalized and centered)', size=15)
    plt.xticks(x_pos, labels=list(measure_nodes_good_order.keys()), rotation=90, size=12)
    plt.legend(prop={"size":15})
    if 'realistic_connectome' not in graph.type_graph:
        plt.xticks([])
    if savefig != False:
        if savefig == True:
            print("give a name of the fig to be saved (variable savefig)")
        if isinstance(savefig, str):
            save_fig(savefig, type_graph) #save_fig("FigS6", type_graph)
    plt.show()

    
def plot_2_dicts_ordered(d_all, d_order=None, ylabel=None, ylim=None, colored_keys=None):
    """
    colored_keys is used to make a list of keys (for instance stimulated nodes here) having a different color
    """
    d1, d2 = list(d_all.values())
    label1, label2 = list(d_all.keys())
    assert set(list(d1.keys())) == set(list(d2.keys()))
    plt.figure(figsize=(20,10))
    width = 0.35
    if d_order == None:
        #sort the keys of d1 and d2 by depending on the ordering of d1
        d1 = sort_dict_by_value(d1) #sort
        d2 = {key:d2[key] for key in d1.keys()}
    else:
        #sort the keys of d using the same order as d_order (where keys of d_order are sorted by value)
        list_keys = list(sort_dict_by_value(d_order).keys())
        d1 = {key:d1[key] for key in list_keys}
        d2 = {key:d2[key] for key in list_keys}
    x_pos = np.array(range(len(d1)))
    if colored_keys is None:
        plt.bar(x_pos - width/2, list(d1.values()), width=width,
                align='center', alpha=0.5, ecolor='black', capsize=10, label=label1)
        plt.bar(x_pos + width/2, list(d2.values()), width=width,
                align='center', alpha=0.5, ecolor='black', capsize=10, label=label2)
    else: #color a few nodes in red instead of blue
        list_colors = ['red' if key in colored_keys else 'blue' for key in d1.keys()]
        plt.bar(x_pos - width/2, list(d1.values()), width=width,
                align='center', alpha=0.5, ecolor='black', capsize=10, color=list_colors, label=label1)
        plt.bar(x_pos + width/2, list(d2.values()), width=width,
                align='center', alpha=0.5, ecolor='black', capsize=10, color=list_colors, label=label2)
    plt.xticks(x_pos, labels=list(d1.keys()), rotation=90, size=12) #add ticks on the x-axis
    if ylabel != None:
        plt.ylabel(ylabel, size=20)
#     if ylim != None:
#         #remove some of the highest bars
#         n_remove = len(list_regions_trigger_hallu)
# #         print(sorted(list(d.values())))
#         maxi_others = sorted(list(d.values()))[-n_remove-1]
#         mini = np.min(list(d.values()))
# #         plt.ylim(ylim[0], ylim[1]) #useful if some values of the dict are very high compared to others (e.g. because of pulse on some regions)
#         plt.ylim(mini - 1, maxi_others + 1) #useful if some values of the dict are very high compared to others (e.g. because of pulse on some regions)
    plt.legend(prop={'size': 20})
    plt.show()
    

def get_keys_max_val(d):
    """
    Gets keys corresponding to the max value of the dictionnary 
    If there are multiple highest values (equality), np.argmax returns one element anyway
    
    Example:
    d = {'a':2, 'c':6, 'b':3, 'd':5}
    get_keys_max_val(d)
    """
    keys = np.array(list(d.keys()))
    vals = np.array(list(d.values()))
    return keys[np.argmax(vals)]

def get_keys_min_val(d, ignore_error_nan=False):
    """
    Gets keys corresponding to the min value of the dictionnary 
    If there are multiple highest values (equality), np.argmin returns one element anyway
    
    Example:
    d = {'a':12, 'c':6, 'b':3, 'd':5}
    get_keys_min_val(d)
    """
    keys = np.array(list(d.keys()))
    vals = np.array(list(d.values()))
    #filter nan values
    isnan = np.isnan(vals)
    if np.all(isnan) and ignore_error_nan:
        return 'error_all_nan' #see function get_best_alpha_list_methods in utils_compensating_alpha.py
    keys = keys[~isnan]
    vals = vals[~isnan]
    return keys[np.argmin(vals)]


def transf_list_of_dict(list_of_dict):
    """
    Transforms list of dict into dict of list
    
    Example: list_of_dict = [{'a': 1, 'b': 2}, {'a':3, 'b':4}]
    """
    list_keys = [list(el.keys()) for el in list_of_dict]
    assert list_keys.count(list_keys[0]) == len(list_keys), "Problem with the keys - all elements should have train and val"
    dict_of_list = {}
    for key in list_keys[0]:
#         print(key)
        dict_of_list[key] = [el[key] for el in list_of_dict]
    return dict_of_list