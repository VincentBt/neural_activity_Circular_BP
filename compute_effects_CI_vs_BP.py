import numpy as np
from utils_CI_BP import *
from utils_graph_rendering import *
from graph_generator import generate_graph
from generate_Mext import *
from pprint import pprint
import networkx as nx
import itertools
import dill
import os, sys
import multiprocessing
from datetime import datetime
import time
from simulate import *

# n_cpu_cores = multiprocessing.cpu_count() #number of CPU cores


# #define path_dir_save depending on the computer used
# which_computer = os.path.abspath("").split("/")[-3]
# # if which_computer == 'ENS Ulm':
# if which_computer == 'research':
#     path_dir_save = '../../results_code/simulations_CI_BP/compute_effects_CI_vs_BP/'
#     path_nextcloud = '/home/vincent/Nextcloud/'  #on my Asus laptop
#     n_parallel_processes = 4
# elif (which_computer == 'Vincent_Bouttier') and (os.path.abspath("").split("/")[-4] == 'cure2_ubuntu'): #Ubuntu in Lille
#     #save into the NextCloud server (in Windows) which synchronizes with my Asus laptop
#     path_dir_save = '/mnt/d/ESPACE DE TRAVAIL COMMUN/Vincent/NextCloud_Vincent_Bouttier/'
#     path_nextcloud = '/mnt/d/ESPACE DE TRAVAIL COMMUN/Vincent/NextCloud_Vincent_Bouttier/' #on Lille's computer
#     n_parallel_processes = 36
# else:
#     print('Computer unknown - Define path_dir_save (directory path to save the simulation files)')
#     print(os.path.abspath(""))
#     sys.exit()
    

def get_suffix_to_folder(type_graph, remove_cerebellum_and_vermis, remove_ofc):
    suffix_folder = ''
    if 'realistic_connectome' in type_graph:
        if remove_cerebellum_and_vermis:
            suffix_folder = suffix_folder + '_without_cerebellum'
        if remove_ofc:
            suffix_folder = suffix_folder + '_without_ofc'
    return suffix_folder
    
    
def spawn(filename_save, type_M_ext, type_graph, keep_history, begin, list_alphac_alphad,
          remove_cerebellum_and_vermis, remove_ofc
         ):
    np.random.seed() #to make sure that simulations are different (otherwise they are given the same random seed)
    
    binarize_realistic_connectome = 'realistic_connectome' in type_graph
    res = Simulate(type_graph=type_graph, type_M_ext=type_M_ext,
                   list_alphac_alphad=list_alphac_alphad,
                   run_also_BP=False,
                   begin=begin, keep_history=keep_history,
                   n_periods=1,
                   predict_frustration_and_bistability=False,
                   remove_cerebellum_and_vermis=remove_cerebellum_and_vermis, remove_ofc=remove_ofc,
                   binarize_realistic_connectome=binarize_realistic_connectome,
                   stimulated_nodes="all",
                   variance_Mext=30, T_period=1000, #mean_Mext='random_2',
                   method_weighting="w_bimodal", w_uniform=None
#                    print_advancement=True
                  )

    
    #Save the simulations into a file
    suffix_folder = get_suffix_to_folder(type_graph, remove_cerebellum_and_vermis, remove_ofc)
    with open(path_dir_save + type_graph + suffix_folder + '/' + filename_save + '.pkl', 'wb') as file:
        dill.dump(res, file) #Saving the object res (of type Simulate) entirely

        

if __name__ == '__main__':
    print("Running simulation")
    
    type_M_ext = 'gaussian_process_by_periods_not_all_nodes' #'gaussian_process_by_periods' #options: 'gaussian_process_by_periods', 'gaussian_process',  'poisson_process'

    type_graph = sys.argv[1]
#     type_graph = 'realistic_connectome_AAL' #options: 'modular_SW', 'realistic_connectome_AAL', 'realistic_connectome_AAL2'
    print("type_graph = {}".format(type_graph))

    keep_history = True
    begin = 0 #50
    
    remove_cerebellum_and_vermis = True
    remove_ofc = True
    if 'realistic_connectome' in type_graph:
        print("remove_cerebellum_and_vermis = {}".format(remove_cerebellum_and_vermis))
        print("remove_ofc = {}".format(remove_ofc))
    
    list_alpha_c = [0.6, 0.7, 0.8, 0.9, 1]
    list_alpha_d = [0.6, 0.7, 0.8, 0.9, 1]
#     print("list_alpha_c", list_alpha_c)
#     print("list_alpha_d", list_alpha_d)
    equal_alphac_alphad = True
    if equal_alphac_alphad == True:
#         print("Taking only alpha_c = alpha_d")
        list_alphac_alphad = [[list_alpha_c[i], list_alpha_d[i]] for i in range(len(list_alpha_c))] #= zip(list_alpha_c, list_alpha_d)?
    else:
        list_alphac_alphad = itertools.product(list_alpha_c, list_alpha_d)
    print(list_alphac_alphad)
    
    
    n = 5

    date = str(datetime.fromtimestamp(time.time()))
    date_and_hour = date.split(' ')[0] + ' ' + date.split(' ')[1][:8]
    date_and_hour = date_and_hour.replace('.', ':')
    date_and_hour = date_and_hour.replace(':', '-')

    for proc in range(n):

        print("proc {} (out of {})".format(proc + 1, n))

        start_time = time.time()
        processes = []
        for i in range(n_parallel_processes):
        #     kwargs = {"i_subject": i_subject, "i_sess": i_sess, "which_data": which_data, "save_mcmc": save_mcmc}
        #     p = multiprocessing.Process(target=spawn, args=(model_name, path_folder,), kwargs=kwargs)
            p = multiprocessing.Process(target=spawn, args=(date_and_hour + ' ' + type_M_ext + ' ' + str(proc*n_parallel_processes + i), type_M_ext, type_graph, keep_history, begin, list_alphac_alphad, remove_cerebellum_and_vermis, remove_ofc))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print('That took {} seconds'.format(time.time() - start_time))