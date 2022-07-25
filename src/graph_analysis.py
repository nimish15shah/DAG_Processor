#=======================================================================
# Created by         : KU Leuven
# Filename           : graph_analysis.py
# Author             : Nimish Shah
# Created On         : 2019-10-21 16:59
# Last Modified      : 
# Update Count       : 2019-10-21 16:59
# Description        : 
#                      
#=======================================================================

import csv
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import csv
import global_var
import pickle
import queue
import copy
import time
import os
import random
import math
import sys
from termcolor import colored
from collections import defaultdict
import re
import logging
import numpy as np
import networkx as nx
import cProfile
from scipy.sparse import linalg
from statistics import mean
import subprocess

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#**** imports from our codebase *****
from . import common_classes
from . import reporting_tools
from . import graph_init
from . import useful_methods
from . import ac_eval
from . import decompose
from . import scheduling_gather
from . import files_parser
from . import hw_struct_methods
from . import verif_helper
from . import psdd
from . import sparse_linear_algebra_main
from . import write_binary
from . import plots
from . import design_explore
from . import get_sizes

def clog2(num):
  """
    clog2(2) = 1
    clog2(32) = 5
  """
  assert num > 0
  return len(bin(num-1)) - 2

#********
# Calls methods to generate superlayers
#********
class graph_analysis_c():  
  """ Main class for AC analysis.
  Important attributes of the AC are stored in the object that are initialized in constructor
  """
  def __init__(self):
    self.global_var= global_var
    
    self.name= "graph_analysis_c"
    
    # Graph that captures the AC structure. KEY= AC node_id =Line number of the operation in (binarized) AC, VAL: Node data structure
    # Created in graph_init.py
    self.graph= {}
    
    # A networkx DiGraph to represent AC structure
    # This object should be used when networkx features are to be used on the AC
    # Created in graph_init.py
    self.graph_nx= None

    self.ac_node_list= []
    self.leaf_list= [] # Lists all leaf node ids
    self.head_node= 0  # ID of the final node of the AC

  def test(self, args):
    mode= args.tmode

    if mode == "null":
      pass
      exit(0)

    if mode == "compile":
      # self.hw_depth= int(args.targs[0])
      # max_depth= int(args.targs[1])
      # min_depth= int(args.targs[2])
      # # fitness_wt_distance= float(args.targs[3])
      # out_mode= args.targs[3] #'TOP_2'
      # reg_bank_depth= int(args.targs[4])
      # fitness_wt_distance_scale= float(args.targs[5])
      # assert out_mode in ['ALL','VECT' , 'TOP_1', 'TOP_2']

      self.hw_depth = clog2(args.banks)
      max_depth = args.depth
      min_depth = min(max_depth, 2)
      reg_bank_depth = args.regs
      out_mode = 'ALL'
 
      nets_d= {
        # 'wilt'            : ("psdd"   , 150.0) ,
        'tretail'            : ("psdd"   , 150.0) ,
        'mnist'              : ("psdd"   , 50.0 ) ,
        'nltcs'              : ("psdd"   , 225.0) ,
        'msweb'              : ("psdd"   , 250.0) ,
        'msnbc'              : ("psdd"   , 125.0) ,
        'bnetflix'           : ("psdd"   , 275.0) ,
        'HB_bp_200'          : ("sptrsv" , 225.0) ,
        'HB_west2021'        : ("sptrsv" , 225.0) ,
        'MathWorks_Sieber'   : ("sptrsv" , 300.0) ,
        'HB_jagmesh4'        : ("sptrsv" , 200.0) ,
        'Bai_rdb968'         : ("sptrsv" , 175.0) ,
        'Bai_dw2048'         : ("sptrsv" , 150.0) ,
        # 'pigs'       :  ("ac",            150.0 ) , 
        # 'andes'      :  ("ac",            150.0 ) , 
        # 'munin3_spu' :  ("ac",            50.0  ) , 
        # 'mildew_spu' :  ("ac", 100.0 ) , 
      }

      for net, tup in nets_d.items():
        cir_type = tup[0]
        fitness_wt_distance_scale = tup[1]
        global_var.BB_FILE_PREFIX        = global_var.INSTR_PATH + net + '_bb.p_'
        global_var.GRAPH_FILE_PREFIX     = global_var.INSTR_PATH + net + '_gr.p_'

        config_obj = common_classes.ConfigObjTemplates()
        config_obj.graph_init(name= net, cir_type= cir_type, graph_mode= "FINE")
        self.graph, self.graph_nx, self.head_node, self.leaf_list, other_obj = graph_init.get_graph(global_var, config_obj)
      
        out_port_mode = 'PAPER_MODE'
        hw_details= hw_struct_methods.hw_nx_graph_structs(self.hw_depth, max_depth, min_depth)
        critical_path_len = nx.algorithms.dag.dag_longest_path_length(self.graph_nx)
        hw_details.init_pru_sync_params(reg_bank_depth, critical_path_len, fitness_wt_distance_scale)
        hw_details.print_details()
        
        decompose_param_obj= decompose_param(hw_details.max_depth, hw_details.min_depth, hw_details.fitness_wt_distance, hw_details.fitness_wt_in, hw_details.fitness_wt_out)
        
        n_outputs= None
        final_output_node_set = set([self.head_node])
        chunk_len_threshold= 100000
        return_dict= decompose.decompose_scalable(self.graph, self.graph_nx, self.hw_depth, final_output_node_set, self.leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode, chunk_len_threshold)
        self.BB_graph= return_dict['BB_graph'] 
        self.BB_graph_nx= return_dict['BB_graph_nx'] 
      
        list_of_depth_lists= hw_details.list_of_depth_lists
        
        logger.info("Done step 1 : Decomposition into blocks")
        sys.setrecursionlimit(20000)
        
        # Do Step 2, 3, and 4 of the compiler
        schedule_param_obj= schedule_param(hw_details.SCHEDULING_SEARCH_WINDOW, hw_details.RANDOM_BANK_ALLOCATE)

        # -- perform scheduling
        instr_ls_obj, map_param_to_addr, map_output_to_addr= scheduling_gather.instruction_gen(net, self.graph, self.BB_graph, self.global_var,\
            self.leaf_list, self.head_node, misc, hw_details,\
            write_asm= False, make_vid= False)  

        schedule_fname= create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj)

        #===========================================
        #       Binary generation
        #===========================================
        prefix= global_var.INSTR_PATH + net.replace('/','_') + schedule_fname
        fname= prefix + '_instr.txt'
        
        instr_ls = instr_ls_obj.instr_ls
        write_binary.write_binary(fname, self.global_var, instr_ls, hw_details, verb=False)

        # parameter initialization and golden output files
        map_addr_to_val= verif_helper.pru_sync(self.graph, prefix, hw_details, instr_ls, map_param_to_addr, map_output_to_addr, write_files = True)
      exit(0) 

    if mode=="plot_charts":
      # Plot instruction breakdown
      hw = design_explore.HwConf(8, 3, 2, 32)
      log_obj = design_explore.LogInfo(hw)
      log_d= {}
      log_d[hw.tuple_for_key()] = log_obj
      design_explore.get_instr_stat(global_var.REPORTS_PATH + 'instr_breakup.txt', log_d)

      path= global_var.PLOTS_PATH + 'instruction_breakdown.pdf'
      savefig= True
      tup = (8, 3, 32)
      design_explore.plot_instr_stat_all_w(log_d, tup, savefig, path)

      # Plot throughput
      log_path= global_var.REPORTS_PATH + 'rtl_latency.txt'
      plot_path= global_var.PLOTS_PATH + 'throughput.pdf'
      savefig= True
      plot_throughput(log_path, plot_path, log_d, savefig)

      exit(0)

  def _add_node(self, graph, node_key, child_key_list, op_type):
    # Add the node and update it's child list
    node_obj= common_classes.node(node_key)
    for item in child_key_list:
      node_obj.add_child(item)
    node_obj.set_operation(op_type)
    
    graph[node_obj.get_self_key()]= node_obj
    
    # Update parent list of the child
    for item in child_key_list:
      graph[item].add_parent(node_key)
  
  def create_file_name(self, hw_depth, max_depth, min_depth, out_mode, fitness_wt_in, fitness_wt_out, fitness_wt_distance):
    hw_details= hw_struct_methods.hw_nx_graph_structs(hw_depth, max_depth, min_depth)
    list_of_depth_lists= hw_details.list_of_depth_lists

    hw_details_str= out_mode + '_' + str(fitness_wt_in) + str(fitness_wt_out) + str(fitness_wt_distance) + '_'.join([''.join(str(y) for y in x) for x in list_of_depth_lists])

    return hw_details_str  


  def create_file_name_full(self, args):
    assert len(args.targs) == 5, len(args.targs) 
    hw_depth= int(args.targs[0])
    self.hw_depth= hw_depth
    max_depth= int(args.targs[1])
    min_depth= int(args.targs[2])
    fitness_wt_distance= float(args.targs[3])
    out_mode= args.targs[4] 
    
    print("hw_depth", hw_depth)
    print("max_depth", max_depth)
    print("min_depth", min_depth)
    print("out_mode", out_mode)

    assert out_mode in ['ALL','VECT' , 'TOP_1', 'TOP_2']

    fitness_wt_in= 0.0
    fitness_wt_out= 0.0

    print('fitness_wt_distance: ', fitness_wt_distance)

    return self.create_file_name(hw_depth, max_depth, min_depth, out_mode, fitness_wt_in, fitness_wt_out, fitness_wt_distance)


class schedule_param():
  def __init__(self, SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE):
    self.SCHEDULING_SEARCH_WINDOW=SCHEDULING_SEARCH_WINDOW
    self.RANDOM_BANK_ALLOCATE= RANDOM_BANK_ALLOCATE

  def str_fname(self):
    fname= ""
    fname += str(self.SCHEDULING_SEARCH_WINDOW)
    fname += str(self.RANDOM_BANK_ALLOCATE)
    return fname

class decompose_param():
  def __init__(self, max_depth, min_depth, fitness_wt_distance, fitness_wt_in, fitness_wt_out):
    self.max_depth= max_depth
    self.min_depth= min_depth
    self.fitness_wt_distance= fitness_wt_distance
    self.fitness_wt_in= fitness_wt_in
    self.fitness_wt_out= fitness_wt_out

  def str_fname(self):
    fname= ""
    fname += str(self.max_depth)
    fname += str(self.min_depth)
    fname += str(self.fitness_wt_distance)
    fname += str(self.fitness_wt_in)
    fname += str(self.fitness_wt_out)
    return fname

def str_with_hw_details(hw_details):
  fname= ""
  fname += "D" + str(hw_details.tree_depth)
  fname += "_N" + str(hw_details.n_tree)
  fname += "_" + str(hw_details.n_banks)
  fname += "_" + str(hw_details.reg_bank_depth)
  fname += "_" + str(hw_details.mem_bank_depth)
  fname += "_" + str(hw_details.mem_addr_bits)
  fname += "_" + str(hw_details.n_bits)
  fname += "_" + str(hw_details.n_pipe_stages)

  return fname

def create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj):
  """
    Creates file name for schedule (or instr_ls) using hw_details_class obj and schedule_param obj
  """

  fname= str_with_hw_details(hw_details)
  fname += decompose_param_obj.str_fname()
  fname += schedule_param_obj.str_fname()

  return fname

def create_decompose_file_name(hw_details, decompose_param_obj):
  """
    Creates file name for output of decompose process using hw_details_class obj and schedule_param obj
  """
 
  fname = ""
  fname += str(hw_details.tree_depth)
  fname += str(hw_details.n_tree)
  fname += decompose_param_obj.str_fname()

  return fname

def get_latency(path, log_d, mode= 'post_rtl_sim'):
  assert mode in ['post_rtl_sim', 'post_compile']

  if mode == 'post_rtl_sim':
    with open(path, 'r') as fp:
      data = csv.reader(fp, delimiter=',')
      data= list(data)

    #idx
    name_idx= 1
    n_tree_idx         = 3
    tree_depth_idx     = 5
    n_banks_idx        = 7
    reg_bank_depth_idx = 9
    latency_idx        = 19

    for d in data:
      if len(d) < latency_idx:
        logger.info("weird line")
        continue

      name           = d[name_idx].strip()
      n_tree         = int(d[n_tree_idx         ])
      tree_depth     = int(d[tree_depth_idx     ])
      n_banks        = int(d[n_banks_idx        ])
      reg_bank_depth = int(d[reg_bank_depth_idx ])
      latency        = int(float(d[latency_idx        ]))

      tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

      if tuple_for_key in log_d:
        obj= log_d[tuple_for_key]
        obj.map_workload_to_latency[name] = latency
      else:
        logger.info(f"n_tree, tree_depth_idx, reg_bank_depth of : {tuple_for_key} is not available in log_d")

  elif mode == 'post_compile':
    with open(path, 'r') as fp:
      data = csv.reader(fp, delimiter=',')
      data= list(data)  

    #initialize latency to 'inf'
    for _, obj in log_d.items():
      obj.map_workload_to_latency = defaultdict(lambda : float('inf'))

    #idx
    name_idx            = 1
    tree_depth_idx      = 3
    n_banks_idx         = 5
    reg_bank_depth_idx  = 7
    total_idx           = 9
    bb_idx              = 11
    initial_ld_idx      = 13
    intermediate_ld_idx = 15
    intermediate_st_idx = 17
    shift_idx           = 19
    nop_idx             = 21
    fitness_wt_distance_idx= 23

    for d in data:
      if len(d) < fitness_wt_distance_idx:
        continue

      name                = d[name_idx].strip()
      tree_depth          = int(d[tree_depth_idx     ])
      n_banks             = int(d[n_banks_idx        ])
      reg_bank_depth      = int(d[reg_bank_depth_idx ])
      total               = int(d[total_idx        ])
      bb              = int(d[bb_idx             ])
      initial_ld      = int(d[initial_ld_idx     ])
      intermediate_ld = int(d[intermediate_ld_idx])
      intermediate_st = int(d[intermediate_st_idx])
      shift           = int(d[shift_idx          ])
      nop             = int(d[nop_idx            ])
      fitness_wt_distance= float(d[fitness_wt_distance_idx])
      
      assert total == bb + initial_ld + intermediate_st + intermediate_ld + shift + nop

      n_tree = int(n_banks / (2**tree_depth))
      tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

      if tuple_for_key in log_d:
        obj= log_d[tuple_for_key]
        if total < obj.map_workload_to_latency[name]:
          obj.map_workload_to_latency[name] = total
      else:
        pass
        # logger.info(f"n_tree, tree_depth_idx, reg_bank_depth of : {tuple_for_key} is not available in log_d")

    # Assert
    workloads= Workloads().workloads
    workloads= [w.replace('/', '_') for w in workloads]
    for tuple_for_key, obj in log_d.items():
      for w in workloads:
        assert w in obj.map_workload_to_latency, f"paths.INSTR_STAT_PATH does not contain a log for {w} for {tuple_for_key} configuration"
  else:
    assert 0

def plot_throughput(log_path, plot_path, log_d, savefig):
  get_latency(log_path, log_d, mode= 'post_rtl_sim')

  map_workload_to_compute = get_sizes.get_psdd_sizes()
  map_mat_to_details = get_sizes.get_matrix_sizes()
  for key, obj in map_mat_to_details.items():
    map_workload_to_compute[key] = obj

  map_w_to_throughput_this_work = {w : map_workload_to_compute[w].compute/(3.3*lat) for w, lat in log_d[(8, 3, 32)].map_workload_to_latency.items()} # GOPS

  system= []
  throughput= []
  workload= []

  for w, t in map_w_to_throughput_this_work.items():
    system.append('This work')
    throughput.append(t)
    workload.append(w)

  # DPU throughput GOPS
  map_w_to_througput_DPU= {
     'tretail'            :3.2279686469855693,
     'mnist'              :5.714442493415277 ,
     'nltcs'              :4.254714624703385 ,
     'msweb'              :5.497139773340529 ,
     'msnbc'              :4.85417179424071  ,
     'bnetflix'           :4.80075056728923  ,
     'HB_bp_200'          :2.1350198110332217,
     'HB_west2021'        :2.74123043712898  ,
     'MathWorks_Sieber'   :3.90464757331504  ,
     'HB_jagmesh4'        :1.9240511974312091,
     'Bai_rdb968'         :1.8668534369059206,
     'Bai_dw2048'         :2.577926343448042 ,
  }
  for w, t in map_w_to_througput_DPU.items():
    system.append('DPU')
    throughput.append(t)
    workload.append(w)

  data_d= {
    'system' : system,
    'throughput': throughput,
    'workload' : workload,
  }
  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (2.8, 2.4)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  sns.set_style({'font.family' : 'STIXGeneral'})

  order= [
    'tretail'         ,
    'mnist'           ,
    'nltcs'           ,
    'msweb'           ,
    'msnbc'           ,
    'bnetflix'        ,
    'HB_bp_200'       ,
    'HB_west2021'     ,
    'MathWorks_Sieber',
    'HB_jagmesh4'     ,
    'Bai_rdb968'      ,
    'Bai_dw2048'      ,
  ]

  hue_order= ['This work', 'DPU']
  ax= sns.barplot(x= 'workload', y= 'throughput', hue= 'system', orient= 'v', data=df, order= order, hue_order = hue_order,
      ci= None, # No error margins
      estimator= mean
      )

  ax.set_ylim([0,10])

  ax.set_xticks(list(range(len(order))))
  ax.set_xticklabels(
    [
      'tretail'         ,
      'mnist'           ,
      'nltcs'           ,
      'msweb'           ,
      'msnbc'           ,
      'bnetflix'        ,
      'bp_200'       ,
      'west2021'     ,
      'sieber',
      'jagmesh4'     ,
      'rdb968'      ,
      'dw2048'      ,
    ]
  )
  
  plt.xticks(rotation = 90)

  # ax.set_ylim([0,119])
  # labels= [4, 8, 16, 32, 64]
  # h, l= ax.get_legend_handles_labels()
  # ax.legend(h, labels, title= "Threads")
  # ax.legend(h, l, title= "")
  ax.legend(ncol= 2, loc= 'upper center')
  # ax.legend(ncol= 1, bbox_to_anchor= (1.2, 0.5))
  # ax.legend([], [])
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

