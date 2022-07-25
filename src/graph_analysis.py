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
        chunk_len_threshold= 10000
        return_dict= decompose.decompose_scalable(self.graph, self.graph_nx, self.hw_depth, final_output_node_set, self.leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode, chunk_len_threshold)
        self.BB_graph= return_dict['BB_graph'] 
        self.BB_graph_nx= return_dict['BB_graph_nx'] 
      
        list_of_depth_lists= hw_details.list_of_depth_lists
        #hw_details_str= out_mode + '_' + str(fitness_wt_in) + str(fitness_wt_out) + str(fitness_wt_distance) + '_'.join([''.join(str(y) for y in x) for x in list_of_depth_lists])
        #print hw_details_str 
        fname= create_decompose_file_name(hw_details, decompose_param_obj)
        
        # Misc. information to store for next steps
        misc= {}
  #      misc['best_hw_set']= return_dict['best_hw_set']

        # files_parser.write_BB_graph_and_main_graph(self.global_var, self.hw_depth, fname, self.graph, self.graph_nx, self.BB_graph, self.BB_graph_nx, misc)

        logger.info("Done step 1 : Decomposition into blocks")
        sys.setrecursionlimit(20000)
        
        # Do Step 2, 3, and 4 of the compiler
        schedule_param_obj= schedule_param(hw_details.SCHEDULING_SEARCH_WINDOW, hw_details.RANDOM_BANK_ALLOCATE)

        # decompose_fname= create_decompose_file_name(hw_details, decompose_param_obj)
    
        # -- read decompose outputs
        # self.BB_graph, self.graph, self.BB_graph_nx, self.graph_nx, misc = \
        #     files_parser.read_BB_graph_and_main_graph(self.global_var, self.hw_depth, decompose_fname)
      
        # -- perform scheduling
        # map_param_to_addr, key: leaf node, val: (MEMORY address, bank)
        # map_output_to_addr, key: head node, val: (register bank, pos), since there is no store at the end
        instr_ls_obj, map_param_to_addr, map_output_to_addr= scheduling_gather.instruction_gen(net, self.graph, self.BB_graph, self.global_var,\
            self.leaf_list, self.head_node, misc, hw_details,\
            write_asm= False, make_vid= False)  

        tot_reg_rd=0
        tot_reg_wr=0
        for bb in list(self.BB_graph.values()):
          #assert len(bb.out_list) <= 4
          tot_reg_rd += len(bb.in_list_unique)
          tot_reg_wr += len(bb.out_list)
        
        print('Reg_rd:', tot_reg_rd, 'Reg_wr:', tot_reg_wr)

        #reporting_tools.reporting_tools.write_c_for_asip_2('./LOG/asip.c', self.BB_graph, self.graph, 16, self.leaf_list)
        #reporting_tools.reporting_tools.write_c_for_asip('./LOG/vect_' + str(self.net)+'_gcc.c', self.graph)
        #reporting_tools.reporting_tools.write_vect_c('./LOG/vect_'+ str(self.net) +'.c', self.graph, self.BB_graph)
        
        # logger.info("Not writing schedule to any file.")
        # exit(0)

        schedule_fname= create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj)
        # files_parser.write_schedule(self.global_var, self.hw_depth, schedule_fname, list(instr_ls_obj.instr_ls), map_param_to_addr, map_output_to_addr)


        # schedule_param_obj= schedule_param(hw_details.SCHEDULING_SEARCH_WINDOW, hw_details.RANDOM_BANK_ALLOCATE)

        # schedule_fname= create_schedule_name(hw_details, schedule_param_obj, decompose_param_obj)
        # instr_ls, map_param_to_addr, map_output_to_addr = src.files_parser.read_schedule(self.global_var, self.hw_depth, schedule_fname)
        # assert self.head_node in map_output_to_addr

        # # reduction in instruction length becuase of the automatic write address generation
        # prog_len= 0
        # prog_len_wo_opt= 0
        # for instr in instr_ls:
        #   instr_len, instr_len_wo_opt = instr.len_w_and_wo_auto_wr_addr(hw_details.max_depth, hw_details.n_banks, hw_details.reg_bank_depth)
        #   prog_len += instr_len
        #   prog_len_wo_opt += instr_len_wo_opt

        # logger.info(f"net, {self.net}, d, {hw_details.max_depth}, b, {hw_details.n_banks}, r, {hw_details.reg_bank_depth}, Program length (bits), {prog_len}, Program length w/o optimization, {prog_len_wo_opt}, reduction, {(prog_len_wo_opt - prog_len) / prog_len_wo_opt}")

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

    if mode=="ac_eval":
      verif_helper.init_leaf_val(self.graph, mode='all_1s')
      return_val= ac_eval.ac_eval(self.graph, self.head_node, elimop= 'PROD_BECOMES_SUM')
      
      print('AC eval:', return_val)
      exit(0)

  def count_edges(self, graph_nx, list_of_partitions, skip_leafs= False):
    map_n_to_pe= {}

    for pe, partitions in enumerate(list_of_partitions):
      for partition in partitions:
        for n in partition:
          assert n not in map_n_to_pe
          map_n_to_pe[n]= pe

    global_edges= 0
    local_edges= 0
    for src, dst in graph_nx.edges():
      if src in map_n_to_pe: # not a leaf
        if map_n_to_pe[src] == map_n_to_pe[dst]:
          local_edges += 1
        else:
          global_edges += 1
      else: # leaf
        if not skip_leafs:
          local_edges += 1

    logger.info(f"total_edges: {graph_nx.number_of_edges()}, global_edges: {global_edges}, local_edges: {local_edges}")

  def process_data(self, verbose=0):
    #self.process_depth_data()
    self.avg_node_reuse = analysis_node_reuse._node_reuse_profile(self.graph, self.head_node)
    if (verbose):
      print("Avg. node reuse: ", self.avg_node_reuse)
  
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

def par_for_sptrsv(name_ls, thread_ls, log_path, openmp_prefix, suffix):
  line_number= 49
  run_log= open(log_path, 'a+')

  cmd= "cd src/openmp/; make set_env"
  os.system(cmd)

  for mat in name_ls:
    for th in thread_ls:
      mat = mat.replace('/', '_')
      data_path= openmp_prefix + f"{mat}_{suffix}_{th}.c"
      data_path = data_path.replace('/', '\/')
      openmp_main_file= "./src/openmp/par_for_sparse_tr_solve_coarse.cpp"
      cmd= f"sed -i '{line_number}s/.*/#include \"{data_path}\"/' {openmp_main_file}"
      os.system(cmd)
      cmd= "cd src/openmp; make normal_cpp"
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {mat}, {th}")
        print(f"{mat},{th},Error compilation", file= run_log, flush= True)
      else:
        logger.info("Excuting 1k iterations of parallel code...")
        cmd= "cd src/openmp; make run"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        msg= f"{mat},{th},{output}"
        print(msg, file= run_log, flush= True)
        logger.info(f"Run statistics: {msg}")
        logger.info(f"Adding result to log file: {log_path}")
    

def par_for_psdd(name_ls, thread_ls, log_path, openmp_prefix, suffix):
  line_number= 8
  run_log= open(log_path, 'a+')

  cmd= "cd src/openmp/; make set_env"
  os.system(cmd)
  for net in name_ls:
    for th in thread_ls:
      data_path= f"{openmp_prefix}{net}_{suffix}_{th}.c" 
      data_path = data_path.replace('/', '\/')
      openmp_main_file= "./src/openmp/par_for_v2.cpp"
      cmd= "sed -i '8s/.*/#include \"" + data_path + f"\"/' {openmp_main_file}"
      logger.info(f"Modifying main openmp file: {openmp_main_file} to include the header file {data_path}")
      print(cmd)
      os.system(cmd)
      cmd= "cd src/openmp; make normal_cpp_psdd"
      err= os.system(cmd)
      if err:
        print(f"Error in compilation {net}, {th}")
        print(f"{net},{th},Error compilation", file= run_log, flush= True)
      else:
        logger.info("Excuting 10k iterations of parallel code...")
        cmd= "cd src/openmp; make run_psdd"
        output= subprocess.check_output(cmd, shell=True)
        # os.system(cmd)
        output = str(output)
        output = output[:-3]
        output= output[output.find('N_layers'):]
        msg= f"{net},{th},{output}"
        print(msg, file= run_log, flush= True)
        logger.info(f"Run statistics: {msg}")
        logger.info(f"Adding result to log file: {log_path}")

