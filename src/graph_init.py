
import time
import re
import queue
import networkx as nx
import logging

#**** imports from our codebase *****
from . import common_classes
from . import useful_methods
from . import psdd
from . import sparse_linear_algebra_main

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_graph(global_var, config_obj):
  #----GRAPH CONSTRUCTION
  other_obj= {}

  if config_obj.cir_type == 'psdd':
    graph, graph_nx, head_node, leaf_list, _ = psdd.main(global_var.DATA_PATH + config_obj.name + ".psdd")
    assert nx.algorithms.components.is_weakly_connected(graph_nx)
  
  elif config_obj.cir_type == 'sptrsv':
    tr_solve_obj= sparse_linear_algebra_main.SparseTriangularSolve(global_var, config_obj.name, write_files= False, verify=False, read_files= True, graph_mode= config_obj.graph_mode, output_mode= 'single_node')

    if config_obj.graph_mode == 'FINE':
      graph= tr_solve_obj.L_graph_obj.graph
      graph_nx= tr_solve_obj.L_graph_obj.graph_nx
    elif config_obj.graph_mode == 'COARSE':
      graph= tr_solve_obj.L_coarse_graph_obj.graph
      graph_nx= tr_solve_obj.L_coarse_graph_obj.graph_nx
    else:
      assert 0
    
    head_ls= useful_methods.get_head_ls(graph)
    head_node = head_ls.pop()

    leaf_list= useful_methods.get_leaves(graph_nx)

    other_obj['tr_solve_obj'] = tr_solve_obj

  else:
    assert 0

  # Mark leaves as computed
  for n, obj in graph.items():
    if obj.is_leaf():
      assert n != head_node
      obj.computed= True

  # Sanity check
  assert nx.algorithms.dag.is_directed_acyclic_graph(graph_nx)

  # precompute reverse and normal levels, some functions accept this to be precomputed
  map_v_to_reverse_lvl= useful_methods.compute_reverse_lvl(graph_nx)
  map_v_to_lvl= useful_methods.compute_lvl(graph_nx)
  for n, obj in graph.items():
    obj.reverse_level = map_v_to_reverse_lvl[n]
    obj.level = map_v_to_lvl[n]

  #verify
  for node, obj in list(graph.items()):
    for child in obj.child_key_list:
      assert graph[child].reverse_level < obj.reverse_level
    
    for parent in obj.parent_key_list:
      assert graph[parent].reverse_level > obj.reverse_level

  logger.info(f"Critical path length: {nx.algorithms.dag.dag_longest_path_length(graph_nx)}")

  return graph, graph_nx, head_node, leaf_list, other_obj

def construct_graph(self, args, global_var):
  verbose= args.v
  mode= args.tmode

  #----GRAPH CONSTRUCTION
  if args.cir_type == 'psdd':
    self.graph, self.graph_nx, self.head_node, self.leaf_list, self.ac_node_list = psdd.main(global_var.PSDD_PATH)

  # Mark leaves as computed
  for n, obj in self.graph.items():
    if obj.is_leaf():
      obj.computed= True

  # Sanity check
  assert nx.algorithms.dag.is_directed_acyclic_graph(self.graph_nx)
  assert nx.algorithms.components.is_weakly_connected(self.graph_nx)

  #--- POST-PROCESSING---
  # Mark level, reverse_level and number of nodes under the given node
  if mode== 'hw_tree_blocks' or mode=='vectorize_inputs_of_building_blocks' or mode== 'try':
    compute_reverse_level(self.graph, self.head_node)
    compute_level(self.graph, self.head_node)
    
    # Verify reverse level
    graph=self.graph
    for node, obj in list(graph.items()):
      for child in obj.child_key_list:
        assert graph[child].reverse_level < obj.reverse_level
      
      for parent in obj.parent_key_list:
        assert graph[parent].reverse_level > obj.reverse_level
  
  logger.info(f"Critical path length: {nx.algorithms.dag.dag_longest_path_length(self.graph_nx)}")


def construct_graph_from_ac(self, args, global_var):
  verbose= args.v
  mode= args.tmode
  
  # Creating op_list from AC file  
  op_list, self.leaf_list, self.head_node= reorder_operations(self.use_ac_file, global_var)
  
  # Create graph from op_list
  self.graph = _create_graph_from_op_list(self, op_list, global_var)
  self.ac_node_list= list(self.graph.keys())
  
  if 'float' in mode or 'fixed' in mode or 'error' in mode or mode== 'output_min_max' or  mode == 'adaptively_optimize_for_Fxpt' or mode == 'munin_single_query_verify' or mode == 'max_FixPt_err_query':
    # Creating BN graph from BN file (*.net)
    net_fp= open(global_var.NET_FILE, 'r')
    net_content= net_fp.readlines()
    self.BN= read_net(net_content)
    
    # Read lmap file and update the state of object accordingly
    lmap_fp= open(global_var.LMAP_FILE, 'r')
    lmap_content= lmap_fp.readlines()
    read_lmap(self, lmap_content)
   

  
  # Create networkx (nx) graph for AC
  self.graph_nx= useful_methods.create_nx_graph_from_node_graph(self.graph)
  
def compute_level(graph, head_node):
  highest_level= 0
  graph[head_node].level= 0
  
  open_set= queue.Queue()
  open_set.put(head_node)

  closed_set= []
  while not open_set.empty():
    curr_node = open_set.get()
    obj= graph[curr_node]
    
    child_level= obj.level + 1
    for child in obj.child_key_list:
      if graph[child].level == None:
        graph[child].level= child_level
        open_set.put(child)
        if child_level > highest_level:
          highest_level= child_level
    
      elif graph[child].level < child_level: 
        # Same thing as above
        graph[child].level= child_level
        open_set.put(child)
        if child_level > highest_level:
          highest_level= child_level

    closed_set.append(curr_node)
  
  # Offset levels such that
  for node, obj in list(graph.items()):
    obj.level= highest_level- obj.level
  
  # Sanity Check
  for node, obj in list(graph.items()):
    for child in obj.child_key_list:
      assert graph[child].level < obj.level
    
    for parent in obj.parent_key_list:
      assert graph[parent].level > obj.level

def compute_reverse_level(graph, curr_node):
  obj= graph[curr_node]
  
  if obj.reverse_level is None:
    if not obj.child_key_list: # leaf node
      obj.reverse_level= 0
    else: # Not a leaf
      max_level= 0
      for child in obj.child_key_list:
        lvl= compute_reverse_level(graph, child)
        if lvl > max_level:
          max_level= lvl

      obj.reverse_level= max_level + 1

  return obj.reverse_level
