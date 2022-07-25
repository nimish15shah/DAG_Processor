
#----------------------
# This file contains methods that decompose an AC graph into small BBs that can be mapped to hardware
#----------------------

import random
import queue
import time
import copy
from collections import defaultdict
import networkx as nx
from statistics import mean
from typing import Mapping, MutableMapping, MutableSequence, Sequence, Iterable, List, Set, Dict

#**** imports from our codebase *****
from . import common_classes
from . import hw_struct_methods
from . import hw_struct_methods
from . import useful_methods
from .useful_methods import printcol, printlog
#** Classes **
      
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class sch_parent_info():
  def __init__(self):
    self.input_cnt= 0
    self.output_cnt=0
    self.uncomputed_cnt= 0

    self.depth= 0

class global_obj():
  """
    A class to keep all the global sets and objects at a single place while decomposition
  """
  def __init__(self, hw_depth, hw_details):
    # Children whose parents to be checked
    # To be initialized to the leaves of AC
    self.children_to_check= set()
    
    # Parents whose decompose unit to be created/updated
    self.parents_to_check= set()

    # Dict to map parents to decompose_unit 
    # Key: node key
    # Val: Obj of type decompose_unit
    self.map_parent_to_decompose_unit= {}
    
    # Dict to map schedulable parent to hw_type that it can be mapped to
    # Key: node key
    # Val: List of 2-dimensional lists (Eg. [[0,1], [1,3]]). These are the indices for list_of_hw_lists (also list_of_struct_lists)
    self.map_parent_to_hw_type= {}

    # List that describe HW
    #self.hw_details= hw_struct_methods.hw_nx_graph_structs(hw_depth)
    self.hw_details= hw_details
    assert self.hw_details.list_of_depth_lists, self.hw_details.list_of_depth_lists

    self.list_of_hw_lists= self.hw_details.list_of_hw_lists 
    
    # List that contains decompose_unit objects that map to list_of_hw_lists
    self.list_of_struct_lists= []
    for idx, hw_sets in enumerate(self.list_of_hw_lists):
      self.list_of_struct_lists.append([])
      for hw in hw_sets:
        self.list_of_struct_lists[idx].append(set())
    
    # self.BB_graph= {}

def decompose_scalable(graph, graph_nx, hw_depth, final_output_node_set, leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode, chunk_len_threshold, verbose= False):

  # create chunks/partitions
  map_v_to_reverse_lvl= useful_methods.compute_reverse_lvl(graph_nx)
  node_list, diff_list= dfs_traversal(graph_nx, map_v_to_reverse_lvl)
  diff_threshold = len(graph_nx) # should not get triggered
  out_degree_threshold = len(graph_nx) # should not get triggered
  chunks= create_chunks(node_list, diff_list, graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold)

  BB_graph= {}

  # Init state of nodes to the desired value before begining
  for obj in graph.values():
    obj.dfs_level = None
    obj.fully_consumed= False
  #--- A DFS traversal to set dfs_level: needed for _choose_most_eligible_set
  _set_dfs_level(graph_nx, graph)

  start= time.time()
  for idx, chunk in enumerate(chunks):
    subg_leaf_list = set([])
    for n in chunk:
      if graph[n].computed:
        subg_leaf_list.add(n)
      else:
        # for l in graph[n].child_key_list:
        for l in graph_nx.predecessors(n):  
          if graph[l].computed:
            subg_leaf_list.add(l)
    assert len(subg_leaf_list) != 0
    useless_leaves= set()
    for l in subg_leaf_list:
      assert graph[l].computed
      parent_in_chunk = set([p for p in graph_nx.successors(l) if p in chunk])
      if len(parent_in_chunk) == 0:
        assert graph[l].is_leaf()
        chunk.remove(l)
        useless_leaves.add(l)
    subg_leaf_list -= useless_leaves
    subg_nx = graph_nx.subgraph(set(chunk) | subg_leaf_list)
    subg = {n: graph[n] for n in subg_nx}

    subg_final_output_node_set = set([n for n in chunk if ((len(list(subg_nx.successors(n))) == 0) or (len(graph[n].parent_key_list) - len(list(subg_nx.successors(n))) != 0))\
        and (not graph[n].is_leaf())])
    assert len(subg_final_output_node_set & subg_leaf_list) == 0, f"{subg_final_output_node_set}, {subg_leaf_list}"
    assert len(subg_final_output_node_set) != 0

    # logger.info(f"subg_leaf_list: {subg_leaf_list}")
    # logger.info(f"subg_final_output_node_set: {subg_final_output_node_set}")
    # logger.info(f"chunk: {chunk}")
    decompose(subg, subg_nx, BB_graph, hw_depth, subg_final_output_node_set, subg_leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode)
    # subg_BB_graph= return_dict['BB_graph'] 
    # subg_BB_graph_nx= return_dict['BB_graph_nx'] 
    logger.info(f"Done {idx + 1} chunks out of {len(chunks)} total chunks in time: {time.time() - start}")
    # print("total bbs: ", len(subg_BB_graph))
    # for bb, obj in subg_BB_graph.items():
    #   print(bb)
    #   obj.print_unit_details()

  # POST processing
  # Create BB graph
  create_full_BB_graph(graph, graph_nx, BB_graph)
  BB_graph_nx= create_BB_NX_from_BB(BB_graph)
  

  return_dict={}
  return_dict['total_BBs']= len(BB_graph)
  return_dict['BB_graph']= BB_graph
  return_dict['BB_graph_nx']= BB_graph_nx

  print_statistics(BB_graph)

  return return_dict


def dfs_traversal(graph_nx, map_v_to_reverse_lvl: Mapping):

  curr_diff= 0
  done_nodes= set()
  node_list= []
  diff_list= []

  head_list= [n for n in graph_nx if len(list(graph_nx.successors(n))) == 0]
  head_list_copy = list(head_list) 

  # the head with the highest reverse level at the end of the list
  head_list= sorted(head_list, key= lambda x: map_v_to_reverse_lvl[x])
  assert len(head_list) != 0

  while len(head_list):    
    stack= []
    stack.append(head_list.pop()) # pops from the end of the list
    while stack:
      curr_node= stack[-1] # do not remove yet

      if curr_node in done_nodes:
        stack.pop()
        continue
        
      ch_ls= [ch for ch in graph_nx.predecessors(curr_node) if not ch in done_nodes]

      curr_diff += 1
      if len(ch_ls) == 0:
        # All ch are done already, so add to node_list
        # and pop from stack
        node_list.append(curr_node)
        assert curr_node not in done_nodes, f"{curr_node}"
        done_nodes.add(curr_node)
        diff_list.append(curr_diff)
        curr_diff = 0
        stack.pop()
      else: 
        # the ch with the highest reverse level added last to the stack
        ch_ls= sorted(ch_ls, key= lambda x: map_v_to_reverse_lvl[x])
        for ch in ch_ls:
          assert ch not in done_nodes
          stack.append(ch)
  
  assert len(node_list) == len(diff_list)
  assert len(node_list) == len(graph_nx), f"{len(node_list)}, {len(graph_nx)}"
  assert len(done_nodes)== len(graph_nx), f"{len(done_nodes)} ,{len(graph_nx)}"

  return node_list, diff_list

def create_chunks(node_list, diff_list, graph_nx, diff_threshold, chunk_len_threshold, out_degree_threshold):
  logger.info("Coarsening the graph if required")
  assert len(node_list) == len(diff_list)
  leaf_ls= useful_methods.get_leaves(graph_nx)

  chunks= []
  chunk= set([node_list[0]])
  diff_threshold_cnt= 0
  chunk_len_threshold_cnt= 0
  out_degree_threshold_cnt= 0

  for idx in range(1, len(node_list)):
    n= node_list[idx]
    d= diff_list[idx]
    # new_chunk_cond = (d >= diff_threshold) 
    # new_chunk_cond |= (len(chunk) >= chunk_len_threshold)
    # new_chunk_cond |= (graph_nx.out_degree(n) >= out_degree_threshold)
    new_chunk_cond = False
    if (d >= diff_threshold):
      new_chunk_cond = True
      diff_threshold_cnt += 1
    if (len(chunk) >= chunk_len_threshold):
      new_chunk_cond = True
      chunk_len_threshold_cnt += 1
    if  (graph_nx.out_degree(n) >= out_degree_threshold):
      new_chunk_cond = True
      out_degree_threshold_cnt += 1

    if new_chunk_cond:
      chunks.append(chunk)
      chunk= set()

    chunk.add(n)
  chunks.append(chunk)

  # logger.info(f"diff_threshold_cnt: {diff_threshold_cnt}, chunk_len_threshold_cnt: {chunk_len_threshold_cnt}, out_degree_threshold_cnt: {out_degree_threshold_cnt}")
  return chunks

#### BEGIN: NEW DECOMPOSE MAIN FUNCTION ######
def decompose(graph, graph_nx, BB_graph, hw_depth, final_output_node_set, leaf_list, n_outputs, decompose_param_obj, hw_details, out_mode, verbose= False):
  """
    New version to replace _map_AC_to_hw method
    More cleaner and faster.
  """
  
  logger.info("Decomposing DAG into Building blocks")
  ## Argument assertions
  assert (hw_depth != 0), "HW depth should not be zero"
  
  fitness_wt_in= decompose_param_obj.fitness_wt_in
  fitness_wt_out= decompose_param_obj.fitness_wt_out
  fitness_wt_distance= decompose_param_obj.fitness_wt_distance

  max_depth= hw_details.max_depth
  assert max_depth <= hw_depth

  #-- Obj containing global sets
  common_obj= global_obj(hw_depth, hw_details)
  
  #-- Initialize children_to_check
  common_obj.children_to_check= set(leaf_list)
  
  # Init state of nodes to the desired value before begining
  for obj in graph.values():
    obj.fully_consumed= False

  cnt_nodes_couldnt_be_stored= [0]
  done_nodes_cnt= 0
  total_done_nodes_cnt= 0
  
  main_start=  time.time()
  all_outputs_computed= True
  for o in final_output_node_set:
    if not graph[o].computed:
      all_outputs_computed= False
      break

  while not all_outputs_computed:
    start= time.time()
    assert len(common_obj.children_to_check) != 0
    _check_children_to_check(graph, graph_nx, common_obj.children_to_check, common_obj.parents_to_check, max_depth)
    # print("parents_to_check", common_obj.parents_to_check)
    # assert len(common_obj.parents_to_check) != 0 
    #print '1', time.time() - start

    start= time.time()
    _create_decompose_units_for_parents_to_check(graph, graph_nx, common_obj.parents_to_check, common_obj.map_parent_to_decompose_unit, max_depth) 
    assert len(common_obj.map_parent_to_decompose_unit) >= len(common_obj.parents_to_check)
    #print '2', time.time() - start

    start= time.time()
    _schedulability_check(common_obj.parents_to_check, common_obj.map_parent_to_decompose_unit, common_obj.hw_details, common_obj.list_of_struct_lists, common_obj.map_parent_to_hw_type)
    # if len(common_obj.BB_graph) < 40:
    #   print('3', time.time() - start, end=' ')

    start= time.time()
    
    return_dict= _choose_most_eligible_set(graph, graph_nx, common_obj.list_of_struct_lists, fitness_wt_in, fitness_wt_out, fitness_wt_distance, hw_details)
    # if len(common_obj.BB_graph) < 40:
    #   print('4', time.time() - start, end=' ')

    best_uncomputed_cnt= return_dict['best_uncomputed_cnt']
    done_nodes_cnt += best_uncomputed_cnt
    total_done_nodes_cnt += best_uncomputed_cnt

    if done_nodes_cnt > 5000:
      logger.info(f"Done {total_done_nodes_cnt/len(graph) * 100}% nodes, time elapsed (sec): {time.time() - main_start}")
      done_nodes_cnt= 0

    start= time.time()
    
    best_set_of_decompose_units= return_dict['best_set_of_decompose_units']
    best_set_idx= return_dict['best_set_idx']

    nodes_to_store, nodes_fully_consumed= _schedule_the_set_new(graph, graph_nx, max_depth, final_output_node_set, best_set_of_decompose_units, best_set_idx, common_obj.list_of_hw_lists, out_mode, cnt_nodes_couldnt_be_stored, verb= False)
    assert len(nodes_to_store) != 0
    # print("nodes_to_store", nodes_to_store)
    # print("nodes_fully_consumed", nodes_fully_consumed)
    
    #assert len(nodes_to_store) <= 12
    #print nodes_to_store, nodes_fully_consumed
    #if len(common_obj.BB_graph) > 0:
    #  for unit in best_set_of_decompose_units:
    #    print "->", unit.child_dict,
    #  print ""

    # print(f"map_parent_to_decompose_unit : {common_obj.map_parent_to_decompose_unit}")
    _create_bb(graph, graph_nx, best_set_of_decompose_units, nodes_to_store, BB_graph, best_set_idx)
    _update_sets(graph, graph_nx, max_depth, nodes_to_store, nodes_fully_consumed, common_obj)
    # print(f"map_parent_to_decompose_unit : {common_obj.map_parent_to_decompose_unit}")
    # if len(common_obj.BB_graph) < 40:
    #   print('5', time.time() - start)

#      print(" ")

    all_outputs_computed= True
    for o in final_output_node_set:
      if not graph[o].computed:
        all_outputs_computed= False
        # logger.info(f"Output node {o} is still not computed. {graph[o].parent_key_list}, {graph[o].child_key_list}, {list(graph_nx.predecessors(o))}, {list(graph_nx.successors(o))}")        
        break

  logger.info(f"Done 100% nodes, time elapsed: {time.time() - main_start}")
  print_statistics(BB_graph)
  final_sanity_check(graph, graph_nx, final_output_node_set, common_obj, hw_depth, n_outputs, cnt_nodes_couldnt_be_stored)
      
#### END: NEW DECOMPOSE MAIN FUNCTION ####


#### BEGIN: PREPROCESSING ######
class Mutable_level():
  max_lvl= 0

def _set_dfs_level(graph_nx, graph): 
  """
    set a level for all the nodes based on a DFS traversal
  """

  # head_node= [node for node,obj in list(graph.items()) if not obj.parent_key_list]
  head_nodes = set([n for n in graph_nx if len(list(graph_nx.successors(n))) == 0])

  # assert len(head_node) == 1
  # head_node = head_node[0]

  mutable_lvl= Mutable_level()

  for h in head_nodes:
    head_lvl= _set_dfs_level_recurse(graph, graph_nx, h, mutable_lvl)
  assert head_lvl == len(graph), f"head_lvl: {head_lvl}, len(graph): {len(graph)}"

def _set_dfs_level_recurse(graph, graph_nx, curr_node, mutable_lvl):
  """
    set a level for all the nodes based on a DFS traversal
  """
  obj= graph[curr_node]

  if obj.dfs_level is None:
    max_lvl= mutable_lvl.max_lvl

    # for child in obj.child_key_list:
    for child in graph_nx.predecessors(curr_node):
      ch_lvl= _set_dfs_level_recurse(graph, graph_nx, child, mutable_lvl)
      if ch_lvl > max_lvl:
        max_lvl= ch_lvl
    
    obj.dfs_level= max_lvl + 1
    mutable_lvl.max_lvl= max_lvl + 1
  
  return obj.dfs_level
#### END: PREPROCESSING ######


##### BEGIN: CREATE PARENT LIST FROM CHILDREN LIST ####
def _check_children_to_check(graph, graph_nx, children_to_check, parents_to_check, max_depth):
  """
    By enumerating all the parents of children_to_check upto hw_depth, populate parents_to_check
  """
  for child in children_to_check:
    res_dict= _get_breadth_first_search_list_up(graph, graph_nx, child, max_depth)
    parent_set= res_dict['parent_set'] 
    parents_to_check.update(parent_set) # Set union
  
  #-- Reinitialize children_to_check
  children_to_check.clear()
def _get_breadth_first_search_list_up(graph, graph_nx, key, depth_to_be_checked): 
  """
  Gives the dictionary of all the (unique) parents until required depth
  In the returned dictionary, parents at different levels are keyed with depth
  """
  open_set= queue.Queue()
  closed_set= []
  
  open_set.put(key)
  open_set.put(None)
  
  depth=0
  parent_dict= defaultdict(list)
  
  parent_lvl_limit= graph[key].reverse_level + depth_to_be_checked # Only choose parents that are within given reverse_level. 

  while not open_set.empty():
    subtree_root = open_set.get()
    
    if (subtree_root == None):
      depth = depth + 1
      if (open_set.empty()):
        continue
      else:
        open_set.put(None)
        continue
    
    #if depth == depth_to_be_checked : # To be onle checked for depth = 0 to (depth_to_be_checked-1)
    #  break
    
    if depth > depth_to_be_checked : # To be onle checked for depth = 0 to (depth_to_be_checked-1)
      break

    if subtree_root in closed_set:
      continue
    
    curr_obj= graph[subtree_root]
    # for parent in curr_obj.parent_key_list:
    for parent in graph_nx.successors(subtree_root):
      #if open_set.qsize() > (depth+1)*50:
      #  break
      par_obj= graph[parent]
      
      #FIXME: Just for debug
      #if par_obj.reverse_level > parent_lvl_limit: 
      #  continue

      if not par_obj.computed and not par_obj.fully_consumed:
        open_set.put(parent)
            
    closed_set.append(subtree_root)
    parent_dict[depth].append(subtree_root)
    
    # Remove original key from closed_set and create parent_set
    parent_set= set(closed_set)
    parent_set.remove(key)

  return {'parent_dict':parent_dict, 'closed_set':closed_set, 'parent_set': parent_set} 
#### END: CREATE PARENT LIST FROM CHILDREN LIST ####

#### BEGIN: CREATE DECOMPOSE UNIT FOR THE CANDIDATE PARENTS ####
def _create_decompose_units_for_parents_to_check(graph, graph_nx, parents_to_check, map_parent_to_decompose_unit, max_depth):
  """
    Create/Update decompose unit for parents in parents_to_check
  """

  for parent in parents_to_check:
    return_dict= _get_breadth_first_search_list_down(graph, graph_nx, parent, max_depth, 'UNCOMPUTED')
    
    decompose_obj= common_classes.decompose_unit()
    decompose_obj.SCHEDULABLE_FLAG= return_dict['SCHEDULABLE_FLAG']
    decompose_obj.parent= parent
    decompose_obj.child_dict= return_dict['child_dict']
    decompose_obj.input_list= return_dict['input_list']
    decompose_obj.struct_graph= return_dict['struct_graph']
    decompose_obj.struct_graph_nx, decompose_obj.struct_graph_nx_tree= _create_struct_graph_nx(decompose_obj.struct_graph, decompose_obj.child_dict, decompose_obj.SCHEDULABLE_FLAG) 
    
    decompose_obj.uncomputed_nodes= set(decompose_obj.struct_graph_nx.nodes()) - set(decompose_obj.input_list)
    decompose_obj.uncomputed_cnt = len(decompose_obj.uncomputed_nodes) 
    decompose_obj.all_nodes= set(decompose_obj.struct_graph_nx.nodes())

    uncomputed_nodes_repeated= []
    for ls in list(decompose_obj.child_dict.values()):
      uncomputed_nodes_repeated += [node for node in ls if decompose_obj.is_uncomputed(node)]
    decompose_obj.uncomputed_nodes_repeated= uncomputed_nodes_repeated
    
    if decompose_obj.SCHEDULABLE_FLAG:
        # Only Assert when Schedulable Flag == True, otherwise child_dic and struct_graph can be out of sync, leading to assertion failure
      assert len(uncomputed_nodes_repeated) >= len(decompose_obj.uncomputed_nodes), [uncomputed_nodes_repeated, decompose_obj.uncomputed_nodes]

    map_parent_to_decompose_unit[parent]= decompose_obj

def _get_breadth_first_search_list_down(graph, graph_nx, key, depth_to_be_checked, option=''): 
  """
  Gives the dictionary of all the (unique) children until required depth
  In the returned dictionary, children at different levels are keyed with depth
  
  Options supported:
    'UNCOMPUTED' : Do not add the children of a computed node to the search list. Meaning, the search will not explore the childrens of computed node
  """
  option_list= ['UNCOMPUTED']
  
  assert isinstance(option, str), "option should be of string type"
  assert (option in option_list), "Unrecognized option" 
  
  # graph as a dict
  # Key: subroot ID
  # Val: list of children 
  # NOTE: if Key is already computed, Val is kept empty 
  # NOTE: Duplicates are merged!
  struct_graph= {}

  open_set= queue.Queue()
  closed_set= []
   
  open_set.put(key)
  open_set.put(None)
  
  # FIXME: Checking for a potential bug
  depth=0
  child_dict= defaultdict(list)
  
  input_list=[]
  
  SCHEDULABLE_FLAG = True
  while not open_set.empty():
    subtree_root = open_set.get()
    
    # IMP. NOTE: This checks for a few extra depth than the depth to be checked
    # This is an extra precausation. We should not cutoff in depth==depth_to_be_checked
    if depth > (depth_to_be_checked + 2) :
      SCHEDULABLE_FLAG= False
      break
    
    if (subtree_root == None):
      depth = depth + 1
      closed_set.append(None)
      if (open_set.empty()):
        continue
      else:
        open_set.put(None)
        continue

    # NOTE: Following code should not be used. Subtree_root should not be skipped even if it is already checked. We have to repeat the common operations as there is no sharing of childred in hw_tree.
    #if subtree_root in closed_set:
    #  continue
    
    struct_graph[subtree_root]= []
    if option== 'UNCOMPUTED' and graph[subtree_root].computed: # Do not add children if curr_node is computed
      input_list.append(subtree_root)
    else:
      # for child in graph[subtree_root].child_key_list:
      for child in graph_nx.predecessors(subtree_root):
        open_set.put(child)
        struct_graph[subtree_root].append(child)

    closed_set.append(subtree_root)
    child_dict[depth].append(subtree_root)
  
  assert option == 'UNCOMPUTED', "input_list is not populated for modes other than 'UNCOMPUTED'"
    
  return {'child_dict':child_dict, 'closed_set':closed_set, 'input_list': input_list, 'SCHEDULABLE_FLAG': SCHEDULABLE_FLAG, 'struct_graph':struct_graph} 

def _create_struct_graph_nx(struct_graph, child_dict, SCHEDULABLE_FLAG):
  """
    Comverts a struct_graph generated by "_get_breadth_first_search_list_down" function to an nx_graph for subgraph isomorphism
  """
  # DiGraph that encodes the unit structure
  # Edge directions from child to parent
  # NOTE: Inputs to the struct_graph, i.e. the nodes that are already computed are also represented as a node in struct_graph_nx
  # NOTE: returned struct_graph_nx is NOT a polytree. struct_graph_nx_tree is tree
  # It does not replicates nodes in case of common children
  struct_graph_nx= nx.DiGraph() 
  
  # DiGraph that encodes the unit structure
  # Edge directions from child to parent
  # New node IDs are used inorder to handle duplication of nodes in original graph
  # NOTE: Inputs to the struct_graph, i.e. the nodes that are already computed are also represented as a node in struct_graph_nx_tree
  # NOTE: returned struct_graph_nx_tree is a polytree. 
  # It replicates nodes in case of common children
  struct_graph_nx_tree= nx.DiGraph()
  
  if SCHEDULABLE_FLAG:
    #--- Construct struct_graph_nx_tree
    #Create new unique IDs
    map_old_id_to_new_id= {node: set() for node in struct_graph}
    node_id= 0
    # Key: new id, Val: object of type common_classes.decompose_unit_node
    map_new_id_to_node_details= {}
    for depth, nodes in list(child_dict.items()):
      for node in nodes:
        struct_graph_nx_tree.add_node(node_id)
        map_new_id_to_node_details[node_id]=  common_classes.decompose_unit_node(node_id, node)
        map_old_id_to_new_id[node].add(node_id)
        node_id += 1
    
    # Add edges
    for node in struct_graph_nx_tree.nodes():
      node_details= map_new_id_to_node_details[node]
      old_id= node_details.global_id
      for child in struct_graph[old_id]:
        child_new_id= map_old_id_to_new_id[child].pop()
        assert map_new_id_to_node_details[child_new_id].global_id == child
        node_details.local_child_key_ls.add(child_new_id)
        node_details.global_child_key_ls.add(child)
        struct_graph_nx_tree.add_edge(child_new_id, node)

    nx.set_node_attributes(struct_graph_nx_tree, map_new_id_to_node_details, 'details')
    assert nx.algorithms.tree.recognition.is_tree(struct_graph_nx_tree)
    assert nx.algorithms.components.is_weakly_connected(struct_graph_nx_tree)
  
  #--- Construct struct_graph_nx
  # Add nodes
  for node in struct_graph:
    struct_graph_nx.add_node(node)
  
  # Add edges
  for node, children in list(struct_graph.items()):
    for child in children:
      struct_graph_nx.add_edge(child, node)
  
  assert nx.algorithms.dag.is_directed_acyclic_graph(struct_graph_nx)
  assert nx.algorithms.components.is_weakly_connected(struct_graph_nx)


  return struct_graph_nx, struct_graph_nx_tree

#### END: CREATE DECOMPOSE UNIT FOR THE CANDIDATE PARENTS ####

#### BEGIN: PROFILE PARENTS ACCORDING TO THEIR SCHEDULABILITY ####
def _schedulability_check(parents_to_check, map_parent_to_decompose_unit, hw_details, list_of_struct_lists, map_parent_to_hw_type):
  """
    Check if parents are schedulable or not
  """
  for parent in parents_to_check:
    _perform_isomorphism_check_for_hw_sets(parent, map_parent_to_decompose_unit, hw_details, list_of_struct_lists, map_parent_to_hw_type)

def _perform_isomorphism_check_for_hw_sets(parent, map_parent_to_decompose_unit, hw_details, list_of_struct_lists, map_parent_to_hw_type):
  """
    Checks if the decompose_obj is mappable on which of the hw_sets in list_of_hw_lists.
    And, populate list_of_struct_lists
  """
  decompose_obj= map_parent_to_decompose_unit[parent]

  if decompose_obj.SCHEDULABLE_FLAG: # Hint from earlier stages of decomposition
    list_of_indices= []
    ATLEAST_ONE_SCHEDULABLE= False
    
    list_of_hw_lists= hw_details.list_of_hw_lists
    list_of_isomorph_indices= hw_details.list_of_isomorph_indices
    
    #-- Keep track of done nx_graphs (even the isomporphic ones)
    done= []
    for hw_set in list_of_hw_lists:
      done.append([])
      for nx_graph in hw_set:
        done[-1].append(False)
    
    count=0
    for set_idx, hw_set in enumerate(list_of_hw_lists):
      for intra_set_idx, hw in enumerate(hw_set):
        SCHEDULABLE_FLAG= False
        if not done[set_idx][intra_set_idx]:
          SCHEDULABLE_FLAG= _check_schedulability_nx_graphs(hw, decompose_obj.struct_graph_nx)
          count += 1
          if SCHEDULABLE_FLAG:
            list_of_struct_lists[set_idx][intra_set_idx].add(decompose_obj)
            list_of_indices.append([set_idx, intra_set_idx])

          # Also update list for all the isomorphic hw, and mark them done
          for indices in list_of_isomorph_indices[set_idx][intra_set_idx]:
            i0= indices[0]
            i1= indices[1]
            
            if SCHEDULABLE_FLAG:
              list_of_struct_lists[i0][i1].add(decompose_obj)
              list_of_indices.append([i0, i1])
            
            done[i0][i1]= True

        ATLEAST_ONE_SCHEDULABLE = ATLEAST_ONE_SCHEDULABLE or SCHEDULABLE_FLAG
    
    map_parent_to_hw_type[parent]= list_of_indices

    # print(f"SCHEDULABLE_FLAG was true, {parent}, current SCHEDULABLE_FLAG: {ATLEAST_ONE_SCHEDULABLE}")
    return_dict= {'SCHEDULABLE_FLAG': ATLEAST_ONE_SCHEDULABLE, 'list_of_indices': list_of_indices}
  else:
    return_dict= {'SCHEDULABLE_FLAG': False, 'list_of_indices': []}

  return return_dict

def _check_schedulability_nx_graphs(hw_graph_nx, struct_graph_nx):
  """
    It checks schedulability of a AC subgraph based on a BFS
    NOTE: This algorithm only works for polytrees only.
    NOTE: hw_graph_nx and struct_graph_nx are assumed to have only one root top node 
  """

  #print len(hw_graph_nx.nodes()),
  # Figure out top nodes in both hw_graph_nx and struct_graph_nx
  top_node_hw_graph= None
  for node in hw_graph_nx.nodes():
    if not hw_graph_nx.out_edges(node):
      top_node_hw_graph= node
      break
  top_node_struct_graph= None
  for node in struct_graph_nx.nodes():
    if not struct_graph_nx.out_edges(node):
      top_node_struct_graph= node
      break
  
  assert top_node_hw_graph is not None
  assert top_node_struct_graph is not None

  # A BFS check to map struct_graph_nx
  open_set_struct= queue.Queue()
  open_set_hw= queue.Queue()
  
  open_set_struct.put(top_node_struct_graph)
  open_set_hw.put(top_node_hw_graph)
  
  SCHEDULABLE_FLAG= True

  while not open_set_struct.empty():
    struct_node = open_set_struct.get()
    hw_node = open_set_hw.get()
    in_struct= struct_graph_nx.in_edges(struct_node)
    in_hw= hw_graph_nx.in_edges(hw_node)
    
    if len(in_struct) > len(in_hw):
      SCHEDULABLE_FLAG= False
      break
    
    # Ancestors are list of nodes with edges To curr_node
    struct_children= list(struct_graph_nx.predecessors(struct_node))
    hw_children= list(hw_graph_nx.predecessors(hw_node))
    
    for idx, child in enumerate(struct_children):
      open_set_struct.put(child)
      open_set_hw.put(hw_children[idx])
        
  
  #GM= nx.algorithms.isomorphism.isomorphvf2.DiGraphMatcher(hw_graph_nx, struct_graph_nx)
  #return GM.subgraph_is_isomorphic()
  
  return SCHEDULABLE_FLAG
#### END: PROFILE PARENTS ACCORDING TO THEIR SCHEDULABILITY ####

#### BEGIN: CHOOSE MOST ELIGIBLE SET ####
def _choose_most_eligible_set(graph, graph_nx, list_of_struct_lists, wt_in, wt_out, wt_distance_penalty, hw_details):
  """
  # Calculate best possible uncomputed_cnt for all decompose_unit in list_of_struct_lists
  """
  best_set_idx= 0
  best_uncomputed_cnt= -1000000
  best_overall_cnt= -10000000
  best_set_of_decompose_units= set()
  
  level_threshold = hw_details.level_threshold
  # Sort sets based on uncomputed_cnt
  #list_of_struct_lists_sorted= []
  #for idx, hw_set_lst in enumerate(list_of_struct_lists):
  #  list_of_struct_lists_sorted.append([])
  #  for hw_set in hw_set_lst:
  #    list_of_struct_lists_sorted[idx].append(sorted(hw_set, key= lambda x: x.uncomputed_cnt, reverse= True))
  
  main_unit_lst= []
  for idx, hw_set_lst in enumerate(list_of_struct_lists):
    best_unit= None
    best_cnt= -1
    candidates= list(sorted(hw_set_lst[0], key=lambda x: x.uncomputed_cnt, reverse= True))
    candidates= candidates[ : max(1, int(0.05*len(candidates)))]
    # mean_level= mean([graph[x.parent].level for x in hw_set_lst[0]])
    # best_unit= min(candidates, key=lambda x: abs(graph[x.parent].level - mean_level))
    mean_level= mean([graph[x.parent].dfs_level for x in hw_set_lst[0]])
    best_unit= min(candidates, key=lambda x: abs(graph[x.parent].dfs_level - mean_level))
    # logger.info(f"mean_level: {mean_level}, best_unit level: {graph[best_unit.parent].level}, max possible uncomputed_cnt: {candidates[0].uncomputed_cnt}, chosen uncomputed_cnt: {best_unit.uncomputed_cnt}")
    # best_unit= min(hw_set_lst[0], key=lambda x: graph[x.parent].level)
    # for unit in hw_set_lst[0]:
      # if unit.uncomputed_cnt > best_cnt:
      #   best_unit= unit
      #   best_cnt= unit.uncomputed_cnt
    main_unit_lst.append(best_unit)

  assert best_unit != None
  
  # Find the set with best uncomputed count
  list_of_set_of_decompose_units=[]
  #for idx, hw_set_lst in enumerate(list_of_struct_lists_sorted):

  max_input_nodes= hw_details.n_banks - max(2, 0.15*hw_details.n_banks)
  assert max_input_nodes > 0
  
  for idx, hw_set_lst in enumerate(list_of_struct_lists):
    curr_cnt=0
    checked_unit_lst= set()
    uncomputed_nodes= set()
    input_nodes= set()
    
    list_of_set_of_decompose_units.append(set())
    
    #main_unit = hw_set_lst[0][0]
    main_unit= main_unit_lst[idx]
    chosen_units= set([main_unit])
    input_nodes |= set(main_unit.input_list)
    uncomputed_nodes |= set(main_unit.struct_graph_nx.nodes()) - input_nodes
    # print(f"len main_unit: {len(uncomputed_nodes)}")
    list_of_set_of_decompose_units[idx].add(main_unit)
    
    fittness= len(uncomputed_nodes) - wt_in*len(input_nodes) -wt_out 
    parent_nodes= set([main_unit.parent])

    if len(hw_set_lst) > 1:
      for hw_set_idx, hw_set in enumerate(hw_set_lst[1:]):
        fittest_unit, curr_fittness, uncomp_cnt= _choose_best_instraset_partner(graph, graph_nx, hw_set, main_unit, chosen_units, uncomputed_nodes, parent_nodes, input_nodes, wt_distance_penalty, wt_in, wt_out, level_threshold)
        
        # Early exits
        if fittest_unit == None:
          break

        # early exist based on curr_fitness, if it is too low, subsequent fittest unit will have even worse fitness. so no point continuing
        if curr_fittness*(len(hw_set_lst) - hw_set_idx - 1) + fittness < best_overall_cnt:
          # print("weird reason i don;t remember: ", curr_fittness*(len(hw_set_lst) - hw_set_idx - 1), fittness, best_overall_cnt)
          break

        if fittest_unit != None and uncomp_cnt > 0.0: # curr_fittness > 0 avoids adding small subtrees if a super tree is already added
          fittness += curr_fittness 
          
          chosen_units.add(fittest_unit)
          input_nodes |= set(fittest_unit.input_list)
          parent_nodes.add(fittest_unit.parent)
          uncomputed_nodes |= set(fittest_unit.struct_graph_nx.nodes()) - input_nodes
          assert fittest_unit not in list_of_set_of_decompose_units[idx] 
          list_of_set_of_decompose_units[idx].add(fittest_unit)

        if len(input_nodes) >= max_input_nodes: # Try not to fetch inputs from all the register banks, which creates problem in reg bank allocation
          # print("info: ", uncomputed_nodes, input_nodes, max_input_nodes, hw_details.n_banks)
          break

    #for hw_set in hw_set_lst:
    #  for unit in hw_set:
        
        ## TRAIL. BAD IF INPUT COULD BE IN ANY BANK. GOOD ONLY IF THERE ARE RESTRICTIONS ON BANKS
        ## FIXME
        #if len(input_nodes | set(unit.input_list)) < len(input_nodes) + len(unit.input_list) - 3:
        #  continue

    #    if not unit in checked_unit_lst:
    #      uncomputed_nodes |= set(unit.struct_graph_nx.nodes()) - set(unit.input_list)
    #      input_nodes |= set(unit.input_list)
    #      checked_unit_lst.add(unit)
    #      assert not unit in list_of_set_of_decompose_units[idx]
    #      list_of_set_of_decompose_units[idx].add(unit)
    #      break
    
    #curr_cnt= float(len(uncomputed_nodes))
    
    #-- Penalize if there are too many inputs
    #curr_cnt -= float(wt_in * float(len(input_nodes)))

    #-- Penalize if there are too many different hw sets in a Hybrid BB (resulting in too many outputs)
    #curr_cnt -= float(wt_out * float(len(hw_set_lst)))

    # print(f"fittness: {fittness}, uncomp_cnt: {len(uncomputed_nodes)}, number of units: {len(list_of_set_of_decompose_units[idx])}")
    
    #if curr_cnt > best_overall_cnt:
    if fittness > best_overall_cnt:
      best_uncomputed_cnt= len(uncomputed_nodes)
      best_overall_cnt= fittness
      best_set_idx= idx
      # print("chosen")
    
  #for hw_set in list_of_struct_lists_sorted[best_set_idx]:
  #  print hw_set
  #  if len(hw_set):
  #    best_set_of_decompose_units.add(hw_set[0])
  best_set_of_decompose_units= list_of_set_of_decompose_units[best_set_idx]
  
  #for unit in best_set_of_decompose_units:
  #  print unit.child_dict
  #  print " "


  # print(f"{best_uncomputed_cnt} [{len(best_set_of_decompose_units)}],", end=' ')
   
  return {'best_set_of_decompose_units': best_set_of_decompose_units, 'best_uncomputed_cnt': best_uncomputed_cnt, 'best_set_idx': best_set_idx}

def get_suitable_hw_set(graph, graph_nx, hw_set, main_unit, level_threshold):
  main_level = graph[main_unit.parent].level
  
  suitable_hw_set = []
  if hw_set:
    for unit in hw_set:
      curr_level= graph[unit.parent].level
      level_diff= abs(main_level - curr_level)
      
      if level_diff <= level_threshold:
        suitable_hw_set.append(unit)

  return ssuitable_hw_set

def _choose_best_instraset_partner(graph, graph_nx, hw_set, main_unit ,chosen_units, uncomputed_nodes, parent_nodes, input_nodes, wt_distance_penalty, wt_in, wt_out, level_threshold):
  """
    chooses the best partner for main_unit depending on the "distance" from the main_unit and number of uncomputed_nodes 
  """
  
#  hw_set = [unit for unit in hw_set if unit not in chosen_units]

  main_dfs_level= graph[main_unit.parent].dfs_level
  # main_level = mean([graph[parent].level for parent in parent_nodes])
  main_level = graph[main_unit.parent].level

  # key: unit
  # val: fitness
  map_unit_fitness= {}
  
  graph_len= len(graph)
  
  best_fittness= -1000000
  fittest_unit= None
  best_uncomp_cnt= -100000
  best_level_diff= None

  min_fitness= -10000

  if hw_set:
    for unit in hw_set:
      curr_dfs_level= graph[unit.parent].dfs_level
      curr_level= graph[unit.parent].level

      distance= abs(curr_dfs_level - main_dfs_level)
      level_diff= abs(main_level - curr_level)
      
      uncomp_cnt= len(unit.uncomputed_nodes - uncomputed_nodes)
      
      extra_inputs= len(set(unit.input_list) - input_nodes)

      DO_NOT_ALLOW = (len(unit.uncomputed_nodes & parent_nodes) != 0)


      if level_diff <= level_threshold and uncomp_cnt != 0 and not DO_NOT_ALLOW:
        fitness= uncomp_cnt - wt_distance_penalty*distance/graph_len - wt_in*extra_inputs - wt_out    
      else:
        fitness= min_fitness
      # fitness= uncomp_cnt - wt_distance_penalty*distance/graph_len - wt_in*extra_inputs - wt_out    
      # fitness= uncomp_cnt - wt_distance_penalty*distance - wt_in*extra_inputs - wt_out    
      # if distance != 0:
      #   # fitness = uncomp_cnt/distance
      # else:
      #   assert uncomp_cnt == 0
      #   fitness = uncomp_cnt/0.5

      
      if fitness > best_fittness and fitness != min_fitness:
        best_fittness= fitness
        best_level_diff= level_diff
        fittest_unit= unit
        best_uncomp_cnt= uncomp_cnt

  # if fittest_unit != None:
  #   logger.info(f"best_level_diff: {best_level_diff}, main_level: {main_level}, uncomp_cnt: {len(fittest_unit.uncomputed_nodes)} curr_level: {graph[fittest_unit.parent].level}, best_fittness: {best_fittness}")
  # else:
  #   logger.info(f"best_level_diff: {best_level_diff}, main_level: {main_level}")

  return fittest_unit, best_fittness, best_uncomp_cnt

#### END: CHOOSE MOST ELIGIBLE SET ####


#### BEGIN: MAP NODES TO OUTPUT NODES ####
def check_parent_consumption(graph, graph_nx, node, nodes_to_store, nodes_fully_consumed):
  """
    counts parent consumption
  """
  consumed_parents= 0
  # for parent in graph[node].parent_key_list:
  for parent in graph_nx.successors(node):
    if graph[parent].computed or graph[parent].fully_consumed or (parent in nodes_to_store) or (parent in nodes_fully_consumed):
      consumed_parents += 1
  
  unconsumed_parents= len(list(graph_nx.successors(node))) - consumed_parents
  
  assert unconsumed_parents >= 0 
  
  return unconsumed_parents

def count_uncomputed_parents(graph, graph_nx, decompose_unit, nodes_to_store, nodes_fully_consumed):
  """
    Counts how many uncomputed parents are there for the nodes in a sub-graph
  """

  # Key: Node
  # Val: Number of parents that are yet to be computed
  map_node_n_unconsumed_parents= {}
  
  for node in decompose_unit.all_nodes:          
    unconsumed_parents= check_parent_consumption(graph, graph_nx, node, nodes_to_store, nodes_fully_consumed)
    map_node_n_unconsumed_parents[node] = unconsumed_parents
  
  return map_node_n_unconsumed_parents

def count_uncomputed_parents_subtree(graph, graph_nx, decompose_unit, curr_node, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed):
  open_set= queue.Queue()
  closed_set= set([])
  
  open_set.put(curr_node)
  
  relevant_nodes= set([])
  
  while not open_set.empty():
    node = open_set.get()
    
    if node in closed_set:
      continue

    if node in decompose_unit.uncomputed_nodes or node in nodes_to_store:
      relevant_nodes.add(node)

      # for child in graph[node].child_key_list:
      for child in graph_nx.predecessors(node):  
        open_set.put(child)
      
      closed_set.add(node)
    
    else:
      assert node in decompose_unit.input_list

  unconsumed_parents_cnt= [map_node_n_unconsumed_parents[node] for node in relevant_nodes]
  
  return sum(unconsumed_parents_cnt)


def _schedule_the_set_new(graph, graph_nx, max_depth, final_output_node_set, best_set_of_decompose_units, best_set_idx, list_of_hw_lists, mode, cnt_nodes_couldnt_be_stored , verb):
  """
     Output ports mapped to a limited number of arith nodes 
     NOTE: ONLY WORKS FOR TREES
  """
  #assert output_port_size >= len(best_set_of_decompose_units), "Output port size should atleast be more than number of units in a set"
  #assert mode in ["STORE_ALL", "STORE_LIMITED", "TOP_1", "TOP_2"]
  assert mode in ["ALL", "VECT", "TOP_1", "TOP_2"]

  nodes_to_store= set()
  nodes_fully_consumed= set()
  
  total_ports= 0
  for idx, unit in enumerate(best_set_of_decompose_units):
    if mode == "ALL" :
      available_ports = 2**(len(unit.child_dict)-1)
      max_level= len(unit.child_dict) - 2
    elif mode == "VECT":
      available_ports = 2**(len(unit.child_dict)-2)
      max_level= len(unit.child_dict) - 2
    elif mode == "TOP_2":
      available_ports = 2**(len(unit.child_dict)-1)
      if len(unit.child_dict) == max_depth + 1:  
        max_level= 1
      else:
        max_level= 0
    elif mode == "TOP_1":
      available_ports = 2**(len(unit.child_dict)-1)
      max_level= 0
    else:
      assert 0

    total_ports += available_ports
    
    map_node_n_unconsumed_parents= count_uncomputed_parents(graph, graph_nx, unit, nodes_to_store, nodes_fully_consumed)
    if verb:
      print("unit_depth:", len(unit.child_dict)-1 , "output_ports:", available_ports)
      print(map_node_n_unconsumed_parents)
      print(unit.child_dict)
    
    top_down_level= 0
    
    _schedule_a_tree_recurse(graph, graph_nx, unit, unit.parent, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed, available_ports, available_ports, top_down_level, max_level, verb) 
    
    if verb:
      print(nodes_to_store, nodes_fully_consumed) 
      print(" ")
    
  if verb:
    print("FINISHED")

  if nodes_to_store.intersection(nodes_fully_consumed):
    pass
    # print("--WASTE: ", len( nodes_to_store.intersection(nodes_fully_consumed)), '--', end=' ') 
  

  nodes_to_store = nodes_to_store - nodes_fully_consumed
  
  for node in nodes_fully_consumed:
    graph[node].fully_consumed= True
  
  for node in nodes_to_store:
    graph[node].computed = True
  
  
  # if not nodes_to_store:
  #   for unit in best_set_of_decompose_units:
  #     parent= unit.parent
  #     if parent in final_output_node_set:
  #       nodes_to_store.add(parent)
  #       graph[parent].computed = True
  #       print("FINAL_NODE")

  for unit in best_set_of_decompose_units:
    # print(f"All uncomputed_nodes nodes in the unit: {unit.uncomputed_nodes}")
    for n in unit.uncomputed_nodes:
      if n in final_output_node_set:
        nodes_to_store.add(n)
        graph[n].computed = True
        # print(f"FINAL_NODE: {n},")

  # For the nodes in final_output_node_set, which may be marked fully consumed if one of its parent in other parition
  nodes_fully_consumed -= nodes_to_store  

  for unit in best_set_of_decompose_units:
    assert unit.parent in nodes_to_store, f"{unit.parent}, {nodes_to_store}, {list(graph_nx.predecessors(unit.parent))}, {list(graph_nx.successors(unit.parent))}, {[_.child_dict for _ in best_set_of_decompose_units]}, {final_output_node_set}"
    for node in unit.uncomputed_nodes:
      if (node not in nodes_to_store) and (node not in nodes_fully_consumed):
        cnt_nodes_couldnt_be_stored[0] += 1
  
  assert nodes_to_store, best_set_of_decompose_units[0].parent 
  assert len(nodes_to_store) <= total_ports 
  assert len(nodes_to_store & nodes_fully_consumed) == 0
  # assert len(final_output_node_set & nodes_fully_consumed) == 0

  return nodes_to_store, nodes_fully_consumed

def _schedule_a_tree_recurse(graph, graph_nx, decompose_unit, curr_node, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed, output_port_size, max_port_size, top_down_level, max_level, verb):
  """
    Only populates nodes_to_store and nodes_fully_consumed objects. Doesn't update Graph state 
  """

  assert output_port_size <= max_port_size, [output_port_size, max_port_size]

  assert map_node_n_unconsumed_parents[curr_node] >= 0, [curr_node, map_node_n_unconsumed_parents, graph[curr_node].child_key_list, graph[curr_node].parent_key_list, decompose_unit.uncomputed_nodes]
  
  if curr_node not in nodes_fully_consumed:
    if map_node_n_unconsumed_parents[curr_node] == 0:
      nodes_fully_consumed.add(curr_node)
      
      # Decreament children count only if this is a not stored already. A node may be newly marked as fully_consumed even if it is already stored!
      if curr_node not in nodes_to_store:
        if curr_node in decompose_unit.uncomputed_nodes:
          # for child in graph[curr_node].child_key_list:
          for child in graph_nx.predecessors(curr_node):
            map_node_n_unconsumed_parents[child] -= 1
  
 
  # Recurse if not an input node
  if curr_node in decompose_unit.uncomputed_nodes:
    
    curr_output= 0
    # Store if required
    if map_node_n_unconsumed_parents[curr_node] > 0 and (curr_node not in nodes_to_store):
      if output_port_size > 0 and (top_down_level <= max_level):
        assert curr_node not in nodes_to_store
        nodes_to_store.add(curr_node)
        
        if curr_node in decompose_unit.uncomputed_nodes:
          # for child in graph[curr_node].child_key_list:
          for child in graph_nx.predecessors(curr_node):
            map_node_n_unconsumed_parents[child] -= 1
        
        output_port_size -= 1
        curr_output = 1

    assert len(graph[curr_node].child_key_list) == 2, (curr_node, graph[curr_node].child_key_list)
    assert len(list(graph_nx.predecessors(curr_node))) == 2, f"{curr_node}, {list(graph_nx.predecessors(curr_node))}, {graph[curr_node].child_key_list}"
    
    child_0 = graph[curr_node].child_key_list[0]
    child_1 = graph[curr_node].child_key_list[1]
    
    child_0_cnt= count_uncomputed_parents_subtree(graph, graph_nx, decompose_unit, child_0, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed)
    child_1_cnt= count_uncomputed_parents_subtree(graph, graph_nx, decompose_unit, child_1, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed)
    
    if child_0_cnt > child_1_cnt:
      first_ch= child_0
      second_ch= child_1
    else:
      first_ch= child_1
      second_ch= child_0
      
    max_port_size_by_2= int(max_port_size/2)
    first_ch_port= output_port_size if (output_port_size < max_port_size_by_2) else max_port_size_by_2
    first_ch_outputs= _schedule_a_tree_recurse(graph, graph_nx, decompose_unit, first_ch, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed, first_ch_port, max_port_size_by_2, top_down_level + 1, max_level, verb)
    
    second_ch_port= (output_port_size-first_ch_outputs) if ((output_port_size-first_ch_outputs) < max_port_size_by_2) else max_port_size_by_2
    second_ch_outputs= _schedule_a_tree_recurse(graph, graph_nx, decompose_unit, second_ch, map_node_n_unconsumed_parents, nodes_to_store, nodes_fully_consumed, second_ch_port, max_port_size_by_2, top_down_level + 1, max_level, verb)
    
    if verb:
      print(curr_node, output_port_size, "First_ch:", first_ch, first_ch_port, first_ch_outputs, "Second_ch:", second_ch, second_ch_port, second_ch_outputs)
    
    return curr_output + first_ch_outputs + second_ch_outputs
  
  if verb:
    print(curr_node, graph[curr_node].child_key_list, graph[curr_node].parent_key_list)
  
  return 0

#### END: MAP NODES TO OUTPUT NODES ####


#### BEGIN: UPDATE THE STATE BASED ON STORED NODES
def _update_sets(graph, graph_nx, hw_depth, nodes_to_store, nodes_fully_consumed, common_obj):
  """
    After schduling a set, update the sets in common_obj for next iteration
  """
  
  # Create children_to_check for next iter
  common_obj.children_to_check= set(nodes_to_store)

  assert len(common_obj.children_to_check) != 0

  # Remove nodes that are store and fully consumed from parent sets
  combined_nodes = nodes_to_store | nodes_fully_consumed
  
  # Remove all details related to parents of new children_to_check 
  for child in common_obj.children_to_check:
    res_dict= _get_breadth_first_search_list_up(graph, graph_nx, child, hw_depth)
    parent_set= res_dict['parent_set'] 
    combined_nodes.update(parent_set)
  
  # Create sets from of lists
  # This is for time efficency. Removing an object from a set is much faster than a list
  #new_list_of_struct_lists= []
  #for idx, hw_set_lst in enumerate(common_obj.list_of_struct_lists):
  #  new_list_of_struct_lists.append([])
  #  for hw_set in hw_set_lst:
  #    new_list_of_struct_lists[idx].append(set(hw_set))
  #    assert len(hw_set) == len(new_list_of_struct_lists[idx][-1]), hw_set
  
  for node in combined_nodes:
    if node in common_obj.map_parent_to_hw_type:
      for indices in common_obj.map_parent_to_hw_type[node]:
        set_idx= indices[0]
        intra_set_idx= indices[1]
        
        decompose_obj= common_obj.map_parent_to_decompose_unit[node]
        
        common_obj.list_of_struct_lists[set_idx][intra_set_idx].remove(decompose_obj)
        #new_list_of_struct_lists[set_idx][intra_set_idx].remove(decompose_obj)
      
      del common_obj.map_parent_to_hw_type[node]
      del common_obj.map_parent_to_decompose_unit[node]

  
  ## Create list back out of sets
  #for idx, hw_set_lst in enumerate(common_obj.list_of_struct_lists):
  #  for hw_set_idx, hw_set in enumerate(hw_set_lst):
  #    assert len(hw_set) >= len(new_list_of_struct_lists[idx][hw_set_idx])
  #    common_obj.list_of_struct_lists[idx][hw_set_idx]= list(new_list_of_struct_lists[idx][hw_set_idx])

  # Clear parents_to_check
  common_obj.parents_to_check.clear()
#### END: UPDATE THE STATE BASED ON STORED NODES
 

#### BEGIN: POST PROCESSING ####
def _create_bb(graph, graph_nx, best_set_of_decompose_units, nodes_to_store, BB_graph, selected_hw_set_idx):
  """
    Create a new BB from best_set_of_decompose_units and insert in BB_graph
  """
  
  bb_key = len(BB_graph) + 1
  bb_obj = common_classes.build_blk(bb_key)

  bb_obj.set_of_decompose_units= best_set_of_decompose_units
  
  # Create input list
  input_list= []
  for unit in best_set_of_decompose_units:
    input_list= input_list + unit.input_list
  
  bb_obj.in_list= input_list
  bb_obj.in_list_unique= list(set(input_list))

  for node in set(input_list):
    graph[node].parent_buildblk_lst.append(bb_key)

  # Create out list
  bb_obj.out_list= list(nodes_to_store)
  
  for node in nodes_to_store:
    assert graph[node].storing_builblk == None , "Should not be assigned another storing build_blk"
    graph[node].storing_builblk= bb_key

  # Internal nodes
  all_nodes= set()
  for unit in best_set_of_decompose_units:
    all_nodes |= set(unit.struct_graph_nx.nodes())
  in_nodes= set(input_list)
  internal_nodes= all_nodes - in_nodes
  bb_obj.internal_node_list= list(internal_nodes)
  
  bb_obj.selected_hw_set_idx= selected_hw_set_idx

  for node in internal_nodes:
    graph[node].compute_buildblk_lst.append(bb_key)
  
  BB_graph[bb_key]= bb_obj
 
def final_sanity_check(graph, graph_nx,final_output_node_set,common_obj, hw_depth, n_outputs, cnt_nodes_couldnt_be_stored):
  """
    Sanity checks
  """
  # Sanity Check
  for unit_set in common_obj.list_of_struct_lists:
    for intra_set_unit in unit_set:
      #assert len(intra_set_unit) == 0, ["list_of_struct_lists should be empty by now", intra_set_unit]
      if len(intra_set_unit) != 0:
        for unit in intra_set_unit:
          print(unit.parent, unit.child_dict)
  assert len(common_obj.parents_to_check) == 0, "common_obj.parents_to_check"
  
  # All nodes should be either computed or fully_consumed
  for node,obj in list(graph.items()):
    #assert obj.fully_consumed or obj.computed
    if node not in final_output_node_set:
      assert obj.fully_consumed
  
  # Assert output ports are used effectively
  # FIXME: Check disabled just for TOP-n mode. Renable it
  #if n_outputs >= (2**hw_depth) - 1:
  #  assert cnt_nodes_couldnt_be_stored[0] == 0, cnt_nodes_couldnt_be_stored[0]


def print_statistics(BB_graph):
  input_cnt= 0
  output_cnt= 0
  for key, BB in list(BB_graph.items()):
    input_cnt += len(set(BB.in_list))
    output_cnt += len(set(BB.out_list))
  
  logger.info(f'Total blocks: {len(BB_graph)}')
  # print('Total input cnt: ', input_cnt)
  # print('Total output cnt: ', output_cnt)
  # print('Total reg rd-wr cnt:', input_cnt + output_cnt)
  
  # print("Nodes that could not be stored due to lack of output ports: ", cnt_nodes_couldnt_be_stored[0])
  
def create_full_BB_graph(graph, graph_nx, BB_graph):
  for bb, obj in list(BB_graph.items()):
    for in_node in obj.in_list_unique:
      child_bb= graph[in_node].storing_builblk
      if child_bb != None: # child_bb_lst_duplicates can have duplicates. This is allowed because BBs can have multiple outputs
        obj.child_bb_lst_duplicates.append(child_bb)
      
      if child_bb != None and child_bb not in obj.child_bb_lst: # Without duplicates
        obj.child_bb_lst.append(child_bb)
      
    
    for out_node in obj.out_list:
      for out_bb in graph[out_node].parent_buildblk_lst:
        obj.parent_bb_lst_duplicates.append(out_bb) # parent_bb_lst can have duplicates, as BBs can have multiple outputs
        
        if out_bb not in obj.parent_bb_lst: # Without duplicates
          obj.parent_bb_lst.append(out_bb)

def create_BB_NX_from_BB(BB_graph):
  """
    Create a networkx graph for BB_graph of type MultiDiGraph. It also encodes the multiple edges between BBs
  """
  G= nx.MultiDiGraph()
  for BB,obj in list(BB_graph.items()):
    G.add_node(BB)

    for child in obj.child_bb_lst_duplicates:
      G.add_edge(child, BB) # Multiple edges could be added between two nodes due to duplicates
  
  return G
#### END: POST PROCESSING ####


##### END OF NEW FUNCTIONS #####
