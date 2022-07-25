
import math

from . import common_classes
import itertools
from . import useful_methods

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def typecast(s):
  try:
    return int(s)
  except ValueError:
    try:
      return float(s)
    except ValueError:
      return s # neither an int nor a float

def main(file_name):
  lines= read_psdd(file_name)

  id_iter= itertools.count()

  graph= create_raw_graph(lines, id_iter)
  useful_methods.check_if_only_one_root(graph)
  # logger.info(f'raw graph len: {len(graph)}')

  binarize(graph, id_iter)
  head_node= useful_methods.check_if_only_one_root(graph)
  logger.info(f'binarized graph len: {len(graph)}')

  graph_nx = useful_methods.create_nx_graph_from_node_graph(graph)

  leaf_list= [node for node, obj in list(graph.items()) if obj.is_leaf()]
  

  ac_node_list= list(graph.keys())
  
  logger.info(f'Leaf nodes: {len(leaf_list)}')
  # print('indicators: ', len([node for node, obj in list(graph.items()) if obj.is_indicator()]))
  # print('weights: ', len([node for node, obj in list(graph.items()) if obj.is_weight()]))
  return graph, graph_nx, head_node, leaf_list, ac_node_list
  

def read_psdd(file_name):
  """
    c ids of psdd nodes start at 0
    c psdd nodes appear bottom-up, children before parents
    c
    c file syntax:
    c psdd count-of-sdd-nodes
    c L id-of-literal-sdd-node literal 
    c T id-of-trueNode-sdd-node id-of-vtree trueNode variable log(litProb)
    c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub log(elementProb)}*
    c
  """

  with open(file_name, 'r') as f:
    lines= f.readlines()
    
  # remove '\n'
  lines= [i.strip() for i in lines]
  
  # Remove comments and other lines
  lines= [i for i in lines if (i[0] == 'T' or i[0]== 'L' or i[0]=='D')]
  
  lines= [l.split(' ') for l in lines]
  lines= [[typecast(s) for s in l] for l in lines]

  for l in lines:
    if l[0] == 'T':
      l[4]= math.exp(l[4]) # Antilog
      assert l[4] <= 1.0 and l[4] >= 0.0
    
    if l[0] == 'D':
      n_AND_gates= l[3]

      for gate in range(n_AND_gates):
        weight_idx= 3*gate + 6
        l[weight_idx] = math.exp(l[weight_idx])
        assert l[weight_idx] <= 1.0 and l[weight_idx] >= 0.0

  assert len(lines) != 0
  return lines  

def create_raw_graph(lines, id_iter):
  """
    lines is output of read_psdd

    creates a raw graph by converting 'T' node to sums and products,
    and explicitly encode product and sum nodes from 'D'
    
    resulting graph would be non binary
  """
  map_psdd_id_to_node_id= {}
  
  map_real_val_to_node_id= {}
  
  # map var_id used by 'T' and 'L' type to the leaf node_id
  # This is to avoid multiple indicators for a variable in the application
  map_var_id_to_node_id= {}

  graph= {}

  for l in lines:
    if l[0] == 'L':
      psdd_id= l[1]
      var_id= l[3]

      node_id= create_indicator_leaf_node(graph, var_id, map_var_id_to_node_id, id_iter)
      map_psdd_id_to_node_id[psdd_id]= node_id

    elif l[0] == 'T':
      # output= pos_var_id * pos_var_val + neg_var_id * neg_var_val

      psdd_id= l[1]
      pos_id= l[3]
      neg_id= -pos_id
      pos_val= l[4]
      neg_val= 1- pos_val

      pos_id_node_id= create_indicator_leaf_node(graph, pos_id, map_var_id_to_node_id, id_iter)
      pos_val_node_id= create_real_valued_leaf_node(graph, pos_val, map_real_val_to_node_id, id_iter)
          
      neg_id_node_id= create_indicator_leaf_node(graph, neg_id, map_var_id_to_node_id, id_iter)
      neg_val_node_id= create_real_valued_leaf_node(graph, neg_val, map_real_val_to_node_id, id_iter)

      prod_0_node_id= create_prod_node(graph, id_iter)
      add_parent_child_edge(graph, prod_0_node_id, pos_id_node_id)
      add_parent_child_edge(graph, prod_0_node_id, pos_val_node_id)

      prod_1_node_id= create_prod_node(graph, id_iter)
      add_parent_child_edge(graph, prod_1_node_id, neg_id_node_id)
      add_parent_child_edge(graph, prod_1_node_id, neg_val_node_id)

      sum_node_id= create_sum_node(graph, id_iter)
      add_parent_child_edge(graph, sum_node_id, prod_0_node_id)
      add_parent_child_edge(graph, sum_node_id, prod_1_node_id)

      map_psdd_id_to_node_id[psdd_id]= sum_node_id

    elif l[0] == 'D':

      psdd_id= l[1]
      n_AND_gates= l[3]

      # create sum node first
      sum_node_id= create_sum_node(graph, id_iter)

      for gate in range(n_AND_gates):
        in_0= l[3*gate + 4]
        in_1= l[3*gate + 5]
        val = l[3*gate + 6]
        
        val_node_id= create_real_valued_leaf_node(graph, val, map_real_val_to_node_id, id_iter)

        prod_node_id= create_prod_node(graph, id_iter)
        add_parent_child_edge(graph, prod_node_id, val_node_id)
        add_parent_child_edge(graph, prod_node_id, map_psdd_id_to_node_id[in_0])
        add_parent_child_edge(graph, prod_node_id, map_psdd_id_to_node_id[in_1])

        add_parent_child_edge(graph, sum_node_id, prod_node_id)
      
      if n_AND_gates == 1: # no need of sum node
        assert len(graph[sum_node_id].child_key_list) == 1
        prod_id= graph[sum_node_id].child_key_list[0] 
        map_psdd_id_to_node_id[psdd_id]= prod_id

        graph[prod_id].parent_key_list.remove(sum_node_id)
        del graph[sum_node_id]

      else:
        map_psdd_id_to_node_id[psdd_id]= sum_node_id
        
    else:
      assert 0

  return graph

def binarize(graph, id_iter):
  for node in list(graph.keys()):
    obj= graph[node]

    if len(obj.child_key_list) > 2:
      assert obj.is_sum() or obj.is_prod()
      
      # create a tree of nodes
      tree_top= create_tree_of_nodes(graph, list(obj.child_key_list), id_iter, obj)
     
      # delete edges in child
      for child in list(obj.child_key_list):
        del_parent_child_edge(graph, node, child)

      # delete and add edges in parents
      for parent in list(obj.parent_key_list):
        del_parent_child_edge(graph, parent, node)
        add_parent_child_edge(graph, parent, tree_top)
      
      del graph[node]

  # Assert that graph is binary
  for node, obj in list(graph.items()):
    assert len(obj.child_key_list) == 2 or len(obj.child_key_list) == 0, [len(obj.child_key_list), node, obj.is_leaf(), obj.is_sum(), obj.is_prod()]

def create_real_valued_leaf_node(graph, val, map_real_val_to_node_id, id_iter):
  """
    Checks if this value already has a node,
    if not, generates a new node
  """

  if not val in map_real_val_to_node_id:
    node_obj= common_classes.node(next(id_iter))
    node_obj.set_leaf_weight()
    node_obj.curr_val= val

    map_real_val_to_node_id[val]= node_obj

    assert not node_obj.key in graph
    graph[node_obj.key]= node_obj

  return map_real_val_to_node_id[val].key

def create_indicator_leaf_node(graph, var_id, map_var_id_to_node_id, id_iter):
  if not var_id in map_var_id_to_node_id:
    node_obj= common_classes.node(next(id_iter))
    node_obj.set_leaf_indicator()
    node_obj.psdd_literal_id = var_id

    map_var_id_to_node_id[var_id]= node_obj
    
    assert not node_obj.key in graph
    graph[node_obj.key]= node_obj

  return map_var_id_to_node_id[var_id].key

def create_sum_node(graph, id_iter):
  key= next(id_iter)
  node_obj= common_classes.node(key)
  node_obj.set_sum()
  graph[key]= node_obj
  return key

def create_prod_node(graph, id_iter):
  key= next(id_iter)
  node_obj= common_classes.node(key)
  node_obj.set_prod()
  graph[key]= node_obj
  return key

def add_parent_child_edge(graph, parent, child):
  graph[parent].child_key_list.append(child)
  graph[child].parent_key_list.append(parent)

def del_parent_child_edge(graph, parent, child):
  graph[parent].child_key_list.remove(child)
  graph[child].parent_key_list.remove(parent)


def create_tree_of_nodes(graph, node_list, id_iter, curr_node_obj):
  len_node_list= len(node_list)

  assert len_node_list > 0

  if len_node_list == 1:
    assert node_list[0] in graph, [node_list[0], curr_node_obj.key]
    return node_list[0]
  
  if len_node_list > 1:
    # Slicing index is the biggest power of 2 smaller than len_node_list
    biggest_power_2= int(math.log(len_node_list,2))
    slicing_idx= 2**biggest_power_2
    if len_node_list == slicing_idx:
      slicing_idx /= 2
      slicing_idx= int(slicing_idx)

    child_0= create_tree_of_nodes(graph, list(node_list[ : slicing_idx]), id_iter, curr_node_obj)
    child_1= create_tree_of_nodes(graph, list(node_list[slicing_idx : ]), id_iter, curr_node_obj)

    if curr_node_obj.is_sum():
      key= create_sum_node(graph, id_iter)
    elif curr_node_obj.is_prod(): 
      key= create_prod_node(graph, id_iter)
    else:
      assert 0

    add_parent_child_edge(graph, key, child_0)
    add_parent_child_edge(graph, key, child_1)

    return key

def instanciate_literals(graph, lit_list):
  for node, obj in graph.items():
    if obj.is_indicator():
      lit_id= obj.psdd_literal_id 
      assert lit_id != None
      
      idx = abs(lit_id) - 1
      curr_val = lit_list[idx]
      assert curr_val in [1, 0]
      
      if (lit_id) < 0:
        curr_val = 0 if curr_val else 1

      obj.curr_val = curr_val
