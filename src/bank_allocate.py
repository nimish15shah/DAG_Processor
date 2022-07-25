import time
import random
import networkx as nx
import itertools
import copy
from collections import defaultdict
import heapq

#**** imports from our codebase *****
from . import common_classes
from . import useful_methods
from .useful_methods import printcol, printlog

from . import write_to_file

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bank_allocate(net, graph, BB_graph, n_mem_banks, hw_details, instr_ls_obj, w_conflict, MEM_LOAD_CONST ):
  """
     w_conflict is an hyperparameter to tradeoff between bank usage and conflicts # NOTE : not used any  more
  """
  useful_methods.printlog('Performing Bank allocation', 'red')
  
  IO_graph= create_IO_graph(graph, BB_graph, instr_ls_obj, MEM_LOAD_CONST)

  graph_nx = useful_methods.create_nx_graph_from_node_graph(graph)
#  exit(1)
#  write_to_file.minizinc_pe_bank_allocate(fpath, IO_graph, n_mem_banks)
#  exit(1)

      
  #-- Assign banks to the nodes, using a graph-coloring method
  color_d = nx.algorithms.coloring.greedy_color(IO_graph, strategy=nx.algorithms.coloring.strategy_largest_first)
  print("Number of banks that ensure 0 conflicts:", max(color_d.values()) + 1)
  
  #print len(list(nx.algorithms.clique.enumerate_all_cliques(IO_graph))[0])
  #print "Max_clique size:", len(list(nx.algorithms.clique.enumerate_all_cliques(IO_graph))[0]) 
  
  #-- Creata a uniform distribution of banks
  max_allowed_colors= n_mem_banks
  new_color_d= redist_colors(IO_graph, max_allowed_colors,w_conflict )
  
  conflicts, conflicts_in, conflicts_out, conflicts_ld= count_conflicts(BB_graph, IO_graph, instr_ls_obj, new_color_d, n_mem_banks, hw_details)
  print("Total conflicts after redistribution: ", conflicts)
  
  num_of_banks= max_allowed_colors
  
  #-- Small crossbar B
#  color_d= small_crossbar_B(graph, BB_graph, IO_graph, n_mem_banks)
#  conflicts= count_conflicts(BB_graph, IO_graph, instr_ls_obj, new_color_d, n_mem_banks)
  
  #--- No crossbar B
  new_color_d, pe_d, conflicts_out = color_according_to_port_map(graph, BB_graph, IO_graph, max_allowed_colors,hw_details)

  # random allocation!
  # total_colors= list(range(n_mem_banks))
  # new_color_d = {n: random.choice(total_colors) for n in new_color_d.keys()}

  conflicts, conflicts_in, conflicts_out_0, conflicts_ld= count_conflicts(BB_graph, IO_graph, instr_ls_obj, new_color_d, n_mem_banks, hw_details, mode= 'out_port_map', pe_d= pe_d)
  if hw_details.out_port_mode == 'CROSSBAR':
    pass
  else:
    assert conflicts_out_0 <= conflicts_out, f"{conflicts_out}, {conflicts_out_0}"
  
  # Update map for output to pe
  for instr in instr_ls_obj.instr_ls:
    if instr.is_type('bb'):
      bb_obj= BB_graph[instr.bb_id]
      for output, pe in list(bb_obj.out_nodes_to_pe.items()):
        instr.output_to_pe[output]= pe
  
  return new_color_d, pe_d, num_of_banks, IO_graph, conflicts, conflicts_in, conflicts_out, conflicts_ld

def color_according_to_port_map_v2(graph, BB_graph, IO_graph, n_mem_banks, hw_details):
  out_port_map= hw_details.out_port_map
  in_port_map= hw_details.in_port_map
  tree_depth= hw_details.tree_depth

  # Reverse map of out_port_map  # Key: bank id  # Val: tuple [tree_id, pe_id]
  map_out_bank_to_pe={bank: set() for bank in range(n_mem_banks)}
  for pe_tup, bank_set in list(out_port_map.items()):
    if bank in bank_set:
      map_out_bank_to_pe[bank].add(pe_tup)

  # Reverse map of in_port_map  # Key: bank id  # Val: tuple [tree_id, pe_id]
  map_in_bank_to_pe={bank: set() for bank in range(n_mem_banks)}
  for pe_tup, bank_set in list(in_port_map.items()):
    if bank in bank_set:
      map_in_bank_to_pe[bank].add(pe_tup)
  

def color_according_to_port_map(graph, BB_graph, IO_graph, n_mem_banks, hw_details):
  """
    Color for no crossbar at output. Only one at input
  """
  out_port_map= hw_details.out_port_map
  n_tree= hw_details.n_tree
  tree_depth= hw_details.tree_depth

#  assert hw_details.max_depth - hw_details.min_depth <= 1, "Some function in map_helper doesn't support other depths"

  # Reverse map of out_port_map  # Key: bank id  # Val: tuple [tree_id, pe_id]
  map_bank_to_pe={}
  
  for bank in range(n_mem_banks):
    map_bank_to_pe[bank]= set()
    for pe_tup, bank_set in list(out_port_map.items()):
      if bank in bank_set:
        map_bank_to_pe[bank].add(pe_tup)
  
  # Map level to pe  # Key: lvl  # Val: tuple [tree_id,lvl, pe_id]
  map_lvl_to_pe={}
  
  for rev_lvl in range(tree_depth):
    lvl= tree_depth- rev_lvl # lvl starts from 1 and not 0
    map_lvl_to_pe[lvl]= set()
    for tree in range(n_tree):
      for pe_id in range(2**rev_lvl):
        map_lvl_to_pe[lvl].add((tree, lvl,pe_id))
  
  # node attributes dict
  # key: node, Val: object of type node_attr
  node_attr_d= {node: node_attr(node, graph, BB_graph, map_lvl_to_pe, out_port_map, n_mem_banks) for node in IO_graph.nodes()}



  # map_helper object per BB
  map_helper_d= {bb: map_helper(n_tree, tree_depth, bb_obj) for bb, bb_obj in list(BB_graph.items())}
  

  # Allocate banks
  bank_d= {}
  pe_d= {}
  total_colors= set(range(n_mem_banks))
  
  output_conflict_bb_set= set()

  output_conflict= 0
  uncolored_nodes=  set(IO_graph.nodes())

  map_allowed_bank_cnt_to_set_uncolored_nodes= {cnt : set() for cnt in range(0, n_mem_banks + 1)}

  for n in uncolored_nodes:
    allowed_bank_cnt= len(node_attr_d[n].allowed_banks)
    map_allowed_bank_cnt_to_set_uncolored_nodes[allowed_bank_cnt].add(n)

  done_nodes= set()
  start_time= time.time()
  while len(bank_d) < IO_graph.number_of_nodes():
    if len(done_nodes) % 10000 == 0:
      logger.info(f"done_nodes: {len(done_nodes)}, done time: {time.time() - start_time}")
#    print len(uncolored_nodes),
#    uncolored_nodes= [node for node, obj in node_attr_d.items() if not obj.bank_allocated_flag]

    # min_col_node= min(uncolored_nodes, key= lambda x: len(node_attr_d[x].allowed_banks))
    min_col_node = None
    # pick up a node with the lowest level, to ensure they get a preference for an output port
    for b in range(0, 3):
      lvl_1_nodes= []
      for n in map_allowed_bank_cnt_to_set_uncolored_nodes[b]:
        if node_attr_d[n].lvl == 1:
          lvl_1_nodes.append(n)
      
      if len(lvl_1_nodes) != 0:
        assert min_col_node == None
        # min_col_node = max(lvl_1_nodes, key = lambda x: len(list(IO_graph.neighbors(x))))
        min_col_node= random.choice(lvl_1_nodes)
        map_allowed_bank_cnt_to_set_uncolored_nodes[b].remove(min_col_node)

      if min_col_node != None:
        break

    
    # The above loops did not choose any node
    if min_col_node == None:
      for b in range(0, n_mem_banks + 1):
        if len(map_allowed_bank_cnt_to_set_uncolored_nodes[b]) != 0:
          # pick up a node with the lowest level, to ensure they get a preference for an output port
          for n in map_allowed_bank_cnt_to_set_uncolored_nodes[b]:
            if node_attr_d[n].lvl == 1:
              assert min_col_node == None
              min_col_node = n
              map_allowed_bank_cnt_to_set_uncolored_nodes[b].remove(n)
              break


          if min_col_node == None:
            for n in map_allowed_bank_cnt_to_set_uncolored_nodes[b]:
              if node_attr_d[n].lvl == 2:
                assert min_col_node == None
                min_col_node = n
                map_allowed_bank_cnt_to_set_uncolored_nodes[b].remove(n)
                break

          # The above loops did not choose any node
          if min_col_node == None:
            assert min_col_node == None
            min_col_node = map_allowed_bank_cnt_to_set_uncolored_nodes[b].pop()
            break
          
          if min_col_node != None:
            break
      
    assert min_col_node != None

    # nodes whose banks and pes would also get affected.
    affected_nodes = set(list(IO_graph.neighbors(min_col_node)))
    if not graph[min_col_node].is_leaf():
      bb = graph[min_col_node].storing_builblk
      bb = BB_graph[bb]
      affected_nodes |= set(bb.out_list) | set(bb.in_list)
      affected_nodes.remove(min_col_node)
    affected_nodes -= done_nodes
    map_affected_old_bank_cnt= {n : len(node_attr_d[n].allowed_banks) for n in affected_nodes}

    min_col_cnt= len(node_attr_d[min_col_node].allowed_banks)

    MAGIC_COL_OFFSET= 2
    MAGIC_DEGREE_FACTOR= 1
    MIN_COL_THRESHOLD= 0.3
    WT_DEGREE_0= 0.9
    WT_DEGREE_1= 0.15
    
    # Nodes with almost similar col cnt as min_col_cnt
#    candidate_list= [node for node in uncolored_nodes if len(node_attr_d[node].allowed_banks) <= min_col_cnt + MAGIC_COL_OFFSET]
#
#    
#    if min_col_cnt > MIN_COL_THRESHOLD * n_mem_banks: # Choose the one with highest degree
#      # Nodes with high degree, but not enough col cnt
#      max_degree= max([IO_graph.degree(x) for x in candidate_list])
#      candidate_list += [node for node in uncolored_nodes if IO_graph.degree(node) > MAGIC_DEGREE_FACTOR*max_degree]
##      curr_node= max(candidate_list, key= lambda x: IO_graph.degree(x))
#      curr_node= min(candidate_list, key= lambda x: len(node_attr_d[x].allowed_banks) + (WT_DEGREE_0 * IO_graph.degree(x)))
#
#    else: # Choose the one with least color
#      curr_node= min(candidate_list, key= lambda x: len(node_attr_d[x].allowed_banks) + WT_DEGREE_1 * IO_graph.degree(x))
    
    curr_node=  min_col_node
    node_obj= node_attr_d[curr_node]

    # Pick a bank
    if len(node_obj.allowed_banks): # Color available
      picked_bank= random.choice(list(node_obj.allowed_banks))
    else: # No color available
      # NOTE: choose a color that can atleast avoid an output conflict
      # possible_pe = list(node_obj.allowed_pe)[0]
      possible_banks = set([])
      if not node_obj.is_leaf(): # try to pick a bank that avoids output conlfict
        possible_banks = pe_to_banks(out_port_map, node_obj.allowed_pe)
        # banks of other outputs
        bb_obj= map_helper_d[graph[curr_node].storing_builblk].bb_obj
        for n in bb_obj.out_list:
          if n in bank_d:
            possible_banks -= set([bank_d[n]]) # To avoid output conflict. As this bank is already used by another output

      if len(possible_banks) == 0:
        possible_banks = set(total_colors)
      
      assert len(possible_banks) != 0

      if len(possible_banks) != 0:
        # Try to avoid conflicts as much as possible
        neighbor_banks= []
        for neighbor in IO_graph.neighbors(curr_node):
          if neighbor in bank_d:
            neighbor_banks.append( bank_d[ neighbor] )
        
        neighbor_banks = sorted(set(neighbor_banks), key= lambda x : neighbor_banks.count(x) , reverse= True)
        for n in neighbor_banks:
          if len(possible_banks) == 1:
            break
          possible_banks -= set([n])
        picked_bank = possible_banks.pop()
      else:
        # NOTE: The code will never reach here in the new implementation
        picked_bank= random.choice(list(total_colors))
      

    # Pick a PE, if applicable 
    if not node_obj.is_leaf():
      picked_pe= banks_to_pe(map_bank_to_pe, set([picked_bank]))
      if (picked_pe & node_obj.allowed_pe) != set(): # No output conflict
        picked_pe &= node_obj.allowed_pe
        if hw_details.out_port_mode == 'CROSSBAR':
          pass
        else:
          assert len(list(picked_pe)) == 1
        picked_pe = list(picked_pe)[0]
      else:
        assert len(node_obj.allowed_banks) == 0
        output_conflict += 1
        output_conflict_bb_set.add(graph[curr_node].storing_builblk)
        assert len(node_obj.allowed_pe) != 0
        # TODO: Better heuristic than minimize
        picked_pe = random.choice(list(node_obj.allowed_pe))

    # Update allowed_pe for nodes in the same BB. Encodes the constraint of absence of output crossbar
    if node_obj.is_leaf():
      node_obj.allocate_bank(picked_bank)
    else:
      node_obj.allocate_bank(picked_bank, picked_pe)
      map_helper_d[graph[curr_node].storing_builblk].allocate_pe(curr_node, picked_pe, node_attr_d, out_port_map, map_lvl_to_pe)
    
    # Update allowed_pe according to IO_graph conflicts
    for node in IO_graph.neighbors(curr_node):
      node_attr_d[node].allowed_banks -= set([picked_bank])
    
    bank_d[curr_node] = picked_bank
    uncolored_nodes.remove(curr_node)
    
    assert len(bank_d) == IO_graph.number_of_nodes() - len(uncolored_nodes)

    if not node_obj.is_leaf():
      pe_d[curr_node]= picked_pe

    done_nodes.add(curr_node)

    # update the tracking for selection of min_col_node
    map_affected_new_bank_cnt= {n : len(node_attr_d[n].allowed_banks) for n in affected_nodes}
    for n in affected_nodes:
      old_bank_cnt= map_affected_old_bank_cnt[n]
      new_bank_cnt= map_affected_new_bank_cnt[n]
      assert new_bank_cnt <= old_bank_cnt
      map_allowed_bank_cnt_to_set_uncolored_nodes[old_bank_cnt].remove(n)
      map_allowed_bank_cnt_to_set_uncolored_nodes[new_bank_cnt].add(n)
  
  map_all_nodes_to_pe(BB_graph, hw_details)

  output_conflict_new = len(output_conflict_bb_set)
  
  assert len(uncolored_nodes) == 0

  return bank_d, pe_d, output_conflict_new

def map_all_nodes_to_pe(BB_graph, hw_details):
  """
    Once IO nodes are mapped to PEs, map all internal nodes to PE
  """
  n_tree= hw_details.n_tree
  tree_depth= hw_details.tree_depth
  out_port_map= hw_details.out_port_map

  all_pes= set(list(out_port_map.keys()))

  for bb, obj in list(BB_graph.items()):
    all_pes_in_bb= set(all_pes)
    for unit in obj.set_of_decompose_units:
      top_tup= unit.map_output_to_pe[unit.parent]
      
      # pes_for_unit contains possible PEs for this decompose unit, based on the PE of top unit
      pes_for_unit= set([])
      for pe in all_pes_in_bb:
        if pe[0] == top_tup[0] and pe[1] <= top_tup[1]: #Same tree as parent, and lower levels
          start= top_tup[2]*(2**(top_tup[1] - pe[1]))
          end= (top_tup[2]+1)*(2**(top_tup[1] - pe[1]))
          if pe[2] in list(range(start, end)):
            pes_for_unit.add(pe)
      
      all_pes_in_bb -= pes_for_unit # PEs for this unit is no longer available for other units

      local_nodes= unit.uncomputed_nodes_local()
      
      allowed_pe= {node: set([pe for pe in pes_for_unit if pe[1] == unit.lvl_local(node)]) for node in local_nodes} # PEs at same level are allowed PEs
      
      allocated_pe= {}

      # Update allowed PEs for non-output nodes based on PEs of output nodes
      for node in local_nodes:
        if node in unit.map_local_to_pe: 
          pe_tup= unit.map_local_to_pe[node]
          allocated_pe[node]= pe_tup
          update_allowed_pe_local(unit, node, pe_tup, local_nodes, allowed_pe)

      # Allocate PEs to the rest of the nodes
      for node in local_nodes:
        if node not in allocated_pe:
          pe_tup= random.choice(list(allowed_pe[node]))
          allocated_pe[node]= pe_tup
          unit.update_pe_detail_local(node, pe_tup)
          update_allowed_pe_local(unit, node, pe_tup, local_nodes, allowed_pe)

      # Sanity checks
      assert len(list(allocated_pe.keys())) == len(local_nodes)
      for pe_set in list(allowed_pe.values()):
        assert len(pe_set)>0

      sanity_check_unit_mapping_of_pe(unit)

      # Update a common map in bb object for all units
      for pe, global_id in list(unit.map_pe_to_global.items()):
        obj.pe_to_node[pe]= global_id
      
    sanity_check_bb(obj)
     
  
def update_allowed_pe_local(unit, node, pe_tup, local_nodes, allowed_pe):
  """
    Update allowed_pe for other nodes when node is mapped to pe_tup 
  """
  assert pe_tup in allowed_pe[node]
  allowed_pe[node]= set([pe_tup])

  t= pe_tup[0]
  lvl= pe_tup[1]
  pe= pe_tup[2]

  # Ancestors
  ancestors= nx.algorithms.dag.descendants(unit.struct_graph_nx_tree, node)
  for ancestor in ancestors:
    allowed_pe[ancestor] &= allowed_pe_local(unit, pe_tup, node, ancestor, mode='ancestor')
    assert len(allowed_pe[ancestor]) > 0, allowed_pe
  
  # descendant
  descendants= nx.algorithms.dag.ancestors(unit.struct_graph_nx_tree, node)
  descendants= [descendant for descendant in descendants if not unit.is_input_local(descendant)]
  for descendant in descendants:
    allowed_pe[descendant] &= allowed_pe_local(unit, pe_tup, node, descendant, mode='descendant')
    assert len(allowed_pe[descendant]) > 0, allowed_pe
  
  # Neither ancestor nor descendant
  for query_node in local_nodes:
    if (query_node is not node) and (query_node not in ancestors) and (query_node not in descendants):
      common_parent= set(nx.algorithms.dag.descendants(unit.struct_graph_nx_tree, query_node)) & set(ancestors)
      assert len(common_parent) > 0
      common_parent= min(common_parent, key= lambda x: unit.lvl_local(x))
      assert common_parent != query_node
      assert common_parent != node

      common_parent_tup= list(allowed_pe[common_parent])
      assert len(common_parent_tup) == 1
      common_parent_tup= common_parent_tup[0]
      h_lvl= common_parent_tup[1] # Level of common parent
      h_pe= common_parent_tup[2]
      l_lvl= unit.lvl_local(query_node) # Level of query_node
      
      assert h_lvl > l_lvl
      
      # All PEs at l_lvl under common parent
      start= h_pe*(2**(h_lvl-l_lvl))
      end= (h_pe+1)*(2**(h_lvl-l_lvl))
      all_pe_at_l_lvl= set([(t, l_lvl, _pe) for _pe in range(start, end)])
      
      # Remove PEs on the node's side of the tree
      if l_lvl > lvl:
        node_on_node_path_at_l_lvl= pe//(2**(l_lvl-lvl))
      else:
        node_on_node_path_at_l_lvl= pe*(2**(lvl-l_lvl))
      
      factor= 2**(h_lvl-l_lvl-1)
      start= int(node_on_node_path_at_l_lvl// factor)*factor
      end= (int(node_on_node_path_at_l_lvl// factor) + 1) * factor
      not_allowed_pe= set([(t, l_lvl, _pe) for _pe in range(start, end)])

      allowed_pe[query_node] &= all_pe_at_l_lvl - not_allowed_pe
      assert len(allowed_pe[query_node]) > 0, allowed_pe

def sanity_check_bb(bb):
  # BB level sanity checks
  
  # Check Consistency
  for unit in bb.set_of_decompose_units:
    for output,pe in list(unit.map_output_to_pe.items()):
      assert pe == bb.out_nodes_to_pe[output]

  # Checks there are same number of uncomputed nodes as alloted PEs
  local_cnt= 0
  for unit in bb.set_of_decompose_units:
    local_cnt += len(unit.uncomputed_nodes_local())

  pe_set= set([pe for unit in bb.set_of_decompose_units for pe in list(unit.map_pe_to_local.keys())])
  
  assert len(pe_set) == local_cnt
  assert local_cnt == len(list(bb.pe_to_node.keys()))
  assert len(pe_set) == len(list(bb.pe_to_node.keys()))

def sanity_check_unit_mapping_of_pe(unit):
  # Unit level sanity checks for node-to-PE mapping
  
  # No multi-mapping
  assert len(set(unit.map_local_to_pe.values())) == len(unit.uncomputed_nodes_local())

  # Maps are consistent
  assert len(set(unit.map_local_to_pe.values()))  == len(list(unit.map_pe_to_local.keys()))
  for pe, local in list(unit.map_pe_to_local.items()):
    assert unit.map_local_to_pe[local] == pe
  
  # Global output nodes mapped to appropriate PEs
  for global_id, pe in list(unit.map_output_to_pe.items()):
    local_id = unit.output_to_local(global_id)
    assert unit.map_local_to_pe[local_id] == pe

  # All mapped to same tree
  assert len(set([pe[0] for pe in list(unit.map_pe_to_local.keys())])) == 1 

  # Lvl check
  for local, pe in list(unit.map_local_to_pe.items()):
    assert pe[1] == unit.lvl_local(local)
  
  # Parent- Children check
  for local, pe in list(unit.map_local_to_pe.items()):
    parent= list(unit.struct_graph_nx_tree.successors(local))
    if parent:
      assert len(parent) == 1
      parent= parent.pop()
      parent_pe= unit.map_local_to_pe[parent]
      
      parent_pe[0] == pe[0]
      parent_pe[1] == pe[1] + 1
      parent_pe[2] == pe[2]//2

def allowed_pe(unit, main_pe, main_node, query_node, mode):
  """
    returns a set of PEs that are allowed for query_node based on the main_node
  """
  assert mode in ['ancestor', 'descendant']
  t= main_pe[0]
  lvl= main_pe[1]
  pe= main_pe[2]
  
  assert main_node in unit.struct_graph_nx.nodes()
  assert query_node in unit.struct_graph_nx.nodes()

  if mode == 'ancestor':
    # NOTE: our ancestor is NetworkX's descendant
    assert query_node in nx.algorithms.dag.descendants(unit.struct_graph_nx, main_node), "query node is not an ancestor Node"

    h_lvl= unit.lvl_ls(query_node)
    assert len(set(h_lvl)) == 1, "All lvl in lvl_ls should be same"
    h_lvl= h_lvl[0]
    assert h_lvl > lvl, "ancestors should have a higher level"

    _pe= int(pe//(2**(h_lvl-lvl)))
    
    return set([(t, h_lvl, _pe)])
  
  if mode == 'descendant':
    # NOTE: our descendant is NetworkX's ancestor
    assert query_node in nx.algorithms.dag.ancestors(unit.struct_graph_nx, main_node), "query node is not an ancestor Node"
    
    l_lvl= unit.lvl_ls(query_node)
    assert len(set(l_lvl)) == 1, "All lvl in lvl_ls should be same"
    l_lvl= l_lvl[0]
    assert l_lvl < lvl, "ancestors should have a lower level"
    start= pe*(2**(lvl-l_lvl))
    end= (pe+1)*(2**(lvl-l_lvl))
    return set([(t, l_lvl, _pe) for _pe in range(start, end)])
    
def allowed_pe_local(unit, main_pe, main_node, query_node, mode):
  """
    returns a set of PEs that are allowed for query_node based on the main_node. All nodes are local unit nodes here.
  """
  assert mode in ['ancestor', 'descendant']
  t= main_pe[0]
  lvl= main_pe[1]
  pe= main_pe[2]
  assert main_node in unit.struct_graph_nx_tree.nodes()
  assert query_node in unit.struct_graph_nx_tree.nodes()
  
  if mode == 'ancestor':
    # NOTE: our ancestor is NetworkX's descendant
    assert query_node in nx.algorithms.dag.descendants(unit.struct_graph_nx_tree, main_node), "query node is not an ancestor Node"

    h_lvl= unit.lvl_local(query_node)
    assert h_lvl > lvl, "ancestors should have a higher level"

    _pe= int(pe//(2**(h_lvl-lvl)))
    
    return set([(t, h_lvl, _pe)])
  
  if mode == 'descendant':
    # NOTE: our descendant is NetworkX's ancestor
    assert query_node in nx.algorithms.dag.ancestors(unit.struct_graph_nx_tree, main_node), "query node is not an ancestor Node"
    
    l_lvl= unit.lvl_local(query_node)
    assert l_lvl < lvl, "ancestors should have a lower level"
    start= pe*(2**(lvl-l_lvl))
    end= (pe+1)*(2**(lvl-l_lvl))
    return set([(t, l_lvl, _pe) for _pe in range(start, end)])
    
class node_attr():
  def __init__(self, node, graph, BB_graph, map_lvl_to_pe, out_port_map, n_mem_banks):
    self.node= node
    
    if graph[node].storing_builblk is not None:
      self.bb= BB_graph[graph[self.node].storing_builblk]
      self.lvl, self.unit= self.bb.node_to_pe_lvl_and_unit(node)
      self.unit.output_to_local(node)
      self.stored= True
    else:
      self.lvl, self.unit= None, None
      self.stored= False
    
    if self.stored:
      self.allowed_pe= set(map_lvl_to_pe[self.lvl])
#      self.prefered_pe= {pe:0 for pe in self.allowed_pe} # Key: a tuple (t,l,p) ... Val: Preference_cnt
      self.allowed_banks= pe_to_banks(out_port_map, self.allowed_pe)
      self.allocated_pe= None
    else:
      self.allowed_banks= set(range(n_mem_banks))

    self.bank_allocated_flag= False
    self.allocated_bank= None
  
  def is_leaf(self):
    if self.stored:
      return False
    else:
      return True

  def allocate_bank(self, bank, tup= None):
    self.allocated_bank= bank
    self.bank_allocated_flag= True

    if not self.is_leaf():
      assert tup != None
      self.allocated_pe= tup
      self.bb.out_nodes_to_pe[self.node] = tup
      local_id= self.unit.output_to_local(self.node)
      self.unit.update_pe_detail_local(local_id, tup)

class map_helper():
  """
    To help map nodes to physical PEs
    One class object per BB
  """
  def __init__(self, n_tree, tree_depth, bb_obj):
    self.bb_obj= bb_obj
#    self.bb_obj= copy.deepcopy(bb_obj)
    self.n_tree= n_tree
    self.tree_depth= tree_depth
    
    # Keeps track of what kind of trees can still be mapped
    # For eg. [3,3,2,2,2,1,1] shows two trees of depth 3, three of depth 2, and two of depth 1 can still be mapped to this hardware
    #self.available_hw_ls= []
    #for rev_lvl in range(tree_depth):
    #  lvl= tree_depth- rev_lvl
    #  available_hw_ls += [lvl for _ in range(2**rev_lvl)]
    #self.available_hw_ls= available_hw_ls * n_tree
    
    # dict of dict of dict to keep track of assigned nodes to physical PE
    # Key 1: Tree, # Key 2: Lvl, Key 3: PE id
    # Val: Node Id
    self.pe_to_node= {}
    for t in range(n_tree):
      self.pe_to_node[t]= {}
      for rev_lvl in range(tree_depth):
        lvl= tree_depth- rev_lvl
        self.pe_to_node[t][lvl]= {}
        for p in range(2**rev_lvl):
          self.pe_to_node[t][lvl][p]= None
    
#    self.allowed_pe_per_unit= {unit: node_attr_d[unit.parent].allowed_pe for unit in bb_obj.set_of_decompose_units}
    
    # Key: decompose unit
    # Val: PE to which unit's top node is mapped
    self.map_unit_top_to_pe= {}
    
  def allocate_pe(self, node, tup, node_attr_d, out_port_map, map_lvl_to_pe, verb= False):
    t=tup[0]
    lvl= tup[1]
    pe= tup[2]
    

    if verb:     
      print('allocate_pe', node, (t,lvl,pe), self.pe_to_node, 'bb_id:' + str(self.bb_obj.blk_key))

    node_obj= node_attr_d[node]
    assert not node_obj.is_leaf()

    unit= node_obj.unit
    
    assert self.pe_to_node[t][lvl][pe] == None, (tup, node, self.pe_to_node[t][lvl][pe], unit.child_dict)
    self.pe_to_node[t][lvl][pe]= node
    
    struct_graph_nx= unit.struct_graph_nx
    
    if unit in self.map_unit_top_to_pe:
      assert self.map_unit_top_to_pe[unit][0] == tup[0], "UNIT is assigned to a different tree"

    assert tup in node_obj.allowed_pe

    # if it is allocated to allowed pes
    if (tup in node_obj.allowed_pe): # Actually, this if condition should always be true
      # Key of allowed_pe is a local id
      _allowed_pe= {}
      _allowed_pe[unit.output_to_local(node)]= set([tup])

      # assign all ancestors to fixed PE 
      ancestors, ancestors_local= unit.ancestors_of_output(node) 
      for parent in ancestors:
        local_id= set(unit.global_to_local(parent)) & ancestors_local
        assert len(local_id) == 1
        local_id= local_id.pop()
        _allowed_pe[local_id]=  allowed_pe(unit, tup, node, parent, 'ancestor')
        
        # Sanity checks
        if parent in unit.output_nodes and unit.output_to_local(parent) == local_id:
          if node_attr_d[parent].bank_allocated_flag:
            assert node_attr_d[parent].allocated_pe == list(_allowed_pe[local_id])[0], (node_attr_d[parent].allocated_pe, _allowed_pe[local_id])
#            assert self.pe_to_node[t][h_lvl][_pe] == parent
     
      # update pe's for descendants
      descendants, descendants_local= unit.descendants_of_output(node)
      for child in descendants: 
        if child in unit.output_nodes:
          local_id= set([unit.output_to_local(child)]) & descendants_local
          if len(local_id) == 0:
            continue
          
          local_id= local_id.pop()
          _allowed_pe[local_id]= allowed_pe(unit, tup, node, child, 'descendant')

          # Sanity checks
          if node_attr_d[child].bank_allocated_flag:
            assert node_attr_d[child].allocated_pe in _allowed_pe[local_id]

      # update pe's for nodes that are niether descendents nor ancestors
      for output in unit.output_nodes:
        local_id = unit.output_to_local(output)
        if (output is not node) and (local_id not in _allowed_pe):
          assert (local_id not in descendants_local) and (local_id not in ancestors_local)
          common_parent, common_parent_local= unit.common_ancestor_of_outputs(output, node)
          assert common_parent != output
          assert common_parent != node

          allowed_tup_set= list(_allowed_pe[common_parent_local])
          assert len(allowed_tup_set) == 1
          h_lvl= allowed_tup_set[0][1] # Level of common parent
          h_pe= allowed_tup_set[0][2]
          l_lvl= node_attr_d[output].lvl # Level of output node
          
          assert h_lvl > l_lvl

          start= h_pe*(2**(h_lvl-l_lvl))
          end= (h_pe+1)*(2**(h_lvl-l_lvl))
          all_pe_at_lvl= set([(t, l_lvl, _pe) for _pe in range(start, end)])
          
          if l_lvl > lvl:
            node_on_node_path_at_l_lvl= pe//(2**(l_lvl-lvl))
          else:
            node_on_node_path_at_l_lvl= pe*(2**(lvl-l_lvl))
          
          factor= 2**(h_lvl-l_lvl-1)
          start= int(node_on_node_path_at_l_lvl// factor)*factor
          end= (int(node_on_node_path_at_l_lvl// factor) + 1) * factor
          not_allowed_pe= set([(t, l_lvl, _pe) for _pe in range(start, end)])

          _allowed_pe[local_id]= all_pe_at_lvl - not_allowed_pe
          
          # Sanity checks
          if node_attr_d[output].bank_allocated_flag:
            assert node_attr_d[output].allocated_pe in _allowed_pe[local_id], (node, output, unit.child_dict, node_attr_d[output].allocated_pe)
  
      # update attr in node_attr_d for all output nodes
      for output in unit.output_nodes:
        if not node_attr_d[output].bank_allocated_flag:
          local_id = unit.output_to_local(output)
          assert node_attr_d[output].allowed_pe & _allowed_pe[local_id] != set(), (output, node_attr_d[output].allowed_pe, _allowed_pe[local_id])
          node_attr_d[output].allowed_pe &= _allowed_pe[local_id]
          node_attr_d[output].allowed_banks &= pe_to_banks(out_port_map, node_attr_d[output].allowed_pe)
      
      # if this is the first node from the unit
      if (unit not in self.map_unit_top_to_pe):
        # update self.map_unit_top_to_pe
        local_id = unit.output_to_local(unit.parent)
        assert unit not in self.map_unit_top_to_pe
        assert len(list(_allowed_pe[local_id])) == 1, (list(_allowed_pe[local_id]), type(_allowed_pe[local_id]))
        top_tup= list(_allowed_pe[local_id])[0]
        assert top_tup not in list(self.map_unit_top_to_pe.values())
        self.map_unit_top_to_pe[unit]= top_tup
        
        # Update pe for other units
        self.update_allowed_pe_other_units(node_attr_d, map_lvl_to_pe, out_port_map)
    
  def update_allowed_pe_other_units(self, node_attr_d, map_lvl_to_pe, out_port_map):
    """
      First update self.map_unit_top_to_pe before calling this 
    """
    # All possible combinations
    combo= set()
    for t in range(self.n_tree):
      for rev_lvl in range(self.tree_depth):
        lvl= self.tree_depth - rev_lvl
        for p in range(2**rev_lvl):
          combo.add((t,lvl,p))
    
    #--- Remove combinations according to mapped units
    for unit, tup in list(self.map_unit_top_to_pe.items()):
      t,lvl,p = tup[0], tup[1], tup[2]
      self.assign_unit_to_combo(combo, t, lvl, p)
    
    #--- Check if there is a critical scenario, i.e. number of available PEs at a depth is exactly same as number of units of that depth 
    unit_ls= [unit for unit in self.bb_obj.set_of_decompose_units if unit not in self.map_unit_top_to_pe]
    unit_ls.sort(key= lambda x: len(x.child_dict)-1, reverse= True)
    
    depth_ls= list(set([len(x.child_dict)-1 for x in unit_ls]))
    depth_ls.sort(reverse= True)
    map_depth_to_unit= {depth: set([unit for unit in unit_ls if len(unit.child_dict)-1==depth]) for depth in depth_ls}
    
    allowed_pe_per_unit= {}
    for depth in depth_ls: # necessary to loop on a sorted depth_ls
      avail_combo= combo & set(map_lvl_to_pe[depth])
      unit_set= map_depth_to_unit[depth]

      assert len(avail_combo) >= len(unit_set)
      
      for unit in unit_set:
        allowed_pe_per_unit[unit]= avail_combo
        assert len(avail_combo) > 0

      if len(avail_combo) == len(unit_set): # critical case
        avail_combo_copy= set(avail_combo)
        for unit in unit_set:
          # Pop a random tup and remove respective PEs from combo, so that PEs at lower depth cannot take it
          tup= avail_combo_copy.pop()
          self.assign_unit_to_combo(combo, tup[0], tup[1], tup[2])

    # update pe's for every output node in units  
    for unit in unit_ls:
      # None of the output node has been assigned, as unit is still in unit_ls. So all possible pe's are available for all outputs at same lvl
      _allowed_pe_set= {node: set() for node in unit.output_nodes}
      for avail_tup in allowed_pe_per_unit[unit]: # avail_tup corresponds to top node
        t= avail_tup[0]
        lvl= avail_tup[1]
        pe= avail_tup[2]
        for node in unit.output_nodes:
          assert node_attr_d[node].bank_allocated_flag== False

          assert node_attr_d[node].unit == unit, (node, unit.parent, node_attr_d[node].unit.parent, unit.output_nodes)

          l_lvl= node_attr_d[node].lvl  
          start= pe*(2**(lvl-l_lvl))
          end= (pe+1)*(2**(lvl-l_lvl))
          allowed_pe= set([(t, l_lvl, _pe) for _pe in range(start, end)])
          
          _allowed_pe_set[node] |= allowed_pe
      
      for node in unit.output_nodes:
        assert len(node_attr_d[node].allowed_pe & _allowed_pe_set[node])  != 0, [node, node_attr_d[node].allowed_pe, _allowed_pe_set[node]]
        node_attr_d[node].allowed_pe &= _allowed_pe_set[node]
        node_attr_d[node].allowed_banks &= pe_to_banks(out_port_map, node_attr_d[node].allowed_pe)
#        node_attr_d[node].prefered_pe= {pe:cnt for pe, cnt in node_attr_d[node].prefered_pe.items() if pe in node_attr_d[node].allowed_pe}

  def sat(self, set_of_decompose_units, node_attr_d):  
    """
      Restirns all possible mapping of decompose units
    """
    
    # All possible combinations
    combo= set()
    for t in range(self.n_tree):
      for rev_lvl in range(self.tree_depth):
        lvl= self.tree_depth - rev_lvl
        for p in range(2**rev_lvl):
          combo.add((t,lvl,p))
    
    #--- Remove combinations according to mapped units
    for unit, tup in list(self.map_unit_top_to_pe.items()):
      t,lvl,p = tup[0], tup[1], tup[2]
      assign_unit_to_combo(combo, t, lvl, p)
    
    #--- Try different mappings
    # starting from a sorted list, it will always provide a solution
    unit_ls= [unit for unit in set_of_decompose_units if unit not in self.map_unit_top_to_pe]
    unit_ls.sort(key= lambda x: len(x.child_dict), reverse= True)
    
    full_sol= self.sat_recurse(unit_ls, combo)

  def sat_recurse(self, unit_ls, combo):
    unit= unit_ls.pop(0)

    lvl= len(unit.child_dict)-1

    good_tup= set([tup for tup in combo if tup[1]==lvl])
    
    assert len(good_tup) > 0

    full_ls_of_sol= []

    for tup in good_tup:
      combo_copy= set(combo)
      unit_ls_copy= list(unit_ls)
#      combo_copy= combo
      self.assign_unit_to_combo(combo_copy, tup[0], tup[1], tup[2])
      if unit_ls_copy:
        ls_of_sol_dict= self.sat_recurse(unit_ls_copy, combo_copy)
        for sol_dict in ls_of_sol_dict:
          sol_dict[unit]=tup
          full_ls_of_sol.append(sol_dict)
      else:
        full_ls_of_sol.append({unit: tup})

    return full_ls_of_sol      

  def assign_unit_to_combo(self, combo, t, lvl, pe):
    assert (t,lvl,pe) in combo, [(t,lvl,pe), combo, list(self.map_unit_top_to_pe.values())]
    combo.remove((t, lvl, pe))

    # try removing > lvl combinations if available
    for h_lvl in range(lvl + 1, self.tree_depth+1):
      try:
        combo.remove((t,h_lvl,pe//2**(h_lvl-lvl)))
      except KeyError:
        pass

    # Remove lower lvl
    for l_lvl in range(1, lvl):
      start= pe*(2**(lvl-l_lvl))
      end= (pe+1)*(2**(lvl-l_lvl))
      for _pe in range(start, end):
        combo.remove((t,l_lvl,_pe))
    
def update_banks_in_bb(node, picked_bank, allowed_banks, graph, BB_graph, map_node_to_unit, map_bank_to_pe): # Constraints due to no output cross bar
  """
    Update allowed_banks for outputs of the bb of this node
  """
  # Not a leaf node
  storing_builblk= graph[node].storing_builblk 
  if storing_builblk is not None:
    unit= map_node_to_unit[node]
    assert unit in BB_graph[storing_builblk].set_of_decompose_units
    
    # update for curr decompose unit
    pe_set= banks_to_pe(map_bank_to_pe, set([picked_bank]))
    lvl, _ = BB_graph[storing_builblk].node_to_pe_lvl_and_unit(node)
    
    picked_pe= [pe for pe in pe_set if lvl == pe[0]]
    assert len(picked_pe)== 1
    picked_pe= picked_pe[0]
    
    

      
  else: # If node is a leaf node
    x=1

def pe_to_banks(out_port_map, pe_set):
  bank_set= set()
  for pe in pe_set:
    bank_set |= set(out_port_map[pe])

  return bank_set

def banks_to_pe(map_bank_to_pe, bank_set):
  pe_set= set()
  for bank in bank_set:
    pe_set |= set(map_bank_to_pe[bank])

  return pe_set

def create_IO_graph(graph, BB_graph, instr_ls_obj, MEM_LOAD_CONST):
  IO_graph= nx.Graph()
  
  # Add nodes to the IO_graph
  for BB, obj in list(BB_graph.items()):
    IO_graph.add_nodes_from(obj.in_list_unique) # add_nodes_from do not add duplicate nodes, if nodes already present
    IO_graph.add_nodes_from(obj.out_list) # This makes sure that the final node is also added  

  # Add edges among inputs and outputs of a BB
  # Edges among inputs ensures no conflict while reading
  # Edges among outputs ensures no conflict while writing
  max_in_size= 0
  for BB, obj in list(BB_graph.items()):
    if len(obj.in_list_unique) > max_in_size:
      max_in_size = len(obj.in_list_unique)
    # Edges among inputs
    edges_obj= itertools.combinations(obj.in_list_unique, 2)
    IO_graph.add_edges_from(edges_obj)

    # Edges among outputs
    if len(obj.out_list) > 1:
      edges_obj= itertools.combinations(obj.out_list, 2)
      IO_graph.add_edges_from(edges_obj)
  

  # Add edges for leaf input loads
  if MEM_LOAD_CONST:
    for instr in instr_ls_obj.instr_ls:
      if instr.is_type('ld'):
        for node in instr.node_set:
          assert graph[node].is_leaf() 
        edges_obj= itertools.combinations(instr.node_set, 2)
        IO_graph.add_edges_from(edges_obj)
  
  # Sanity Check
  # All outputs and inputs in IO graph
  for obj in list(BB_graph.values()):
    for node in obj.out_list:
      assert node in IO_graph.nodes()
    
    for node in obj.in_list:
      assert node in IO_graph.nodes()

  print("Max number of inputs: ", max_in_size)
  
  return IO_graph

def gen_possible_group_list_recurse(list_of_ordered_lists_sorted, list_of_avail_groups, map_node_to_group, map_decompose_unit_to_group_range, color_d, map_bank_to_groups):  
  """
    Generate all combinations of group mappings
  """

  curr_list= list_of_ordered_lists_sorted.pop(0)
  len_curr_list= len(curr_list)

  curr_list_of_avail_groups= list(list_of_avail_groups)
  
  possible_group_lists= []
  
  min_conflict_cnt= 10000
  min_total_conflict_cnt= 10000
  chosen_assign= None
  chosen_group_idx= None

  input_list_0= curr_list[ : len_curr_list//2]
  input_list_1= curr_list[len_curr_list//2 : ]
  
  #print "curr_list", curr_list
  #print "groups", [[node, map_node_to_group[node]] for node in curr_list if node != None]
  
  for group_idx, avail_groups in enumerate(curr_list_of_avail_groups):
    len_consecutive_groups= avail_groups[1] - avail_groups[0] + 1 
    
    # Early Exit
    if min_total_conflict_cnt == 0:
      break

    if len_consecutive_groups >= len_curr_list and len_consecutive_groups % len_curr_list == 0:      
      for idx in range(len_consecutive_groups//len_curr_list):
        curr_assign= [ avail_groups[0] + idx*len_curr_list , avail_groups[0] + (idx+1)*len_curr_list - 1]
        
        first_group_range_tup= [ curr_assign[0], curr_assign[0]+ len_curr_list//2 - 1 ]
        second_group_range_tup= [ curr_assign[0] + len_curr_list//2, curr_assign[1] ]
        conflicted_nodes= []

        # Exact conflicts for this decompose unit
        cnt= count_group_conflicts_recurse(list(input_list_0), list(input_list_1), first_group_range_tup, second_group_range_tup, map_node_to_group, conflicted_nodes, color_d, map_bank_to_groups)
        
        # Initial estimation of conflixt for all the other decompose units
        other_conflict_cnt= 0
        for other_list in list_of_ordered_lists_sorted:
          for node in other_list:
            if node != None:
              curr_bank= color_d[node]
              curr_groups= map_bank_to_groups[curr_bank]

              if bool(curr_groups & set(range(curr_assign[0],curr_assign[1]+1))):
                other_conflict_cnt += 1 

        assert cnt <= len([node for node in curr_list if node != None]), "There could not be more conflicts than the number of inputs"
          
        #print first_group_range_tup, second_group_range_tup, curr_assign, idx, "cnt:", cnt
        
        if cnt < min_conflict_cnt:
        #if cnt + other_conflict_cnt < min_total_conflict_cnt:
          min_conflict_cnt= cnt
          min_total_conflict_cnt= cnt + other_conflict_cnt
          chosen_assign= list(curr_assign)
          chosen_group_idx= group_idx
        
        # Early Exit
        if min_total_conflict_cnt == 0:
          break
  
  #print 'chosen_assign:', chosen_assign, "min_conflict_cnt", min_conflict_cnt
  
  conflict_cnt= min_conflict_cnt
  
  next_list_of_avail_groups= list(list_of_avail_groups)
  avail_groups=  next_list_of_avail_groups.pop(chosen_group_idx)
  
  new_groups= []
  new_groups.append([avail_groups[0], chosen_assign[0] - 1])
  new_groups.append([chosen_assign[1] + 1, avail_groups[1]])
  
  new_groups= [[group[0], group[1]] for group in new_groups if group[1] >= group[0]]
  #new_groups_copy= list(new_groups)
  #for temp_idx, group in enumerate(new_groups_copy):
  #  if group[1] < group [0]:
  #    del new_groups[temp_idx]
  
  next_list_of_avail_groups = copy.deepcopy(next_list_of_avail_groups + new_groups)
  
  #print 'avail_groups:', next_list_of_avail_groups

  if list_of_ordered_lists_sorted:
    conflict_cnt += gen_possible_group_list_recurse(list_of_ordered_lists_sorted, next_list_of_avail_groups, map_node_to_group, map_decompose_unit_to_group_range, color_d, map_bank_to_groups)
        
  #possible_group_lists += [group_list.insert(0, curr_assign) for group_list in return_possible_group_lists]
  
  return conflict_cnt


def count_group_conflicts_recurse(input_list_0, input_list_1, first_group_range_tup, second_group_range_tup, map_node_to_group, conflicted_nodes, color_d, map_bank_to_groups):
  """
    group_range_tuple: 0th element is the starting group and 1st element is ending group (both inclusive)
    conflicted nodes: Nodes that are known to be in conflict and already accounted for
  """
  
  # NOTE: All the code is prepared assuming possibility of different lenghts of input_list_0 and input_list_1
  # But in actual usage they should be of same length
  assert len(input_list_0) == len(input_list_1)
  
  #assert first_group_range_tup[0] % 2 == 0, "Lowest group should be an even group as groups are meant to be aligned to 2"
  #assert second_group_range_tup[0] % 2 == 0, "Lowest group should be an even group as groups are meant to be aligned to 2"
  assert first_group_range_tup[0] <= first_group_range_tup[1], "Group range should be from lower to higher"
  assert second_group_range_tup[0] <= second_group_range_tup[1], "Group range should be from lower to higher"
  
  first_groups= list(range(first_group_range_tup[0], first_group_range_tup[1] + 1))
  
  second_groups= list(range(second_group_range_tup[0], second_group_range_tup[1] + 1))
  
  assert len(input_list_0) > 0
  assert len(input_list_1) > 0
  assert len(input_list_0) + len(input_list_1) == len(first_groups) + len(second_groups), "Number of groups does not match length of input list"    
  conflicts_list_0_group_0= []
  conflicts_list_0_group_1= []
  #print "in_list_0:",
  for node in input_list_0:
    if node != None:
      #print '[', node,':', color_d[node], map_node_to_group[node], ']' ,
      curr_bank= color_d[node]
      curr_groups= map_bank_to_groups[curr_bank]

      if not bool(curr_groups & set(first_groups)):
        conflicts_list_0_group_0.append(node)
      
      if not bool(curr_groups & set(second_groups)):
        conflicts_list_0_group_1.append(node)
  
  #print " "

  conflicts_list_1_group_0= []
  conflicts_list_1_group_1= []
  #print "in_list_1:",
  for node in input_list_1:
    if node != None:
      #print '[', node,':', color_d[node], map_node_to_group[node] ,']' ,
      curr_bank= color_d[node]
      curr_groups= map_bank_to_groups[curr_bank]

      if not bool(curr_groups & set(first_groups)):
        conflicts_list_1_group_0.append(node)
      
      if not bool(curr_groups & set(second_groups)):
        conflicts_list_1_group_1.append(node)
  
  #print " "

#  groups_in_list_0= set([map_node_to_group[node] for node in input_list_0 if node != None])
#  groups_in_list_1= set([map_node_to_group[node] for node in input_list_1 if node != None])
  
#  conflicts_list_0_group_0= groups_in_list_0 - set(first_groups)
#  conflicts_list_0_group_1= groups_in_list_0 - set(second_groups)
#  conflicts_list_1_group_0= groups_in_list_1 - set(first_groups)
#  conflicts_list_1_group_1= groups_in_list_1 - set(second_groups)
  
  if len(conflicts_list_0_group_0) + len(conflicts_list_1_group_1) < len(conflicts_list_0_group_1) + len(conflicts_list_1_group_0):
    list_0_allowed_groups= first_groups
    list_1_allowed_groups= second_groups
    conflicts_list_0= conflicts_list_0_group_0
    conflicts_list_1= conflicts_list_1_group_1
  else:
    list_1_allowed_groups= first_groups
    list_0_allowed_groups= second_groups
    conflicts_list_0= conflicts_list_0_group_1
    conflicts_list_1= conflicts_list_1_group_0

  
#  conflicts_list_0= groups_in_list_0 - set(list_0_allowed_groups)
#  conflicts_list_1= groups_in_list_1 - set(list_1_allowed_groups)
  
#  conflicted_nodes_0= [node for node in (set(input_list_0) - set([None])) if map_node_to_group[node] in conflicts_list_0]
#  conflicted_nodes_1= [node for node in (set(input_list_1) - set([None])) if map_node_to_group[node] in conflicts_list_1]
  
  conflicted_nodes_0= conflicts_list_0
  conflicted_nodes_1= conflicts_list_1

  # NOTE: Should not be done using sets! Because there could be repetitions, which should be accounted for
  new_conflicted_nodes_0 = [node for node in conflicted_nodes_0 if node not in conflicted_nodes]
  new_conflicted_nodes_1 = [node for node in conflicted_nodes_1 if node not in conflicted_nodes]

  conflict_cnt = len(new_conflicted_nodes_0) + len(new_conflicted_nodes_1)
  
  len_list_0= len(input_list_0)
  len_list_1= len(input_list_1)
   
  if len_list_0 > 1:
    if set(new_conflicted_nodes_0) != set(input_list_0) - set([None]): # All the node conflicts are already accounted for
      conflict_cnt += count_group_conflicts_recurse( input_list_0[0: len_list_0//2], input_list_0[len_list_0//2 : ], [list_0_allowed_groups[0], list_0_allowed_groups[len_list_0//2 -1]], [list_0_allowed_groups[len_list_0//2], list_0_allowed_groups[-1]], map_node_to_group, conflicted_nodes_0, color_d, map_bank_to_groups)

  if len_list_1 > 1:
    if set(new_conflicted_nodes_1) != set(input_list_1) - set([None]): 
      conflict_cnt += count_group_conflicts_recurse( input_list_1[0: len_list_1//2], input_list_1[len_list_1//2 : ], [list_1_allowed_groups[0], list_1_allowed_groups[len_list_1//2 -1]], [list_1_allowed_groups[len_list_1//2], list_1_allowed_groups[-1]], map_node_to_group, conflicted_nodes_1, color_d, map_bank_to_groups)
  
  return conflict_cnt


def redist_colors(IO_graph, max_colors, w_conflict):
  """
    Once we know the number of colors that would be enough to color the graph (not perfectly, but with less conflicts),
    We can randomize the color selection to make it more uniform.

    This code is inspired by the open-source graph color code in networkx python package
  """
  
  assert w_conflict >0 and w_conflict <= max_colors and isinstance(w_conflict, int), 'w_conflict should be an int in the range [1,total_banks]. It specified how many banks to look at in the sorted list'

  #-- Dict to map nodes to color
  #-- Key: node
  #-- Val: a color -- an int within [0,max_colors)
  color_d={}
  
  total_colors= set(range(max_colors))
  
  conflicts= 0
  
  # Sort the nodes with largest degree first
  # nodes is a list
  nodes= nx.algorithms.coloring.strategy_largest_first(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_random_sequential(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_smallest_last(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_independent_set(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_connected_sequential_bfs(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_connected_sequential_dfs(IO_graph, {})
  #nodes= nx.algorithms.coloring.strategy_saturation_largest_first(IO_graph, {})
 
  #print [IO_graph.degree(x) for x in nodes[:500]]

  total_colors_count= defaultdict(int)

  for node in nodes:
    neighbor_colors= set()
    neighbor_color_count= defaultdict(int)

    for v in list(IO_graph.neighbors(node)):
      if v in color_d:
        neighbor_colors.add(color_d[v])
        neighbor_color_count[color_d[v]] += 1
    
    assert len(neighbor_colors) == len(neighbor_color_count)

    avail_colors= total_colors - neighbor_colors

    if len(avail_colors) == 0: # No color available
      conflicts += 1
      # Pick the color that is used by least neighbors
      sorted_colors= sorted(list(neighbor_color_count.keys()), key= lambda x: neighbor_color_count[x])
      picked_color= sorted_colors[0]

      # Assign any color at random
      #picked_color= random.sample(total_colors,1)[0]
      

    else: # Colors available to assign
      sorted_colors= sorted(list(avail_colors), key= lambda x: total_colors_count[x])
      picked_color= min(sorted_colors[:w_conflict])
      
      #picked_color= min(list(avail_colors))
      
      # Assign any color at random
      #picked_color= random.sample(avail_colors,1)[0]
   
    # To compare the efficiency of our algorithm. COMMENT THIS IN NORMAL EXECUTION!
#    picked_color= random.choice(list(total_colors))

    color_d[node]= picked_color
    total_colors_count[picked_color] += 1

  #print "Total conflicts after redistribution: ", conflicts
  
  return color_d

def count_conflicts(BB_graph, IO_graph, instr_ls_obj, color_d, total_colors, hw_details, mode="plain", pe_d= {} ):
  """
    Counts total number of BBs with conflicts in input, output and total 
  """
  assert mode in ["plain", "out_port_map"]

  cnt_in= 0
  cnt_out=0
  
  node_to_cnt_d= defaultdict(lambda : 0)

  for BB, obj in list(BB_graph.items()):
    set_N= set()
    set_C= set()
    list_C= []
    for idx, in_node in enumerate(obj.in_list_unique):
      set_N.add(in_node)
      color= color_d[in_node]
      set_C.add(color)
      list_C.append(color)
    
    max_conflict_cnt = 0
    conflicted_color= set()
    for color in set_C:
      curr_cnt= list_C.count(color) - 1
      if curr_cnt > 0:
        conflicted_color.add(color)
      if curr_cnt > max_conflict_cnt:
        max_conflict_cnt = curr_cnt 

    for in_node in obj.in_list_unique:
      color= color_d[in_node]
      if color in conflicted_color:
        node_to_cnt_d[in_node] += 1
    
    #if len(set_N) != len(set_C):
    #  cnt_in += 1
    cnt_in += max_conflict_cnt
    
    set_N= set()
    set_C= set()
    list_C= []

    for out_node in obj.out_list:
      set_N.add(out_node)
      color= color_d[out_node]
      set_C.add(color)
      list_C.append(color)

    max_conflict_cnt = 0
    conflicted_color= set()
    for color in set_C:
      curr_cnt= list_C.count(color) - 1
      if curr_cnt > 0:
        conflicted_color.add(color)
      if curr_cnt > max_conflict_cnt:
        # print(f"conflict: {list_C} {obj.out_list}")
        max_conflict_cnt = curr_cnt 
    
    #if len(set_N) != len(set_C):
    #  cnt_out += 1
    cnt_out += max_conflict_cnt

    if mode == 'out_port_map':
      for out_node in obj.out_list:
        color= color_d[out_node]
        pe = pe_d[out_node]
        if color not in hw_details.out_port_map[pe]:
          if color not in conflicted_color:
            cnt_out += 1
  
  cnt_ld= 0
  for instr in instr_ls_obj.instr_ls:
    if instr.is_type('ld'):
      set_N= instr.node_set
      list_C= [color_d[node] for node in set_N]
      set_C= set(list_C)
  
      max_conflict_cnt = 0
      for color in set_C:
        curr_cnt= list_C.count(color) - 1
        if curr_cnt > max_conflict_cnt:
          max_conflict_cnt = curr_cnt 
      
      cnt_ld += max_conflict_cnt

  total_conflicts= cnt_in + cnt_out + cnt_ld
  
  print('cnt_in, cnt_out, cnt_ld:', cnt_in, cnt_out, cnt_ld)
  top_nodes= sorted(list(node_to_cnt_d.keys()), reverse= True, key= lambda x : node_to_cnt_d[x]) [ : 100]
  # print(f'Top nodes involved in most input conflicts: {[(n , node_to_cnt_d[n]) for n in top_nodes]}')
  # Sanity check
  node_cnt =0
  for node in IO_graph.nodes():
    if IO_graph.degree(node) > total_colors:
      node_cnt += 1
  # assert node_cnt > total_conflicts, [node_cnt, total_conflicts]

  return total_conflicts, cnt_in, cnt_out, cnt_ld

def assert_no_conflicts(BB_graph, color_d):
  """
   Verifies that there are no bank conflicts
  """
  for BB, obj in list(BB_graph.items()):
    set_N= set()
    set_C= set()
    for in_node in obj.in_list_unique:
      set_N.add(in_node)
      set_C.add(color_d[in_node])
    assert len(set_N) == len(set_C)

    set_N= set()
    set_C= set()
    if len(obj.out_list) > 1:
      for in_node in obj.out_list:
        set_N.add(in_node)
        set_C.add(color_d[in_node])

      assert len(set_N) == len(set_C)

def bank_dist(color_d):
  """
   Prints distribution of banks (or colors)
  """
  color_lst= list(color_d.values())
  for color in range(max(color_lst)+1):
    print("# of nodes at Bank", color , ":",color_lst.count(color))

def update_siblings_prefer_cnt(graph, node, chosen_group, dict_of_list_of_ordered_in_list, prefer_cnt, prefer_cnt_adv, done_dict, n_groups):
  """
    Update prefer count for all the siblings of all the BBs
  """
  
  parent_BBs= graph[node].parent_buildblk_lst
  
  done_nodes= set()

  for BB in parent_BBs:
    list_of_lists= dict_of_list_of_ordered_in_list[BB]
    
    # Check which decompose_units has current node as an input
    # And choose the candidate with maximum length
    candidate_list_len = 0
    candidate_list= None
  
    for input_list in list_of_lists:
      if node in input_list:
        if len(input_list) > candidate_list_len:
          candidate_list = input_list
          candidate_list_len = len(input_list)
    
    # Update preference of siblings in candidate_list
    list_copy= list(candidate_list)
    list_copy= rearrange_list(node, list_copy)
     
    mod_factor = 2
    
    while len(list_copy) >= mod_factor:

      if chosen_group % mod_factor < mod_factor//2:
        first_group= int(chosen_group//mod_factor)*mod_factor + mod_factor//2
        last_group= int(chosen_group//mod_factor)*mod_factor + mod_factor - 1
      else:
        first_group= int(chosen_group//mod_factor)*mod_factor
        last_group= int(chosen_group//mod_factor)*mod_factor + mod_factor//2 - 1

      for idx in range(mod_factor//2, mod_factor):
        node_to_update= list_copy[idx] 
        
        if node_to_update == None or done_dict[node_to_update] or node_to_update in done_nodes:
          continue
        
        for group_idx in range(first_group, last_group + 1):
          if node_to_update not in prefer_cnt:
            prefer_cnt[node_to_update]= {group: 0 for group in range(n_groups)}
            prefer_cnt_adv[node_to_update]= {}

          if BB not in prefer_cnt_adv[node_to_update]:
            prefer_cnt_adv[node_to_update][BB]= {group: 0 for group in range(n_groups)}
          
          prefer_cnt[node_to_update][group_idx] += 1
          prefer_cnt_adv[node_to_update][BB][group_idx] += 1
          done_nodes.add(node_to_update)

      mod_factor *= 2 
      
    available_groups= list(range(n_groups))
    for group_idx in range(int(chosen_group//mod_factor)*mod_factor, (int(chosen_group//mod_factor) + 1)*mod_factor):
      available_groups.remove(group_idx)

    rest_of_all= list(list_of_lists)
    rest_of_all.remove(candidate_list)

    for input_list in rest_of_all:
      for node_to_update in input_list:
        if node_to_update == None or done_dict[node_to_update] or node_to_update in done_nodes:
          continue
        
        for group_idx in available_groups:
          if node_to_update not in prefer_cnt:
            prefer_cnt[node_to_update]= {group: 0 for group in range(n_groups)}
            prefer_cnt_adv[node_to_update]= {}

          if BB not in prefer_cnt_adv[node_to_update]:
            prefer_cnt_adv[node_to_update][BB]= {group: 0 for group in range(n_groups)}
          
          prefer_cnt[node_to_update][group_idx] += 1
          prefer_cnt_adv[node_to_update][BB][group_idx] += 1
          done_nodes.add(node_to_update)

def rearrange_list(node, input_list):
  """
    Rearrange list elemnts to bring node in position 0
  """

  tot_elements= len(input_list)
  
  curr_idx= input_list.index(node) # If duplicates, return the first index
  
  div_factor= 1
  curr_list= input_list

  while curr_idx != 0:
    #print curr_idx, curr_list
    pivot = tot_elements//(2**div_factor)
    
    assert pivot >= 1, "input list should have a length of multiple of 2, %s %s %s" % (tot_elements, div_factor, pivot)

    if curr_idx >= pivot:  # Exchange two halfs of the list
      #print "replace", pivot, tot_elements, dict_of_list_of_ordered_in_list
      new_list= []
      
      new_list= curr_list[pivot:2*pivot] + curr_list[0:pivot]
      #print new_list
       
      if tot_elements > 2*pivot:
        new_list= new_list + curr_list[2*pivot : ]
      
      #print new_list

      curr_list= new_list
      curr_idx -= pivot

    div_factor += 1
  
  return curr_list
  
def choose_group(node, prefer_cnt, prefer_cnt_adv, n_groups, n_banks_per_group):
  """
    Choose a group based on preference count
  """
  
  # cnt_dict
  # Key: group ID
  # Val: Cnt
  if node in prefer_cnt:
    cnt_dict= prefer_cnt[node]
  else:
    cnt_dict= {group:0 for group in range(n_groups)}
  
  # Advanced way of choocing group
  cnt_dict_adv= {group:0 for group in range(n_groups)}
  BB_cnt_dict= {}
  if node in prefer_cnt_adv:
    BB_cnt_dict= prefer_cnt_adv[node]
    
  for BB, local_cnt_dict in list(BB_cnt_dict.items()):
    local_max_cnt= max(local_cnt_dict.values())
    local_best_group_list= [group for group, cnt in list(local_cnt_dict.items()) if cnt == local_max_cnt]
    local_chosen_group= random.choice(local_best_group_list)  # Random tie-breaker
    cnt_dict_adv[local_chosen_group] += 1
  
  #print cnt_dict
  #cnt_dict= cnt_dict_adv

  max_prefer_cnt= max(cnt_dict.values())
  
  best_group_list= [group for group, cnt in list(cnt_dict.items()) if cnt == max_prefer_cnt]
  
  non_zero_groups= [group for group, cnt in list(cnt_dict.items ()) if cnt != 0]
  

  #sets_of_group_with_same_bank=

  #for group in non_zero_groups:


  chosen_group= random.choice(best_group_list)  # Random tie-breaker
  
  return chosen_group


def create_ordered_input_list(BB_graph):
  """
    For assigning bank groups (To reduce crossbar size by always making sure that inputs are in appropriate bank),
    we need an ordered list of inputs for every decompose_unit

    Ordered list mean that: 
      Two direct siblings should be adjacent to each other
      Counsins that are closer in the tree, should also be closer in the ordered list
  """
  
  # Key: BB ID
  # Val: A list of ordered input list. Each list corresponds to a decompose unit in that BB
  dict_of_list_of_ordered_in_list= {}

  for BB_key, BB in list(BB_graph.items()):
    list_of_oredered_lists= []
    
    # Also populate this global variable
    BB.map_decompose_unit_to_ordered_input_list= {}

    for decompose_unit in BB.set_of_decompose_units:
      parent= decompose_unit.parent
      struct_graph= decompose_unit.struct_graph
      total_levels= len(decompose_unit.child_dict)
      
      ordered_input_list= [None] * (2**(total_levels-1))
      curr_idx= 0
      curr_level= total_levels
      input_list= decompose_unit.input_list

      create_ordered_input_list_recurse(struct_graph, parent, ordered_input_list, curr_idx, curr_level, total_levels, input_list)
      
      BB.map_decompose_unit_to_ordered_input_list[decompose_unit]= ordered_input_list

      list_of_oredered_lists.append(list(ordered_input_list))
   
    dict_of_list_of_ordered_in_list[BB_key]= list_of_oredered_lists
  
  return dict_of_list_of_ordered_in_list

def create_ordered_input_list_recurse(struct_graph, node, ordered_input_list, curr_idx, curr_level, total_levels, input_list):
  
  # curr_level == 1 corresponds to leaf level, which should all have computed nodes
  assert curr_level > 0 , "curr_level can be at minimum become 1"
  
  assert curr_idx < len(ordered_input_list)
  
  if len(struct_graph[node]) == 0: # This is input
    assert node in input_list, "Decompose_unit is not constructed properly. struct_graph and input_list are incosistent"
    
    ordered_input_list[curr_idx]= node
    curr_idx += 2** (curr_level - 1) 

    return curr_idx
  
  curr_level -= 1
  for child in struct_graph[node]:
    curr_idx= create_ordered_input_list_recurse(struct_graph, child, ordered_input_list, curr_idx, curr_level, total_levels, input_list)
  
  return curr_idx


def small_crossbar_B(graph, BB_graph, IO_graph, n_mem_banks):
  """
    Assign banks in a way that use small crossbar B
    Also, try to restrict crossbar A's size 
  """

  #-- Dict to map nodes to color
  #-- Key: node
  #-- Val: a color -- an int within [0,max_colors)
  color_d={}

  nodes= nx.algorithms.coloring.strategy_largest_first(IO_graph, {})

  total_colors= set(range(n_mem_banks))
  
  # Dict that map nodes to possible colors
  # Key: node ID, Val: possible colors
  avail_col_dict= {}
  for node in IO_graph:
    avail_col_dict[node]= set(range(n_mem_banks))


  while len(avail_col_dict) > 0:
    min_col_cnt= len(min(list(avail_col_dict.values()), key= len))
    MAGIC_NUM= 2
    min_col_list= [node for node in list(avail_col_dict.keys()) if len(avail_col_dict[node]) <= min_col_cnt + MAGIC_NUM]

    curr_node= max(min_col_list, key= lambda x: IO_graph.degree(x))
#    curr_node= nodes.pop(0)
    #print curr_node, avail_col_dict[curr_node]
    if len(avail_col_dict[curr_node]): # Color available
      picked_color= random.sample(avail_col_dict[curr_node],1)[0]
    else: # No color available
      picked_color= random.sample(total_colors,1)[0]

    color_d[curr_node]= picked_color

    # remove colors from neighbors
    for v in list(IO_graph.neighbors(curr_node)):
      if v in avail_col_dict:
        avail_col_dict[v] -= set([picked_color])

    # remove colors from sibling outputs
    storing_bb= graph[curr_node].storing_builblk
    if storing_bb is not None:
      BANK_NUM= 4
      start_col= picked_color- picked_color%BANK_NUM
      col_occupied= set(range(start_col, start_col+BANK_NUM))
      for out in BB_graph[storing_bb].out_list:
        if out is not curr_node:
          if out in avail_col_dict:
            avail_col_dict[out] -= col_occupied


    del avail_col_dict[curr_node]

  # Count conflicts for sibling outputs
  count_conflicts_sibling_out = 0
  for bb, bb_obj in list(BB_graph.items()):
#    print 'BB'
    CONFLICT= False
    for out_i in bb_obj.out_list:
#      print out_i, color_d[out_i], '|', 
      for out_j in bb_obj.out_list:
        if out_i != out_j:
          col_set_i= color_d[out_i] - color_d[out_i]%BANK_NUM
          col_set_j= color_d[out_j] - color_d[out_j]%BANK_NUM
          if col_set_i== col_set_j:
            CONFLICT= True
            break
#            count_conflicts_sibling_out += 1
#            print 'Here'
#    print ' ' 
    if CONFLICT:
      count_conflicts_sibling_out += 1
#  count_conflicts_sibling_out //= 2

  printcol('NEW conflict count sibling: ' + str(count_conflicts_sibling_out), 'blue')

  return color_d
