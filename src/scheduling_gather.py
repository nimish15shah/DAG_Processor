import time
import queue
import random
import networkx as nx
from collections import defaultdict
import copy
import itertools

#**** imports from our codebase *****
from . import common_classes
from .common_classes import io_node_details
from . import scheduling
from . import reporting_tools
from . import bank_allocate
from . import useful_methods
from .useful_methods import printcol, printlog
from . import pipeline

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def assign_reverse_level_graph(graph, graph_nx):
  for key, obj in list(graph.items()):
    # reset reverse_level
    obj.reverse_level= None
  
  # Children before parents in the topological_list
  topological_list= nx.algorithms.dag.topological_sort(graph_nx)

  for node in topological_list:
    obj= graph[node]

    lvl= 0
    for child in obj.child_key_list:
      if graph[child].reverse_level >= lvl:
        lvl= graph[child].reverse_level + 1

    obj.reverse_level = lvl

  # Sanity check
  for node, obj in list(graph.items()):
    child_lvl= set()
    for child in obj.child_key_list:
      child_lvl.add(graph[child].reverse_level)
    
    if len(child_lvl) != 0:
      assert obj.reverse_level - 1 in child_lvl

def assign_sch_reverse_lvl_wise(BB_graph, head_bb, map_sch_lvl_to_bb):
  curr_level= 0
  sch_level= 0
  while BB_graph[head_bb].sch_level == None:
    for bb, obj in list(BB_graph.items()):
      if obj.reverse_level == curr_level:
        obj.sch_level= sch_level
        map_sch_lvl_to_bb[sch_level]= bb
        sch_level += 1
    curr_level += 1

def instruction_gen(net, graph, BB_graph, global_var, leaf_list, output_node, misc, hw_details, 
    # w_conflict, MEM_LOAD_CONST, base_scratch_pad_addr, last_scratch_pad_addr, base_param_addr, last_param_addr, SCHEDULING_SEARCH_WINDOW, RANDOM_BANK_ALLOCATE, 
    write_asm= False, make_vid= False, verbose= False):
  
  printlog('ALERT!!! SCHEDULING_SEARCH_WINDOW: ' + str(hw_details.SCHEDULING_SEARCH_WINDOW), 'red')

  n_pipe_stages= hw_details.n_pipe_stages
  n_mem_banks= hw_details.n_banks
  reg_bank_size= hw_details.reg_bank_depth

  w_conflict               = hw_details.w_conflict               
  MEM_LOAD_CONST           = hw_details.MEM_LOAD_CONST           
  base_scratch_pad_addr    = hw_details.base_scratch_pad_addr    
  # NOTE: FIXME: Modifying the scratchpad to cover the entire datamemory to avoid going out-of-memory. 
  # Revert back, unless an execution may overwrite param memory and a second execution cannot be performed.
  # last_scratch_pad_addr    = hw_details.last_scratch_pad_addr    
  last_scratch_pad_addr    = hw_details.last_param_addr    
  base_param_addr          = hw_details.base_param_addr          
  last_param_addr          = hw_details.last_param_addr          
  SCHEDULING_SEARCH_WINDOW = hw_details.SCHEDULING_SEARCH_WINDOW 
  RANDOM_BANK_ALLOCATE     = hw_details.RANDOM_BANK_ALLOCATE     
 

  if write_asm:
    # clear the asm file
    fp= open(global_var.ASM_FILE, 'w+')
    fp.close()
  
  instr_ls_obj= common_classes.instr_ls()

  # Pipeline aware scheduing
  bb_seq= BB_scheduling_gather(graph, BB_graph, n_pipe_stages, instr_ls_obj)
  
  # Add initial loads
  initial_ld_cnt, map_param_to_addr= add_initial_loads(net, graph, BB_graph, instr_ls_obj, n_mem_banks, leaf_list, base_param_addr, last_param_addr)

  # Allocate banks
  bank_d, pe_d, num_of_banks, IO_graph, conflicts, conflicts_in, conflicts_out, conflicts_ld = bank_allocate.bank_allocate(net, graph, BB_graph, n_mem_banks, hw_details, instr_ls_obj, w_conflict, MEM_LOAD_CONST)
  
  # NOTE: RANDOM_BANK_ALLOCATE can lead to errors in extra_load_stores because of force invalidates for the "sh" instructions in the insert_invalidates function. 
  # I have no clue how that force invalidate worked in the first place!!
  if RANDOM_BANK_ALLOCATE: # For comparison
    printlog('\n\nALERT!!! RANDOM BANK ALLOCATION', 'red')
    bank_d= {node: random.choice(list(range(n_mem_banks))) for node in list(bank_d.keys())}

  # FIXME: sch_level is not updated properly
  # Perform liveness analysis
  # liveness_analysis(graph, BB_graph, IO_graph, bb_seq, bank_d, num_of_banks, global_var)
  # exit(1)
  
  # Add shuffles
  insert_shuffle(BB_graph, instr_ls_obj, bank_d, hw_details, n_mem_banks)

  # update bank information in map_param_to_addr
  update_bank_for_param(map_param_to_addr, instr_ls_obj)

  # instruction rescheduling for hazardfree operation after inserting initial loads and shuffles
  instr_ls_obj.instr_ls= pipeline.reschedule_for_hazardfree(instr_ls_obj.instr_ls, n_pipe_stages - 1, n_mem_banks, SCHEDULING_SEARCH_WINDOW)
  printlog("Total cycles fro hazardfree operation after inserting ls and sh: " + str(len(instr_ls_obj.instr_ls)))  

  # Invalidate information
  # NOTE: BBs cannot be shuffled across each other after invalidates are added
  insert_invalidates(graph, BB_graph, instr_ls_obj, RANDOM_BANK_ALLOCATE)
  
  verbose= False
  if verbose:
    for instr in instr_ls_obj.instr_ls:
      if instr.is_type('nop'):
        pass
      elif instr.is_type('ld') or instr.is_type('ld_sc'):
        print('ld or ld_sc', [(node, obj.bank) for node, obj in list(instr.node_details_dict.items())], instr.id)
      elif instr.is_type('sh'):
        print('sh', instr.sh_dict_bank, instr.invalidate_node_set, instr.id)
      elif instr.is_type('bb'):
        print('bb', [(node, obj.bank) for node, obj in list(instr.in_node_details_dict.items())], [(node, obj.bank) for node, obj in list(instr.out_node_details_dict.items())], instr.invalidate_node_set, instr.id)
      else:
        assert False
  
    print("END of instruction description")

  # Spilling instruction: Add load-stores for limited reg bank size
  extra_load_stores(net, global_var, graph, BB_graph, instr_ls_obj, num_of_banks, reg_bank_size, base_scratch_pad_addr, last_scratch_pad_addr, n_pipe_stages, verb= verbose)  

  pipeline.sanity_check_hazard(instr_ls_obj.instr_ls, num_of_banks, n_pipe_stages - 1)

  # Add PE details
  add_pe_details(graph, BB_graph, instr_ls_obj, hw_details)
  
  # TODO: Possible merge shuffle instructions
  
  # print instruction details
  print_instr_breakup(global_var, net, instr_ls_obj, conflicts_in, conflicts_out, conflicts_ld, hw_details)

  # map final output nodes to addr
  map_output_to_addr = get_final_output_addr(instr_ls_obj, output_node)
  insert_final_store(instr_ls_obj, output_node, n_pipe_stages)
  
  return instr_ls_obj, map_param_to_addr, map_output_to_addr

def insert_final_store(instr_ls_obj, output_node, n_pipe_stages):
  last_instr= instr_ls_obj.instr_ls[-1]
  assert last_instr.is_type('bb')

  assert output_node in last_instr.out_node_details_dict, f"Output node {output_node}, produced nodes: {last_instr.produced_nodes}"

  bank= last_instr.out_node_details_dict[output_node].bank
  pos= last_instr.out_node_details_dict[output_node].pos

  instr = common_classes.st_instr()
  instr.node_set= set([output_node])
  instr.mem_addr = 0

  obj = io_node_details( output_node )
  obj.bank = bank
  obj.pos = pos
  instr.node_details_dict= {output_node : obj}

  for n in range(n_pipe_stages):
    instr_ls_obj.instr_ls.append(common_classes.nop_instr())
  instr_ls_obj.instr_ls.append(instr)

def get_final_output_addr(instr_ls_obj, output_node):
  last_instr= instr_ls_obj.instr_ls[-1]

  # last_instr.print_details()

  assert last_instr.is_type('bb')

  assert output_node in last_instr.out_node_details_dict, f"Output node {output_node}, produced nodes: {last_instr.produced_nodes}"

  bank= last_instr.out_node_details_dict[output_node].bank
  pos= last_instr.out_node_details_dict[output_node].pos

  map_output_to_addr= {}
  map_output_to_addr[output_node] = tuple([bank, pos])

  logger.info(f"output node {output_node} will be stored in bank: {bank} and pos: {pos}")

  return map_output_to_addr

def update_bank_for_param(map_param_to_addr, instr_ls_obj):
  """
    Call this function only after shuffles have been inserted and ld does not have any conflicts left
  """
  for instr in instr_ls_obj.instr_ls:
    if instr.is_type('ld'):
      for node, obj in list(instr.node_details_dict.items()):
        addr_bank_tup= map_param_to_addr[node]
        addr_bank_tup[1]= obj.bank

  # Sanity
  for node, addr_bank_tup in list(map_param_to_addr.items()):
    assert addr_bank_tup[1] != None

def add_pe_details(graph, BB_graph, instr_ls_obj, hw_details):
  """
    Add details for PEs
  """
  out_port_map= hw_details.out_port_map
  
  for instr in instr_ls_obj.instr_ls:
    done_child= set() # For sanity check
    if instr.is_type('bb'):
      bb_obj= BB_graph[instr.bb_id]
      
      active_pes= set(list(bb_obj.pe_to_node.keys()))
      # Create pe_details object for all possible PEs
      for pe in list(out_port_map.keys()):
        pe_obj= common_classes.pe_details(pe)

        # Set operations
        if pe in bb_obj.pe_to_node:
          node= bb_obj.pe_to_node[pe]
          pe_obj.node= node

          if graph[node].is_sum():
            pe_obj.set_op('sum')
          elif graph[node].is_prod():
            pe_obj.set_op('prod')
          else:
            assert False
          
        else: # pe not mapped to a node
          descendent_pes = hw_details.get_descendent_pes(pe)
          if len(descendent_pes & active_pes) == 0:
            # pe_obj.set_op('pass_1') # indication that pe is supposed to be inactive
            pe_obj.set_op('pass_1')
          else:
            pe_obj.set_op('pass_0')

        instr.pe_details[pe]= pe_obj

      # Set output bank,pos
      for node, pe in list(bb_obj.out_nodes_to_pe.items()):
        assert node in bb_obj.out_list
        bank= instr.out_node_details_dict[node].bank
        pos= instr.out_node_details_dict[node].pos
        
        assert bank in out_port_map[pe]
        instr.pe_details[pe].output_reg= (bank, pos)

      # set input bank,pos
      for pe, node in list(bb_obj.pe_to_node.items()):
        for child_idx, child in enumerate(list(set(graph[node].child_key_list) & set(bb_obj.in_list))):
          bank= instr.in_node_details_dict[child].bank
          pos= instr.in_node_details_dict[child].pos
          
          FIRST_PATH= False
          SECOND_PATH= False
          if child_idx == 0: # First child
            # For higher PEs, Check which of the children path is free
            if pe[1] > 1:
              if instr.pe_details[(pe[0], pe[1]-1, 2*pe[2])].is_pass_0(): # First child PE is not sum or product, so first path is available
                FIRST_PATH= True
              else: # have to choose second path, because first path is occupied by some operating active nodes
                SECOND_PATH= True
            else:
              FIRST_PATH= True
          else:
            SECOND_PATH= True
          
          if FIRST_PATH:
            target_pe=  (pe[0], 1, pe[2] * (2**(pe[1]-1))) # PE at the 1st level, on first path
          else:
            assert SECOND_PATH == True
            target_pe=  (pe[0], 1, pe[2] * (2**(pe[1]-1)) + int((2**(pe[1]-2)))) # PE at the 1st level, on second path
          
          if target_pe != pe: 
            assert pe[1] > 1, [target_pe, pe]
            assert instr.pe_details[target_pe].is_pass_0(), "When PEs are at higher level, target_pe should perform pass_0"
            instr.pe_details[target_pe].input_0_reg= (bank,pos)
          else: # pe at first level
            assert pe[1] == 1
            if FIRST_PATH:
              instr.pe_details[target_pe].input_0_reg= (bank,pos)
            else:
              instr.pe_details[target_pe].input_1_reg= (bank,pos)

          done_child.add(child)
      
      assert len(done_child) == len(set(bb_obj.in_list))


def map_nodes_to_pe(BB_graph, bank_d, hw_details):
  """
    Maps logical AC nodes to physical PE
    Takes into account bank-allocation info
  """
  x= 1 

def print_instr_breakup_1(instr_ls):
  """
    Display instruction breakup
  """
  print("Total cycles:", len(instr_ls)) 

  print("Node blocks:", len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('bb')]))
  
  initial_ld_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('ld')])
  intermediate_ld_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('ld_sc')])
  print("Load:", intermediate_ld_cnt + initial_ld_cnt)
  print("  initial:", initial_ld_cnt, ", intermediate:", intermediate_ld_cnt)

  print("Store:", len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('st')]))

  shift_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('sh')])
  print("Shifts: ", shift_cnt)
  print("NOP:", len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('nop')]))

  print(" ")

def print_instr_breakup(global_var, net, instr_ls_obj, conflicts_in, conflicts_out, conflicts_ld, hw_details):
  """
    Display instruction breakup
  """
  print(" ")
  printcol("Insruction breakup", 'blue')
  
  conflicts= conflicts_in + conflicts_out + conflicts_ld

  print("Total cycles:", len(instr_ls_obj.instr_ls)) 

  print("Node blocks:", len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('bb')]))
  
  initial_ld_cnt= len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('ld')])
  intermediate_ld_cnt= len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('ld_sc')])
  print("Load:", intermediate_ld_cnt + initial_ld_cnt)
  print("  initial:", initial_ld_cnt, ", intermediate:", intermediate_ld_cnt)

  print("Store:", len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('st')]))

  print("Conflicts:", conflicts)
  print("  in:", conflicts_in, ", out:", conflicts_out, ", ld:", conflicts_ld)
  shift_cnt= len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('sh')])
  print("Shifts: ", shift_cnt)
  print("NOP:", len([idx for idx, obj in enumerate(instr_ls_obj.instr_ls) if obj.is_type('nop')]))

  print(" ")
  
  instr_ls = instr_ls_obj.instr_ls
  total= len(instr_ls)
  bb= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('bb')])
  initial_ld_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('ld')])
  intermediate_ld_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('ld_sc')])
  store= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('st')])
  shift_cnt= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('sh')])
  nop= len([idx for idx, obj in enumerate(instr_ls) if obj.is_type('nop')])
  
  msg= f"net, {net.replace( '/', '_')}, tree_depth, {hw_details.max_depth}, n_banks, {hw_details.n_banks}, reg_bank_depth, {hw_details.reg_bank_depth}, total, {total}, bb, {bb}, initial_ld, {initial_ld_cnt}, intermediate_ld, {intermediate_ld_cnt}, intermediate_st, {store}, shift, {shift_cnt}, nop, {nop}, fitness_wt_distance, {hw_details.fitness_wt_distance}, out_port_mode, {hw_details.out_port_mode}"

  fp= open(global_var.REPORTS_PATH + 'instr_breakup.txt', 'a+')
  print(msg, file=fp, flush= True)
  fp.close()

class io_node_details_during_extra_ld_st(io_node_details):
  def __init__(self, node_id):
    io_node_details.__init__(self, node_id)

    # list of cycles where this node would be needed
    self.parent_cycles= []
    self.parent_bb_lst= []

  def populate_parent_cycles(self, graph, map_bb_to_cycle, done_bb):
    
    if len(graph[self.node_id].parent_key_list):
      assert graph[self.node_id].parent_buildblk_lst

    parent_buildblk_lst = graph[self.node_id].parent_buildblk_lst
    
    cycle_list= [ [bb,map_bb_to_cycle[bb]] for bb in parent_buildblk_lst if bb not in done_bb]
    
    cycle_list.sort(key= lambda x: x[1])
  
    self.parent_bb_lst= [tup[0] for tup in cycle_list]
    self.parent_cycles= [tup[1] for tup in cycle_list]

def insert_shuffle(BB_graph, instr_ls_obj, bank_d, hw_details, n_mem_banks):
  """
    insert shuffles
  """
  updated_instr_ls= []
  
  for curr_instr in instr_ls_obj.instr_ls:

    if curr_instr.is_type('bb'):
      # Input node conflicts
      bb_obj= BB_graph[curr_instr.bb_id]
      node_to_bank_d= {node: bank_d[node] for node in bb_obj.in_list_unique}
      bank_ls= list(node_to_bank_d.values())

      CONFLICT, bank_to_node_d = conflicting_nodes(node_to_bank_d, bank_ls, hw_details, node_to_pe= None, mode= 'bb_in')

      if CONFLICT:
        # Create a list of sh_instr objects, whose length depends on maximum number of nodes to be shuffled from a bank
        max_node_ls_len= max([len(node_ls) for node_ls in list(bank_to_node_d.values())])
        sh_instr_ls= [common_classes.sh_instr('bb_in', curr_instr.id) for _ in range(0, max_node_ls_len)]
        free_banks= list(set(range(n_mem_banks)) - set(bank_ls))

        for bank, node_ls in list(bank_to_node_d.items()):
          for node_idx, node in enumerate(node_ls):
            final= free_banks.pop()
            intermediate= bank 
            node_to_bank_d[node]= final # This will be used in bb instr
            sh_instr_ls[node_idx].insert_sh_bank(node, intermediate, final)

        for sh_instr in sh_instr_ls:
          updated_instr_ls.append(sh_instr)
        
      # Update banks for inputs in current bb instruction
      assert len(set(node_to_bank_d.keys())) == len(set(node_to_bank_d.values())), "All conflicats are not resolved"
      curr_instr.in_node_details_dict= {}
      for node, bank in list(node_to_bank_d.items()):
        node_obj= io_node_details(node)
        node_obj.bank= bank
        curr_instr.in_node_details_dict[node]= node_obj

      # Output node conflict
      node_to_bank_d= {node: bank_d[node] for node in bb_obj.out_list}
      bank_ls= list(node_to_bank_d.values())
      out_port_map= hw_details.out_port_map

      CONFLICT, bank_to_node_d = conflicting_nodes(node_to_bank_d, bank_ls, hw_details, bb_obj.out_nodes_to_pe, mode= 'bb_out')

      if CONFLICT:
        max_node_ls_len= max([len(node_ls) for node_ls in list(bank_to_node_d.values())])
        sh_instr_ls= [common_classes.sh_instr('bb_out', curr_instr.id) for _ in range(0, max_node_ls_len)]

        conflict_node_set= set([node for node_ls in list(bank_to_node_d.values()) for node in node_ls])
        consumed_banks= set([bank for node, bank in list(node_to_bank_d.items()) if node not in conflict_node_set])

        node_to_final_bank= {}
        node_to_sh_instr_idx= {}
        
        for bank, node_ls in list(bank_to_node_d.items()):
          for node_idx, node in enumerate(node_ls):
            node_to_final_bank[node] = bank
            node_to_sh_instr_idx[node] = node_idx

        assert len(conflict_node_set) == len(node_to_final_bank)

        print("conflict out!!")
        print(conflict_node_set)
        print(consumed_banks)
        print(bb_obj.out_nodes_to_pe)
        print(node_to_final_bank)

        # pick nodes that have the least options for intermediate bank first.
        while conflict_node_set:
        # for node in sorted(conflict_node_set, key= lambda x : bb_obj.out_nodes_to_pe[x][1]):
          node = min(conflict_node_set, key= lambda x : len(out_port_map[bb_obj.out_nodes_to_pe[x]] - consumed_banks))
          conflict_node_set.remove(node)
          final= node_to_final_bank[node]
          #free_banks= out_port_map[bb_obj.out_nodes_to_pe[node]] - set(node_to_bank_d.values())
          free_banks= out_port_map[bb_obj.out_nodes_to_pe[node]] - consumed_banks
          print(node, bb_obj.out_nodes_to_pe[node], out_port_map[bb_obj.out_nodes_to_pe[node]], final, consumed_banks, free_banks)
          # NOTE: possible that free_banks has no bank available. 
          # This is when higher level PEs (level 3) are prefered over lower level PEs (eg. level 0) for output port mapping
          if len(free_banks) > 0:
            intermediate= free_banks.pop()
          else:
            logger.warning("WARNING!!!!! Output hazard! PE cannot store the output. For continuation, choosing a free bank.")
            free_banks = set(range(n_mem_banks)) - consumed_banks
            intermediate = random.choice(list(free_banks))

          consumed_banks.add(intermediate)
          node_to_bank_d[node]= intermediate # This will be used in bb instr
          
          sh_instr_idx = node_to_sh_instr_idx[node] 
          sh_instr_ls[ sh_instr_idx ].insert_sh_bank(node, intermediate, final)
        
      for node, bank in list(node_to_bank_d.items()):
        assert bank in out_port_map[bb_obj.out_nodes_to_pe[node]]
      curr_instr.out_node_details_dict= {}
      for node, bank in list(node_to_bank_d.items()):
        node_obj= io_node_details(node)
        node_obj.bank= bank
        curr_instr.out_node_details_dict[node]= node_obj

      updated_instr_ls.append(curr_instr)

      # Add shift instructions for output shifting, if any
      if CONFLICT:
        for sh_instr in sh_instr_ls:
          updated_instr_ls.append(sh_instr)
            
              
    elif curr_instr.is_type('ld'):
      node_to_bank_d= {node: bank_d[node] for node in curr_instr.node_set}
      bank_ls= list(node_to_bank_d.values())
      
      CONFLICT, bank_to_node_d= conflicting_nodes(node_to_bank_d, bank_ls, hw_details, node_to_pe= None, mode= 'ld')
      
      if CONFLICT:
        # Create a list of sh_instr objects, whose length depends on maximum number of nodes to be shuffled from a bank
        max_node_ls_len= max([len(node_ls) for node_ls in list(bank_to_node_d.values())])
        sh_instr_ls= [common_classes.sh_instr('ld', curr_instr.id) for _ in range(0, max_node_ls_len)]
        free_banks= list(set(range(n_mem_banks)) - set(bank_ls))

        for bank, node_ls in list(bank_to_node_d.items()):
          for node_idx, node in enumerate(node_ls):
            final= bank
            intermediate= free_banks.pop() 
            node_to_bank_d[node]= intermediate # This will be used in bb instr
            sh_instr_ls[node_idx].insert_sh_bank(node, intermediate, final)
        
        
      assert len(set(node_to_bank_d.keys())) == len(set(node_to_bank_d.values())), "All conflicats are not resolved"
      curr_instr.node_details_dict= {}
      for node, bank in list(node_to_bank_d.items()):
        node_obj= io_node_details(node)
        node_obj.bank= bank
        curr_instr.node_details_dict[node]= node_obj
        
      updated_instr_ls.append(curr_instr)
      
      # Add shifting instructions to the instruction list
      if CONFLICT:
        for sh_instr in sh_instr_ls:
          updated_instr_ls.append(sh_instr)

    elif curr_instr.is_type('nop'):
      updated_instr_ls.append(curr_instr)
    else:
      assert 0 
          

  # Remove duplicate shifts
  for idx, instr in enumerate(updated_instr_ls):
    if instr.is_type('sh'):
      for future_instr in updated_instr_ls[idx + 1:]:
        if future_instr.is_type('sh'):
          for node, src_dst_bank in list(instr.sh_dict_bank.items()):
            if node in future_instr.sh_dict_bank:
              future_src= future_instr.sh_dict_bank[node][0]
              future_dst= future_instr.sh_dict_bank[node][1]
              if future_dst == src_dst_bank[1]: # If the destination is same. Src may be different, we don't care
                del future_instr.sh_dict_bank[node]
  
  instr_ls_new= []
  for instr in updated_instr_ls:
    if instr.is_type('sh'):
      if len(instr.sh_dict_bank) != 0:
        instr_ls_new.append(instr)
    else:
      instr_ls_new.append(instr)

  instr_ls_obj.instr_ls= instr_ls_new
  
def conflicting_nodes(node_to_bank_d, bank_ls, hw_details, node_to_pe, mode):
  """
    Finds if there is a bank conflict, and creates useful datastructure to resolve it
  """
  assert mode in ['bb_in', 'bb_out', 'ld']
  if mode == 'bb_in' or mode == 'ld':
    # Conflict
    if len(list(node_to_bank_d.keys())) > len(set(bank_ls)): # more nodes than banks allocated
      bank_conflict_ls= [bank for bank in set(bank_ls) if bank_ls.count(bank)>1]
      
      bank_to_node_d= {bank: [node for node, resp_bank in list(node_to_bank_d.items()) if resp_bank == bank] for bank in bank_conflict_ls}
      
      # Remove one node that needs not be shuffled
      bank_to_node_d= {bank: node_ls[1:] for bank, node_ls in list(bank_to_node_d.items())}
      
      return True, bank_to_node_d

    else:
      return False, None
  elif mode == 'bb_out':
    node_conflict_ls= []
    out_port_map= hw_details.out_port_map

    consumed_banks= set()
    for node in sorted(list(node_to_bank_d.keys()), key= lambda x: node_to_pe[x][1]):
      bank= node_to_bank_d[node]
      pe= node_to_pe[node]
      if bank not in out_port_map[pe] or (bank in consumed_banks):
        node_conflict_ls.append(node)
      else:
        consumed_banks.add(bank)

    bank_to_node_d= {node_to_bank_d[node]: [] for node in node_conflict_ls}
    for node in node_conflict_ls:
      bank_to_node_d[node_to_bank_d[node]].append(node)

    if len(node_conflict_ls) > 0:
      return True, bank_to_node_d
    else:
      return False, None

  else:
    assert 0

def check_consuming_parent(graph, instr_ls_obj, node, curr_cycle, bank, map_node_to_parent_cycle):
  """
    Returns True if there is a parent instr that needs this node in this bank
    Returns False if fully consumed
  """
  for parent_cycle in map_node_to_parent_cycle[node]:
    if parent_cycle > curr_cycle: # parent at a later cycle
      parent_instr= instr_ls_obj.instr_ls[parent_cycle]
      if parent_instr.is_type('bb'):
        if parent_instr.in_node_details_dict[node].bank == bank:  
          return True
      elif parent_instr.is_type('sh'):
        if parent_instr.sh_dict_bank[node][0] == bank:  
          return True
      else:
        assert 0

  return False

def insert_invalidates(graph, BB_graph, instr_ls_obj, RANDOM_BANK_ALLOCATE):
  """
    Inserts 'invalidate' information for bb and shift instructions
    NOTE: BBs cannot be shuffled across each other after invalidates are added
  """
  # map_bb_to_cycle
  # Key: bb id, Val: istr cycle
  map_bb_to_cycle={curr_instr.bb_id: curr_instr_idx for curr_instr_idx, curr_instr in enumerate(instr_ls_obj.instr_ls) if curr_instr.is_type('bb')}
  
  # key: node, val: parent cycle (could be bb, sh or any instr)
  map_node_to_parent_cycle= defaultdict(list)
  for instr_idx, instr in enumerate(instr_ls_obj.instr_ls):
    if instr.is_type('nop'):
      pass
    elif instr.is_type('ld'):
      pass
    elif instr.is_type('sh'):
      for node in list(instr.sh_dict_bank.keys()):
        map_node_to_parent_cycle[node].append(instr_idx)
    elif instr.is_type('bb'):
      for node in list(instr.in_node_details_dict.keys()):
        map_node_to_parent_cycle[node].append(instr_idx)
    else:
      assert 0


  # Map instruction ID to bb_id
  map_instr_id_to_bb_id= {curr_instr.id: curr_instr.bb_id for curr_instr in instr_ls_obj.instr_ls if curr_instr.is_type('bb')}

  for curr_instr_idx, curr_instr in enumerate(instr_ls_obj.instr_ls):
    if curr_instr.is_type('bb'):
      for node, obj in list(curr_instr.in_node_details_dict.items()):
        bank= obj.bank
        if not check_consuming_parent(graph, instr_ls_obj, node, curr_instr_idx, bank, map_node_to_parent_cycle):
          curr_instr.invalidate_node_set.add(node)
    
    if curr_instr.is_type('sh'):
      for node, src_dst_tup in list(curr_instr.sh_dict_bank.items()):
#        if RANDOM_BANK_ALLOCATE and (curr_instr.is_bb_out() or curr_instr.is_ld()): # Always invalidate otherwise too many useless shifts
        if RANDOM_BANK_ALLOCATE or (curr_instr.is_bb_out() or curr_instr.is_ld()): # Always invalidate otherwise too many useless shifts
          curr_instr.invalidate_node_set.add(node)
        else:
          bank= src_dst_tup[0]
          if not check_consuming_parent(graph, instr_ls_obj, node, curr_instr_idx, bank, map_node_to_parent_cycle):
            curr_instr.invalidate_node_set.add(node)

def reg_rd_stage(BB_graph, curr_instr, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set):

    if curr_instr.is_type('nop'):
      pass
    elif curr_instr.is_type('ld') or curr_instr.is_type('ld_sc'):
      pass
    elif curr_instr.is_type('st'):
      for node in curr_instr.node_set:
        bank= curr_instr.node_details_dict[node].bank 
        assert bank in node_details_dict[node]
        assert curr_instr.node_details_dict[node].pos == node_details_dict[node][bank].pos
        deassign(node, bank, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set) 

    elif curr_instr.is_type('sh'):
      for node, src_dst_bank in list(curr_instr.sh_dict_bank.items()):
        src_bank= src_dst_bank[0]
        assert src_bank in node_details_dict[node], [node, node_details_dict[node], src_bank]
        src_pos= node_details_dict[node][src_bank].pos
        # Deassign from src location
        if node in curr_instr.invalidate_node_set:
          deassign(node, src_bank, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set)

        curr_instr.sh_dict_pos[node] = (src_pos, None)

    elif curr_instr.is_type('bb'):
      bb_obj= BB_graph[curr_instr.bb_id]
      for node in set(bb_obj.in_list):
        ## update parent cycle
        for node_obj in list(node_details_dict[node].values()):
          del node_obj.parent_cycles[0]
          del node_obj.parent_bb_lst[0]
        
        bank= curr_instr.in_node_details_dict[node].bank 
        assert bank in node_details_dict[node], [node, list(node_details_dict[node].keys()), bank, curr_instr.bb_id, curr_instr.id]
        curr_instr.in_node_details_dict[node].pos = node_details_dict[node][bank].pos
        assert node in reg_node_set[curr_instr.in_node_details_dict[node].bank]

        # Deassign if fully consumed
        if node in curr_instr.invalidate_node_set:
          deassign(node, bank, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set)
    else:
      assert False

def reg_wr_stage(graph, BB_graph, curr_instr, free_pos, consumed_pos, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb, free_mem_addr):
    if curr_instr.is_type('nop'):
      pass
    elif curr_instr.is_type('ld') or curr_instr.is_type('ld_sc'):
      for node in curr_instr.node_set:
        bank= curr_instr.node_details_dict[node].bank
        assign(graph, node, free_pos, consumed_pos, bank, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb)
        curr_instr.node_details_dict[node].pos= node_details_dict[node][bank].pos
    
      if curr_instr.is_type('ld_sc'):
        free_mem_addr.add(curr_instr.mem_addr)
    elif curr_instr.is_type('st'):
      pass

    elif curr_instr.is_type('sh'):
      for node, src_dst_bank in list(curr_instr.sh_dict_bank.items()):
        # Assign to destination location
        dst_bank= src_dst_bank[1]
#        print(node, src_dst_bank[0], dst_bank)
        assign(graph, node, free_pos, consumed_pos, dst_bank, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb)
        dst_pos= node_details_dict[node][dst_bank].pos
        
        src_pos= curr_instr.sh_dict_pos[node][0]
        curr_instr.sh_dict_pos[node] = (src_pos, dst_pos)

    elif curr_instr.is_type('bb'):
      bb_obj= BB_graph[curr_instr.bb_id]
      # Assign outputs
      for node in bb_obj.out_list:
        bank= curr_instr.out_node_details_dict[node].bank
        assign(graph, node, free_pos, consumed_pos, bank, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb)
        curr_instr.out_node_details_dict[node].pos= node_details_dict[node][bank].pos
        
      done_bb.add(curr_instr.bb_id)
    else:
      assert False

def extra_load_stores(net, global_var, graph, BB_graph, instr_ls_obj, num_of_banks, reg_bank_size, base_scratch_pad_addr, last_scratch_pad_addr, n_pipe_stages, verb= False):
  """
    Load stores for limited bank size
  """
  useful_methods.printlog('Adding load-stores for fixed size banks', 'red')
  
  # node details map.. dict of dict
  # 1st Key: node id, Val: Dictionary
  # 2nd Key: bank, Val: object of io_node_details_during_extra_ld_st
  node_details_dict= defaultdict(dict) 

  # Simulate the memory with 2-D list
  reg_status= [[None for pos in range(reg_bank_size)] for bank in range(num_of_banks)]
  
  # average instr_cycle when a vector reg is consumed
  avg_cycle_cnt= [0] * reg_bank_size

  # outer most index: bank id
  # next: cycle
  bank_occup_profile= [[] for i in range(num_of_banks)]
  
  # consumed set
  consumed_pos= [set() for bank in range(num_of_banks)]
  
  # Free pos
  free_pos= [set(list(range(reg_bank_size))) for bank in range(num_of_banks)]
  
  free_mem_addr= set(list(range(base_scratch_pad_addr, last_scratch_pad_addr)))

  # Bank-wise set of active nodes
  # Key: Bank id, Val: set of active nodes
  reg_node_set= {bank: set() for bank in range(num_of_banks)}

  # map_bb_to_cycle
  # Key: bb id, Val: istr cycle
  map_bb_to_cycle={curr_instr.bb_id: curr_instr_idx for curr_instr_idx, curr_instr in enumerate(instr_ls_obj.instr_ls) if curr_instr.is_type('bb')}
  
  # map bb to instr_id
  # Key= bb id, Val: instr_id
  map_bb_to_instr_id= {curr_instr.bb_id: curr_instr.id for curr_instr in instr_ls_obj.instr_ls if curr_instr.is_type('bb')}

  done_bb= set()
  
  instr_in_pipe= [common_classes.nop_instr()] * (n_pipe_stages - 1)
  # instr_in_pipe= [common_classes.nop_instr()] * (n_pipe_stages)

  # Collecting statistics
  # Key: pos_diff, Val: frequency of bb
  pos_diff_dict= defaultdict(int)

  curr_instr_idx= 0

  original_instr_ls_len= len(instr_ls_obj.instr_ls)
  
  start_time= time.time()
  while curr_instr_idx < len(instr_ls_obj.instr_ls):
    if curr_instr_idx % 100 == 0:
      logger.info(f"Done {curr_instr_idx} out of total {len(instr_ls_obj.instr_ls)} instructions in {time.time() - start_time}s")
    curr_instr= instr_ls_obj.instr_ls[curr_instr_idx]
    #print curr_instr_idx
    if curr_instr.is_type('nop'):
      if verb:
        print(curr_instr.id)
        printlog('nop')
    elif curr_instr.is_type('ld') or curr_instr.is_type('ld_sc'):
      if verb:
        print(curr_instr.id)
        printlog(['ld or ld_sc', curr_instr.node_set])
    elif curr_instr.is_type('st'):
      if verb:
        print(curr_instr.id)
        printlog(['st', curr_instr.node_set])
    elif curr_instr.is_type('bb'):
      if verb:
        print(curr_instr.id)
        printlog(['bb', list(curr_instr.in_node_details_dict.keys()), list(curr_instr.out_node_details_dict.keys()), curr_instr.invalidate_node_set, curr_instr.bb_id])
    elif curr_instr.is_type('sh'):
      if verb:
        print(curr_instr.id)
        printlog(['shuffle', curr_instr.corresponding_instr_id ,curr_instr.sh_dict_bank, curr_instr.invalidate_node_set])
    else:
      assert 0
    
    instr_in_pipe.append(curr_instr)
    commit_instr= instr_in_pipe.pop(0)
    reg_wr_stage(graph, BB_graph, commit_instr, free_pos, consumed_pos, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb, free_mem_addr)

    #-- Moving reg_rd_stage AFTER reg_wr_stage to handle SRAM based RF's rd-wr contention. 
    #-- i.e. Same address should not be written and read in the same cycle
    reg_rd_stage(BB_graph, curr_instr, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set)

    # TODO FIXME : A better way of intermediate load-store is possible in which
    # the stored vector is loaded when it is absolutely needed, 
    # depending on whether the stored word is over-written or not.
    #Currently, it is always assumed to be overwritten. 

    # Add a store-load if bank full
    add_store_load_if_needed(net, curr_instr_idx, consumed_pos, free_pos, instr_ls_obj, node_details_dict, reg_status, reg_bank_size, map_bb_to_cycle, map_bb_to_instr_id, free_mem_addr, n_pipe_stages)

    if original_instr_ls_len < 5000:
      sanity_check_extra_load_store_0(consumed_pos, free_pos, reg_bank_size)
    curr_instr_idx += 1

    if curr_instr_idx > 20*original_instr_ls_len:
      logger.info(f"Timeout. Too many intermediate loads and stores. Probably stuck in a forever loop due to small register file. Original instrucions count: {original_instr_ls_len}, curr_instr_idx: {curr_instr_idx}")
      assert 0

    for b in range(num_of_banks):
      bank_occup_profile[b].append( len(consumed_pos[b]))

  # Flush pipe
  for commit_instr in instr_in_pipe:
    reg_wr_stage(graph, BB_graph, commit_instr, free_pos, consumed_pos, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb, free_mem_addr)
  
  sanity_check_extra_load_store_1(consumed_pos, free_pos, reg_bank_size, num_of_banks)
  assert len(done_bb) == len(BB_graph), [len(done_bb), len(BB_graph)]
  printlog('Total instruction w/o considering conflicts: ' + str(len(instr_ls_obj.instr_ls)) + ', Reg bank size: ' + str(reg_bank_size), 'blue')
  printlog(f'Total intermediate load stores: {len(instr_ls_obj.instr_ls) -  original_instr_ls_len}', 'blue')

  # Bank occupancy profie:
  # global_var.BANK_OCCUP_PROFILE_FILE += f'_{reg_bank_size}_{num_of_banks}.csv'
  # open (global_var.BANK_OCCUP_PROFILE_FILE,'w+')
  # for bank, profile in enumerate(bank_occup_profile):
  #   reporting_tools._write_csv(global_var.BANK_OCCUP_PROFILE_FILE, profile, mode= 'a')
    

def sanity_check_extra_load_store_0(consumed_pos, free_pos, reg_bank_size):
  for bank, pos_set in enumerate(consumed_pos):
    assert len(pos_set) + len(free_pos[bank]) == reg_bank_size, [len(pos_set), len(free_pos[bank]), reg_bank_size]

def sanity_check_extra_load_store_1(consumed_pos, free_pos, reg_bank_size, num_of_banks):
  total_free_pos= 0
  for bank, pos_set in enumerate(consumed_pos):
    assert len(pos_set) < 2 
    total_free_pos += len(free_pos[bank])

  assert total_free_pos == reg_bank_size * num_of_banks - 1, [total_free_pos, reg_bank_size]

def add_store_load_if_needed(net, curr_instr_idx, consumed_pos, free_pos, instr_ls_obj, node_details_dict, reg_status, reg_bank_size, map_bb_to_cycle, map_bb_to_instr_id, free_mem_addr, n_pipe_stages):
  
  FULL_WATERMARK= max(5, min(n_pipe_stages, 0.25*reg_bank_size))
  # check_occupancy
  FULL= False
  for bank, free_pos_set in enumerate(free_pos):
    if len(free_pos_set) < FULL_WATERMARK:
      FULL= True
      break
  
  
  # Create list of nodes to be stored
  if FULL:
    FULL_THRESHOLD_d= {8:5, 16:11, 32: 20, 64: 54, 128:110, 256: 235}
    FULL_THRESHOLD= FULL_THRESHOLD_d[reg_bank_size]
    # FULL_THRESHOLD= 0.8*reg_bank_size
    # Find the node to store from each bank
    store_nodes_set= set()
    map_nodes_to_bank= {}
    map_nodes_to_parent_cycle= {}

    for bank,pos_set in enumerate(consumed_pos):
      if len(pos_set) > FULL_THRESHOLD or (reg_bank_size-len(pos_set))< FULL_WATERMARK: 
        # logger.info(f"bank: {bank, len(pos_set), FULL_THRESHOLD, FULL_WATERMARK}")
        node_lst= [reg_status[bank][pos] for pos in pos_set]

        for node in node_lst:
          assert len(node_details_dict[node][bank].parent_bb_lst) > 0, [node, bank, list(node_details_dict[node].keys())]

        # Sort according to parent consumption cycle
        # map_nodes_to_parent_cycle_temp= map_node_to_first_parent_cycle(curr_instr_idx, node_details_dict, node_lst, bank ,instr_ls_obj.instr_ls)
        # node= max(node_lst, key= lambda x: map_nodes_to_parent_cycle_temp[x])

        node, map_nodes_to_parent_cycle_temp= map_node_to_first_parent_cycle(curr_instr_idx, node_details_dict, node_lst, bank ,instr_ls_obj.instr_ls)

        if node_lst:
          if map_nodes_to_parent_cycle_temp[node] > curr_instr_idx + n_pipe_stages + 1: # only store if parent is sufficiently far from current instruction
            store_nodes_set.add(node)
            map_nodes_to_bank[node]= bank
            map_nodes_to_parent_cycle[node]= map_nodes_to_parent_cycle_temp[node]
            # logger.info(f"ADDED: {node, curr_instr_idx, map_nodes_to_parent_cycle_temp[node]}")
          # else:
            # logger.info(f"NOT ADDED:{ node, curr_instr_idx, map_nodes_to_parent_cycle_temp[node]}")

    
    # assert len(store_nodes_set) != 0
    # some banks are about to be full but their parents are very close to current instruction so they will get freed up eventually, and no need to spill data to memory
    if len(store_nodes_set) == 0: 
      return
    
    # Creat store and load instruction objects
    st_instr= common_classes.st_instr()
    st_instr.node_set= store_nodes_set    
    st_cycle= curr_instr_idx + 1
    
    ld_instr= common_classes.ld_sc_instr()
    ld_instr.node_set= store_nodes_set    

    st_instr.node_details_dict= {}
    ld_instr.node_details_dict= {}
    for node in store_nodes_set:
      ld_instr.node_details_dict[node]=  io_node_details(node)
      st_instr.node_details_dict[node]=  io_node_details(node)
      ld_instr.node_details_dict[node].bank= map_nodes_to_bank[node]
      st_instr.node_details_dict[node].bank= map_nodes_to_bank[node]
      st_instr.node_details_dict[node].pos=  node_details_dict[node][map_nodes_to_bank[node]].pos

    assert len(free_mem_addr) != 0, f"Scratch pad size not enough {net}, reg_bank_size: {reg_bank_size}, tree_depth: {n_pipe_stages - 2}, n_banks: {len(free_pos)}"
    
    mem_addr= min(free_mem_addr)
    st_instr.mem_addr= mem_addr
    ld_instr.mem_addr= mem_addr
    free_mem_addr.remove(mem_addr)

    earliest_parent_node= min(store_nodes_set, key= lambda x: map_nodes_to_parent_cycle[x])  
    earliest_parent_cycle= map_nodes_to_parent_cycle[earliest_parent_node]
    ld_cycle= earliest_parent_cycle + 1 - n_pipe_stages
    
    assert ld_cycle > st_cycle + 1, [ld_cycle, st_cycle, store_nodes_set, earliest_parent_node, earliest_parent_cycle, curr_instr_idx, n_pipe_stages]
    
    assert ld_cycle != st_cycle, [ld_cycle, st_cycle]

    # logger.info(f"Intermediate store_cycle: {st_cycle}, load cycle: {ld_cycle}")
    if ld_cycle != st_cycle + 1:
      instr_ls_obj.instr_ls.insert(st_cycle, st_instr)
      instr_ls_obj.instr_ls.insert(ld_cycle, ld_instr)
    else:
      printcol("WARNING: Not inserting intermediate loads and stores even if regbank is about to be fulll, because load has to be just after the store")
      assert 0
    
def map_node_to_first_parent_cycle(curr_instr_idx, node_details_dict, node_lst, bank, instr_ls):
  # Key: node, val: first dependent parent cycle (> curr_instr_idx)
  map_parent_cycle= {}

  node = max(node_lst, key= lambda x : min(node_details_dict[x][bank].parent_cycles))
  original_parent_cycle = min(node_details_dict[node][bank].parent_cycles)

  # the new parent cycle will definitely be after the max(curr_instr_idx, original_parent_cycle) cycle
  start_cycle = max(curr_instr_idx + 1, original_parent_cycle)
  for instr_idx, instr in enumerate(instr_ls[start_cycle : ]):
    reg_input_dict= instr.reg_input()

    if node in reg_input_dict:
      if reg_input_dict[node] == bank:
        # map_parent_cycle only contains the cycle for this selected node
        map_parent_cycle[node]= instr_idx + start_cycle
        break

  # It may happen that the previous loop did not find any parent_cycle
  # because the last instruction consuming node, consumes it from a different "bank".
  # We fall back to original code to iterate over the entire instr_ls starting from curr_instr_idx
  if len(map_parent_cycle) == 0:
    node_lst_copy= set(node_lst)
    for instr_idx, instr in enumerate(instr_ls[curr_instr_idx + 1:]):
      reg_input_dict= instr.reg_input()

      curr_nodes= set(reg_input_dict.keys()) & node_lst_copy
      for node in curr_nodes:
        if reg_input_dict[node] == bank:
          map_parent_cycle[node]= instr_idx + curr_instr_idx + 1
          node_lst_copy.remove(node)
    
      if len(node_lst_copy) == 0:
        break

    assert len(set(node_lst)) == len(list(map_parent_cycle.keys()))

  assert len(map_parent_cycle) != 0
  # return map_parent_cycle
  return node, map_parent_cycle

def deassign(node, bank, free_pos, consumed_pos, node_details_dict, reg_status, reg_node_set):
  node_obj= node_details_dict[node][bank]

  pos= node_obj.pos
  
  free_pos[bank].add(pos)
  consumed_pos[bank].remove(pos)
  
  reg_status[bank][pos]= None

  assert node in reg_node_set[bank]

  reg_node_set[bank].remove(node)

  del node_details_dict[node][bank]
  assert bank not in list(node_details_dict[node].keys())

def assign(graph, node, free_pos, consumed_pos, bank, map_bb_to_cycle, node_details_dict, reg_status, reg_node_set, done_bb):
  """
    Free position should always be available
  """
  
  # NOTE: This assertion can fire due to improper shift to resolve ld or bb_out conflicts
  assert node not in reg_node_set[bank] 

  node_obj= io_node_details_during_extra_ld_st(node)
  
  node_obj.populate_parent_cycles(graph, map_bb_to_cycle, done_bb)
  
  if node_obj.parent_cycles:
    curr_node_cycle_cnt= node_obj.parent_cycles[0]
  else:
    curr_node_cycle_cnt= 0 # For head node
  
  assert len(free_pos[bank]) != 0, free_pos[bank]
  
  node_obj.bank= bank
  # pos= sorted(list(free_pos[bank]))[0]
  pos= min(list(free_pos[bank]))
  node_obj.pos= pos
  free_pos[bank].remove(pos)
  consumed_pos[bank].add(pos)
  
  node_details_dict[node][bank]= node_obj
  reg_status[bank][pos]= node
  
  reg_node_set[bank].add(node)

  return bank, pos

def add_initial_loads(net, graph, BB_graph, instr_ls_obj, n_mem_banks, leaf_list, base_addr, last_addr):
  """
    Group together leaf inputs and add vector loads in instruction sequence
  """
  LOAD_LEN= n_mem_banks - min(4, 0.25*n_mem_banks)
  useful_methods.printlog("Adding initial loads", 'red')
  
  # List of [first_bb_id, last_bb_id, first_bb_pos, set] groups
  # Each sets consists of group of leaves
  load_list_details=[]
    
  curr_set= set()
  dependent_bb_set= set()
  done_leaf= set()
  for instr_idx, instr in enumerate(instr_ls_obj.instr_ls):
    if instr.is_type('bb'):
      bb= instr.bb_id
      for in_node in BB_graph[bb].in_list_unique:
        if in_node not in done_leaf:
          if graph[in_node].is_leaf():
            if len(curr_set)==0:
              first_bb_id= bb
              first_bb_pos= instr_idx
    
            curr_set.add(in_node)
            dependent_bb_set.add(bb)
            done_leaf.add(in_node) 

          if len(curr_set) == LOAD_LEN:
            last_bb_id= bb
            load_list_details.append([first_bb_id, last_bb_id, first_bb_pos, dependent_bb_set, curr_set])
            curr_set= set()
            dependent_bb_set= set()
  
  if len(curr_set) != 0:
    last_bb_id= bb
    load_list_details.append([first_bb_id, last_bb_id, first_bb_pos, dependent_bb_set, curr_set])
    

  assert len(done_leaf) == len(leaf_list) > (len(load_list_details)-1)*LOAD_LEN, [len(done_leaf), len(leaf_list)]
  
  ld_mem_counter= itertools.count(base_addr)

  # Map parameter nodes to their location in param memory
  # Key: node id, Val: [addr, bank] where addr is n_mem_banks aligned
  map_param_to_addr= {}

  # create ld_instr object and insert in instr_ls
  for load_idx, load_details in enumerate(load_list_details):
    instr= common_classes.ld_instr()
    
    instr.dependent_bb_set= load_details[3]
    instr.node_set= load_details[4]
    instr.mem_addr= next(ld_mem_counter)
    
    assert instr.mem_addr <= last_addr, f"{net}"
    assert instr.mem_addr >= base_addr

    first_bb_pos= load_details[2]
    
    instr_ls_obj.instr_ls.insert(first_bb_pos + load_idx, instr)

    for node in instr.node_set:
      map_param_to_addr[node]= [instr.mem_addr, None] # Only addr is updated here, bank will be added after bank_allocate

  # Sanity Check 
  instr_ls_copy= list(instr_ls_obj.instr_ls)
  for instr_idx, instr in enumerate(instr_ls_obj.instr_ls):
    if instr.is_type('ld'):
      bb_set= instr.dependent_bb_set
      bb_instr_lst= [depend_instr.bb_id for depend_instr in instr_ls_copy[instr_idx:] if depend_instr.is_type('bb')]
      for bb in bb_set:
        assert bb in bb_instr_lst, [bb, bb_instr_lst]
   
#  printcol("Number of initial loads: " + str(len(load_list_details)), 'blue')
  
  return len(load_list_details), map_param_to_addr

def BB_scheduling_gather(graph, BB_graph, n_pipe_stages, instr_ls_obj):
  useful_methods.printlog("Scheduling building blocks (GATHER)", 'red')
  
  # Dictionary to map sch_level to BB
  # Key: sch_level
  # Val: BB number
  map_sch_lvl_to_bb= {}
  
  # Scheduling based on a DFS, without worrying about pipelining hazards
  head_bb= scheduling.create_bb_dependecny_graph(graph, BB_graph) 
  scheduling.compute_reverse_lvl(BB_graph, head_bb)
  
  scheduling.assign_sch_lvl(BB_graph, head_bb, common_classes.Mutable_level(), map_sch_lvl_to_bb)
  
  # NOTE: Following assignment style gives the best NOP count possible
  #assign_sch_reverse_lvl_wise(BB_graph, head_bb, map_sch_lvl_to_bb)

  # Reorder to avoid hazards due to pipelining
  sorted_sch_level= sorted(list(map_sch_lvl_to_bb.keys()))
  bb_seq = [map_sch_lvl_to_bb[sch_level] for sch_level in sorted_sch_level]
  
  bb_seq= pipeline.pipeline_reorder(BB_graph, bb_seq, n_pipe_stages)
  print(f"Total NOP cycles:{len(bb_seq)- len(BB_graph)}") 
  bb_seq= pipeline.pipeline_reorder_reverse(BB_graph, bb_seq, n_pipe_stages)
  print(f"Total NOP cycles:{len(bb_seq)- len(BB_graph)}") 
  check_for_opportunity(BB_graph, bb_seq, n_pipe_stages)
  bb_seq= pipeline.pipeline_reorder(BB_graph, bb_seq, n_pipe_stages)
  print(f"Total NOP cycles:{len(bb_seq)- len(BB_graph)}") 
  
  update_sch_level(BB_graph, bb_seq)

  count_hazards(BB_graph, bb_seq, n_pipe_stages)
  print(f"Total cycles required for hazardfree operation: {len(bb_seq)}")
  print(f"Total NOP cycles:{len(bb_seq)- len(BB_graph)}") 
  
  check_for_opportunity(BB_graph, bb_seq, n_pipe_stages)
  # Sanity check
  # all the schedule levels should be unique
  for bb, obj in BB_graph.items():
    for child in obj.child_bb_lst:
      assert BB_graph[child].sch_level < obj.sch_level
  sch_lvl_lst = [obj.sch_level for obj in BB_graph.values()]
  assert len(sch_lvl_lst) == len(set(sch_lvl_lst)), "Repeated sch_level not allowed. Problem in assign_sch_lvl() method"
  assert None not in sch_lvl_lst, "sch_level should never be None"
  
  # reset computed bit
  for node, obj in list(graph.items()):
    obj.computed= False
  
  
  # Populate instr_ls
  for bb in bb_seq:
    if bb == 'NOP':
      instr= common_classes.nop_instr()
#      instr= common_classes.instr('nop')
    else:
      instr= common_classes.bb_instr()
      instr.bb_id= bb
      
    instr_ls_obj.instr_ls.append(instr)

  return bb_seq

def check_for_opportunity(BB_graph, bb_seq, n_pipe_stages):
  """
    Check if there is any scope for improvement
  """

  count= 0
  for bb, obj in list(BB_graph.items()):
    PARENT_FOUND= False
    first_parent= None
    first_parent_level= -1
    for parent in obj.parent_bb_lst:
      if BB_graph[parent].sch_level== obj.sch_level + n_pipe_stages:
        PARENT_FOUND= True
      
      if first_parent_level > BB_graph[parent].sch_level:
        first_parent_level = BB_graph[parent].sch_level
        first_parent= parent

    if not PARENT_FOUND:
      for idx in range(obj.sch_level, first_parent_level):
        if bb_seq[idx] == 'NOP':
          count += 1
          print(bb, obj.parent_bb_lst, obj.child_bb_lst, end=' ')
  
  print("Total count:", count)

def update_sch_level(BB_graph, bb_seq):
  """
    Update object propery for sch_level
  """

  for level, bb in enumerate(bb_seq):
    if bb != "NOP":
      BB_graph[bb].sch_level= level

  # Sanity check
  for bb,obj in list(BB_graph.items()):
    for parent in obj.parent_bb_lst:
      assert BB_graph[parent].sch_level > obj.sch_level
      

def count_hazards(BB_graph, bb_seq, n_pipe_stages):
  """
    Count the number of hazards based on n_pipe_stages
  """

  # remove "NOP" from bb_seq
  bb_seq= [bb for bb in bb_seq if bb != "NOP"]

  bb_in_pipe= ['NOP'] * n_pipe_stages 
  
  hazard_count = 0

  while bb_seq: 
    chosen_bb= 'NOP'
    bb_not_yet_commited= set(bb_in_pipe)

    for bb in bb_seq:
      obj= BB_graph[bb]
      
      bb_schedulable= True

      for child in obj.child_bb_lst:
        if child in bb_not_yet_commited:
          bb_schedulable= False
          bb_not_yet_commited.add(bb)
          break

      if bb_schedulable:
        chosen_bb= bb
        break
    
    if chosen_bb != 'NOP':
      bb_seq.remove(chosen_bb)
    else:
      hazard_count += 1

    pipeline.update_pipe_state(bb_in_pipe, chosen_bb)
    
    assert len(bb_in_pipe) == n_pipe_stages
 
  print("#Hazards with n_pipe_stages = ", n_pipe_stages, " is ", hazard_count)



def set_attr_sch_level(graph, IO_graph, BB_graph):
  """
    For each node in IO graph, it assigns a list as attribute. 
      Attr1: The list contains sch_level of the consuming BBs 
      Attr2: The list contains list of all the parent BBs
  """
  
  attr1= 'par_sch_levels'
  attr2= 'comp_sch_level' 
  for node in IO_graph:
    obj= graph[node]
    par_sch_levels = [BB_graph[BB].sch_level for BB in obj.parent_buildblk_lst] 

    if len(obj.compute_buildblk_lst) == 1:
      assert len(obj.compute_buildblk_lst) == 1
      comp_sch_level = obj.compute_buildblk_lst[0]
    else:
      comp_sch_level= None
    
    attr_dict= {node: {attr1: par_sch_levels, attr2: comp_sch_level}}

    nx.set_node_attributes(IO_graph, attr_dict)

def liveness_analysis(graph, BB_graph, IO_graph, bb_seq, bank_d, total_banks, global_var):
  """
    Checks how many variables are 'live' in each bank, for every cycle 
  """

  # A dict to keep track if which IOs are live in a bank
  # Key: bank id
  # Val: a set of live IO nodes
  banks_liveset_d= {}

  # A dict to keep track of maximum liveset length for every bank
  # Key: bank ID
  # Val: max live set length
  max_liveset_len={}
  
  # bank occupancy profile
  # key: bank
  # Van: profile (a list)
  bank_occup_profile= {}
  
  # Initialize all the banks with empty live-set
  for bank in range(total_banks):
    banks_liveset_d[bank]= set()
    max_liveset_len[bank]= 0
    bank_occup_profile[bank]= []

  # construct par_sch_levels for every IO node
  # It will be a dict with key as node_id and val as a list of par_sch_levels
  par_sch_levels_d= {}
  for io_node in IO_graph:
    par_sch_levels = [BB_graph[BB].sch_level for BB in graph[io_node].parent_buildblk_lst] 
    par_sch_levels_d[io_node]= par_sch_levels
  

  # Liveness analysis
  for instr_idx, curr_bb in enumerate(bb_seq):
    if curr_bb == 'NOP':
      continue
    
    inputs= BB_graph[curr_bb].in_list_unique
    outputs= BB_graph[curr_bb].out_list

    # Allocate inputs to banks
    # This is for leaf-node inputs
    for i in inputs:
      bank= bank_d[i]
      banks_liveset_d[bank].add(i)

    # Update max liveset len
    for bank in list(bank_occup_profile.keys()):
      bank_occup_profile[bank].append(len(banks_liveset_d[bank]))

    for bank,max_len in list(max_liveset_len.items()):
      if len(banks_liveset_d[bank]) > max_len:
        max_liveset_len[bank]= len(banks_liveset_d[bank])
    
    # Deallocate the inputs that are fully consumed
    for i in inputs:
      par_sch_levels = par_sch_levels_d[i]
      if instr_idx == max(par_sch_levels):
        bank= bank_d[i]
        banks_liveset_d[bank].remove(i)
      else: # Just a check
        assert max(par_sch_levels) > instr_idx
    
    # Allocate outpus 
    for o in outputs:
      bank= bank_d[o]
      banks_liveset_d[bank].add(o)


  # Sanity check
  last_bb= BB_graph[bb_seq[-1]].out_list[0]
  for liveset in list(banks_liveset_d.values()):
    if last_bb in liveset:
      assert len(liveset) == 1, "liveset with head_bb should be exactly of len 1"
    else:
      assert len(liveset) == 0, "All live set should be empty by now"

  print("Bank-depth required:", max(max_liveset_len.values()), max_liveset_len)
  
  global_var.BANK_OCCUP_PROFILE_FILE += f'_{total_banks}.csv'
  open (global_var.BANK_OCCUP_PROFILE_FILE,'w+')
  for bank, profile in list(bank_occup_profile.items()):
    reporting_tools._write_csv(global_var.BANK_OCCUP_PROFILE_FILE, profile, mode= 'a')
    
