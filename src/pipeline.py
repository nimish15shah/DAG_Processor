
from collections import defaultdict

from . import common_classes
from .useful_methods import printcol, printlog

class pipe_graph_node():
  def __init__(self, instr_id, n_banks):
    self.id= instr_id
    
    self.reg_output_nodes= [set() for bank in range(n_banks)] 
    self.reg_input_nodes= [set() for bank in range(n_banks)]

    self.mem_output_nodes= [set() for bank in range(n_banks)]
    self.mem_input_nodes= [set() for bank in range(n_banks)]


def dependency_graph(instr_ls, n_banks):
  """
    Creates dependecies between instructions using pipe_graph_node class object
  """

  dependency_ls= []
  
  # key: instr id, val: object of type pipe_graph_node
  map_instr_to_pipe_node= {}

  for instr in instr_ls:
    obj= pipe_graph_node(instr.id, n_banks)

    if instr.is_type('nop'):
      pass

    elif instr.is_type('ld') or instr.is_type('ld_sc'):
      for node in instr.node_set:
        bank= instr.node_details_dict[node].bank
        obj.reg_output_nodes[bank].add(node)
        obj.mem_input_nodes[bank].add(node)
    
    elif instr.is_type('st'):
      for node in instr.node_set:
        bank= instr.node_details_dict[node].bank
        obj.reg_input_nodes[bank].add(node)
        obj.mem_output_nodes[bank].add(node)

    elif instr.is_type('sh'):
      for node, src_dst_bank in list(instr.sh_dict_bank.items()):
        obj.reg_input_nodes[src_dst_bank[0]].add(node)
        obj.reg_output_nodes[src_dst_bank[1]].add(node)

    elif instr.is_type('bb'):
      for node, node_details in list(instr.in_node_details_dict.items()):
        obj.reg_input_nodes[node_details.bank].add(node)
      for node, node_details in list(instr.out_node_details_dict.items()):
        obj.reg_output_nodes[node_details.bank].add(node)

    else:
      assert False

    dependency_ls.append(obj)
    map_instr_to_pipe_node[obj.id]= obj
  
  return map_instr_to_pipe_node


def reschedule_for_hazardfree(instr_ls, n_pipe_stages, n_banks, WINDOW= 20):
  
  # Original list should atleast be hazard free for 1 pipeline stage
  # sanity_check_hazard(instr_ls, n_banks, n_pipe_stages= 1)

  map_instr_to_pipe_node= dependency_graph(instr_ls, n_banks)
  

  # NOTE: Disabling reverse reschedule because it may lead to very high register consumption. 
  # The BB instruction in instr_ls is supposed to be scheduled such that it is as close to source load instruction, so that the register do not remain occupied for too long
  # Reverse reschedule may disturb this
  # updated_instr_ls= reschedule(instr_ls, map_instr_to_pipe_node, n_pipe_stages, n_banks, WINDOW, reverse= False, insert_nop= False)
  # updated_instr_ls= reschedule(updated_instr_ls, map_instr_to_pipe_node, n_pipe_stages, n_banks, WINDOW, reverse= True, insert_nop= True)

  updated_instr_ls= reschedule(instr_ls, map_instr_to_pipe_node, n_pipe_stages, n_banks, WINDOW, reverse= False, insert_nop= True)

  sanity_check_hazard(updated_instr_ls, n_banks, n_pipe_stages)
  
  return updated_instr_ls
  
def reschedule(instr_ls, map_instr_to_pipe_node, n_pipe_stages, n_banks, WINDOW,reverse= False, insert_nop= True):    
  
  # A list that models a FIFO to keep track of BBs in execution
  # A queue is not used as it doesn't allow iteration or checking of elements
  # Initialized with dummy pipe_graph_node objects
  instr_in_pipe= [pipe_graph_node(None, n_banks)] * n_pipe_stages
  
  instr_ls_w_hazards= list(instr_ls)
  if reverse:
    instr_ls_w_hazards.reverse()

  instr_hazardfree= []
  
  while instr_ls_w_hazards: # Until instr are left to be scheduled
    
    chosen_instr= None 
    
    active_set= set(instr_in_pipe)

    for idx, instr in enumerate(instr_ls_w_hazards): 
      if not dependency_check_reg(active_set, map_instr_to_pipe_node[instr.id], reverse):
        chosen_instr = instr
        chosen_idx= idx
        break
      else:
        active_set.add(map_instr_to_pipe_node[instr.id])
      
      if WINDOW*n_pipe_stages < idx:
        break

    if chosen_instr == None:
      chosen_instr= common_classes.nop_instr()
      pipe_node= pipe_graph_node(chosen_instr.id, n_banks)
      if insert_nop:
        map_instr_to_pipe_node[chosen_instr.id]= pipe_node
        instr_hazardfree.append(chosen_instr)
    else:
      del instr_ls_w_hazards[idx]
      pipe_node= map_instr_to_pipe_node[chosen_instr.id]
      instr_hazardfree.append(chosen_instr)

    update_pipe_state(instr_in_pipe, pipe_node) 

    assert len(instr_in_pipe) == n_pipe_stages

  assert len(instr_hazardfree) >= len(instr_ls), [len(instr_hazardfree), len(instr_ls)]
  
  if reverse:
    instr_hazardfree.reverse()

  return list(instr_hazardfree)


def dependency_check_reg(source_set, target, reverse):
  """
    Checks if target is dependent on source
  """
  if not reverse:
    for source in source_set:
      for bank, node_set in enumerate(source.reg_output_nodes):
        if len(node_set & target.reg_input_nodes[bank]) != 0:
          #print bank, target, source, node_set, target.reg_input_nodes, source.reg_output_nodes
          return True
  else:
    for source in source_set:
      for bank, node_set in enumerate(source.reg_input_nodes):
        if len(node_set & target.reg_output_nodes[bank]) != 0:
          return True

  return False

def dependent_check_mem(source_set, target):
  """
    Checks if target is dependent on source
  """
  if not reverse:
    for source in source_set:
      for bank, node_set in enumerate(source.mem_output_nodes):
        if len(node_set & target.mem_input_nodes[bank]) != 0:
          return True
  else:
    for source in source_set:
      for bank, node_set in enumerate(source.mem_input_nodes):
        if len(node_set & target.mem_output_nodes[bank]) != 0:
          return True
    
  return False

def sanity_check_hazard(instr_ls, n_banks, n_pipe_stages):
  map_instr_to_pipe_node= dependency_graph(instr_ls, n_banks)
  
  instr_in_pipe= [pipe_graph_node(None, n_banks)] * n_pipe_stages

  for instr in instr_ls:
    pipe_node= map_instr_to_pipe_node[instr.id]
    assert not dependency_check_reg(instr_in_pipe, pipe_node, reverse= False), [instr.id, [instr_obj.id for instr_obj in instr_in_pipe]]
    update_pipe_state(instr_in_pipe, pipe_node)

def pipeline_reorder_reverse(BB_graph, bb_seq, n_pipe_stages):
  """
    Reordering same as pipeline_reorder, but in reverse direction  
  """
  # remove "NOP" from bb_seq
  bb_seq= [bb for bb in bb_seq if bb != "NOP"]
  
  # This list can have "NOP" apart from BB keys
  bb_seq_hazardfree= []

  # A list that models a FIFO to keep track of BBs in execution
  # A queue is not used as it doesn't allow iteration or checking of elements
  bb_in_pipe= ['NOP'] * n_pipe_stages 
  
  while bb_seq: # Until all BBs are sequenced hazard-free
    
    chosen_bb= 'NOP'
    
    dependent_bb_lst= set(bb_in_pipe)

    for bb in reversed(bb_seq):
      obj= BB_graph[bb]
      
      bb_schedulable= True

      for parent in obj.parent_bb_lst:
        if parent in dependent_bb_lst:
          bb_schedulable= False
          dependent_bb_lst.add(bb)
          break

      if bb_schedulable:
        chosen_bb= bb
        break
    
    if chosen_bb != 'NOP':
      bb_seq.remove(chosen_bb)
    
    bb_seq_hazardfree.append(chosen_bb)
    update_pipe_state(bb_in_pipe, chosen_bb)
    
    assert len(bb_in_pipe) == n_pipe_stages
  
  bb_seq_hazardfree= list(reversed(bb_seq_hazardfree))

  assert len(bb_seq_hazardfree) >= len(BB_graph)
  
  return bb_seq_hazardfree

def pipeline_reorder(BB_graph, bb_seq, n_pipe_stages):
  """
    Reorder to avoid hazards due to pipelining
  """
  
  assert isinstance(BB_graph, dict)
  assert isinstance(bb_seq, list)
  
  # remove "NOP" from bb_seq
  bb_seq= [bb for bb in bb_seq if bb != "NOP"]
  
  # This list can have "NOP" apart from BB keys
  bb_seq_hazardfree= []

  # A list that models a FIFO to keep track of BBs in execution
  # A queue is not used as it doesn't allow iteration or checking of elements
  bb_in_pipe= ['NOP'] * n_pipe_stages 

  while bb_seq: # Until all BBs are sequenced hazard-free
    
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
    
    bb_seq_hazardfree.append(chosen_bb)
    update_pipe_state(bb_in_pipe, chosen_bb)
    
    assert len(bb_in_pipe) == n_pipe_stages
  
  assert len(bb_seq_hazardfree) >= len(BB_graph)
  
  return bb_seq_hazardfree


def update_pipe_state(bb_in_pipe, new_op):
  """
    update bb_in_pipe queue by appending new_op at the end and popping the last op
  """
  bb_in_pipe.append(new_op)
  bb_in_pipe.pop(0)
