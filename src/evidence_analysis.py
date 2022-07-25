import random
import subprocess
from collections import defaultdict
import copy
import queue
import time

from . import common_classes
from . import ac_eval


def populate_BN_list(graph, BN, curr_node, done_nodes):
  #graph = analysis_obj.graph
  #BN= analysis_obj.BN

  if done_nodes.get(curr_node, False) == True:    
    return graph[curr_node].BN_node_list

  # This is a leaf node
  if graph[curr_node].operation_type == common_classes.OPERATOR.LEAF:
    done_nodes[curr_node]= True
    # If leaf is weight return empty list
    if not graph[curr_node].leaf_type == graph[curr_node].LEAF_TYPE_WEIGHT:
      graph[curr_node].BN_node_list= [graph[curr_node].leaf_BN_node_name + '$' + graph[curr_node].leaf_BN_node_state]
    else:
      graph[curr_node].BN_node_list= [] 
    
    return graph[curr_node].BN_node_list

  BN_node_list= []
  
  assert len(graph[curr_node].child_key_list) == 2, "This function assumes a binarized AC"
  
  child_0_list= populate_BN_list(graph, BN, graph[curr_node].child_key_list[0], done_nodes)
  child_1_list= populate_BN_list(graph, BN, graph[curr_node].child_key_list[1], done_nodes)
  
  child_0_set= set(child_0_list)
  child_1_set= set(child_1_list)

  in_child_0_not_1= child_0_set - child_1_set
  in_child_1_not_0= child_1_set - child_0_set
  
  BN_node_list_clean= list(child_0_set - in_child_0_not_1)
  
  if graph[curr_node].operation_type == common_classes.OPERATOR.PRODUCT:
    BN_node_list= BN_node_list_clean + list(in_child_0_not_1) + list(in_child_1_not_0)
  
  # To find which BN node this sum node is eliminating
  if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
    BN_node_list= BN_node_list_clean

    in_child_0_not_1_list= list(in_child_0_not_1)       
    in_child_1_not_0_list= list(in_child_1_not_0)       
    
    # Create dict of those var that have partial state instantiation
    BN_var_state_0_dict= defaultdict(list)      
    for BN_var_state in in_child_0_not_1_list:
      # This var is already marginalized (or eliminated)
      if '$' not in BN_var_state:
        BN_var_state_0_dict[BN_var_state]= (BN[BN_var_state].states)
        BN_node_list.append(BN_var_state)
      else: # Var with partial state instantition
        BN_var_state = BN_var_state.split('$')
        var= BN_var_state[0]
        state= BN_var_state[1]
        BN_var_state_0_dict[var].append(state)

    
    BN_var_state_1_dict= defaultdict(list)      
    for BN_var_state in in_child_1_not_0_list:
      # This var is already marginalized (or eliminated)
      if '$' not in BN_var_state:
        BN_var_state_1_dict[BN_var_state]= (BN[BN_var_state].states)
        BN_node_list.append(BN_var_state)
      else: # Var with partial state instantition
        BN_var_state = BN_var_state.split('$')
        var= BN_var_state[0]
        state= BN_var_state[1]
        BN_var_state_1_dict[var].append(state)
    
    BN_var_state_merge_dict= defaultdict(list)
    for var, dict_0_states in list(BN_var_state_0_dict.items()):
      BN_var_state_merge_dict[var]= dict_0_states
      
      dict_1_states= BN_var_state_1_dict.pop(var, None)
      if dict_1_states is not None:
        # If following condition is false, the variable had been eliminated before, but one of the indicator has been pruned
        if not (len(set(dict_1_states) - set(dict_0_states)) == 0 or len(set(dict_0_states) - set(dict_1_states)) == 0):
          BN_var_state_merge_dict[var] += dict_1_states
        
          # If var is there in both dicts, it is being eliminated.
          graph[curr_node].elim_BN_var.append(var)
    
    BN_var_state_merge_dict.update(BN_var_state_1_dict)

    for var, states in list(BN_var_state_merge_dict.items()):
      # var completely marginalized
      if len(BN[var].states) == len(states):
        BN_node_list.append(var)
      else: # A few states yet to be marginalized
        for state in states:
          BN_node_list.append(var + '$' + state)
  
  done_nodes[curr_node] = True
  graph[curr_node].BN_node_list = BN_node_list

  return BN_node_list

def print_BN_list(analysis_obj):
  """
  Usage from graph_analysis.py:
    #src.evidence_analysis.populate_BN_list(graph, BN, self.head_node, dict.fromkeys(self.ac_node_list, False) ) 
    #src.evidence_analysis.print_BN_list(self)
  """
  for key, obj in list(analysis_obj.graph.items()):
    if obj.operation_type == common_classes.OPERATOR.SUM and len(obj.BN_node_list) < 4: 
      print(key, obj.BN_node_list)

def set_evidence_in_AC(graph, BN, evidence_dict, leaf_list):
  evidence= evidence_dict
  ac= graph
  
  # Initialize all indicator leafs to True
  set_all_AC_EvidLeaf_true(graph, leaf_list)

  for BN_var, BN_state in list(evidence.items()):
    # Iterate over all states and set and reset Lambdas according to evidence
    for state, AC_key in list(BN[BN_var].AC_leaf_dict.items()):
      if state == BN_state:
        ac[AC_key].curr_val= 1.0
      else:
        ac[AC_key].curr_val= 0.0
        
def set_all_AC_EvidLeaf_true(graph, leaf_list):
  ac= graph

  for leaf in leaf_list:
    leaf_node= ac[leaf]
    if leaf_node.leaf_type == leaf_node.LEAF_TYPE_INDICATOR:
      leaf_node.curr_val= 1.0 

def set_BN_evidence(BN, BN_evidence, net, global_var, iter_type):
  """ Selects a state for BN nodes that are being queried
  """
  assert iter_type in ['rand','exhaustive', 'MAP', 'MAP_30', 'exhaustive_init'], "Invalid iter_type passed"
  
  BN= BN
  evidence= BN_evidence
  net_name= net
  global_var= global_var

  if iter_type == 'rand':
    for BN_var, obj in list(BN.items()):
      if obj.node_type == obj.EVIDENCE_NODE:
        n_states= len(obj.states)
        rand_idx= random.randint(0,n_states-1)
        state_rnd= obj.states[rand_idx]

        evidence[BN_var]= state_rnd
  
  elif iter_type == 'MAP':
    file_content = open(global_var.MAP_INSTANCE_FILE, 'r').readlines()
    read_evidence_file(BN_evidence, file_content)
    print('MAP:',BN_evidence)

  elif iter_type == 'MAP_30':
    file_content = open(global_var.MAP_30_INSTANCE_FILE, 'r').readlines()
    read_evidence_file(BN_evidence, file_content)

  elif iter_type == 'exhaustive_init':
    node_list= list(evidence.keys())
    for node in node_list:
      evidence[node]= BN[node].states[0]
  
  elif iter_type == 'exhaustive':
    node_list= list(evidence.keys())
    incr_flag= True
    for node in node_list:
      if incr_flag:
        curr_state= evidence[node]
        state_pos= BN[node].states.index(curr_state)
        if state_pos == len(BN[node].states)-1:
          next_state= BN[node].states[0]
          incr_flag = True
        else:
          next_state= BN[node].states[state_pos+1]
          incr_flag = False
        
        evidence[node]= next_state
      
      else:
        break

def read_evidence_file(BN_evidence, file_content):
  """ Read evidence file in the format:
  Var_name_1=State
  Var_name_2=State
  """
  
  # reset evidence
  evidence = {}

  for line in file_content:
    # Remove '\n'
    line= line[:-1]
    # Split from '='
    inst= line.split('=')

    var_name= inst[0]
    state= inst[1]

    evidence[var_name]= state
 
  BN_evidence= evidence

def perform_MAP(analysis_obj, evidence_nodes, query_nodes ):
  """ Performs MAP for the (Query var | Evidence var) using the java code provided with SAMIAM
  NOTE: Currently only supports MAP on ALARM net
  """
  
  net_file = analysis_obj.global_var.NET_FILE
  MAP_file = './LOG/MAPTutorial.java'

  MAP_java_command= command
  
  MAP_compile_command= command
   
  subprocess.check_output(MAP_compile_command, shell=True) 
  output = subprocess.check_output(MAP_java_command, shell=True) 

def find_worst_case_query(analysis_obj, node):
  """ Recursively determines which instance of query is the worst case for current node
  Usage from graph_analysis.py:
  #src.evidence_analysis.populate_BN_list
  
  #if unifrom bit-width to be set for every node: src.ac_eval.set_ac_node_bits
  #else custom bit-width to be set-- src.graph_init.read_custom_bits_file
  
  #src.evidence_analysis.find_worst_case_query
  """

  evidence= analysis_obj.BN_evidence
  ac= analysis_obj.graph

lb= 1000.0
m_glob= {}
def find_inst_worst_case_error(graph, head_node, BN, leaf_list, M_orig, path, m, lb_arr, ac_node_list, mode):
  """ This function corresponds to the algorithm 1- "AceMAP(C,M, path)" of the paper 'Solving MAP exactly by searching in Compiled Arithmetic Circuits'
  @param M: at top-level call, set M to a list with all the observable nodes
  path: empty at top-level. It is a dict
  graph: AC with no evidence
  """
  ## Modes:
  # error: finds the worst case combination of observable node for max fixed-point error
  # max: finds the combination for max probability AT HEAD NODE
  # min: finds the combnation for min AT HEAD NODE
  # max and min modes are used to compute the max and min value for every node
  mode_list= ['error', 'max', 'min'] 

  assert mode in mode_list, 'Mode is not supported'

  # Following is not for safety, it is important to create a copy
  M= M_orig[:]
  
  #random.shuffle(M)

  recur_count= 0
  if len(M) != 0:
    X= M.pop(0)
    # curr_evidence is the partial MAP instantiation 
    curr_evidence= {}
    curr_evidence= copy.deepcopy(path)
    BN_states= BN[X].states
    #random.shuffle(BN_states)
    for state in BN_states:
      curr_evidence[X]= state

      set_evidence_in_AC(graph, BN, curr_evidence, leaf_list)
      
      t_start= time.time()
      curr_err, curr_ac_val= eval_val(graph, head_node, M, {}, {}, {}, mode)
      #curr_err, curr_ac_val= eval_val(graph, head_node, M, dict.fromkeys(ac_node_list, False), dict.fromkeys(ac_node_list, 0), dict.fromkeys(ac_node_list, 0), mode)
      #print 'IN2:', time.time() - t_start

      recurse= False
      if mode == 'error':
        if curr_err > lb_arr[0]:
          recurse= True
      elif mode== 'max':
        if curr_ac_val > lb_arr[0]:
          recurse= True
      elif mode== 'min':
        if curr_ac_val < lb_arr[0]:
          recurse= True
      
      recur_count = recur_count + 1
      if recurse:
        ret_dict= find_inst_worst_case_error(graph, head_node, BN, leaf_list, M, curr_evidence, m, lb_arr, ac_node_list, mode)
        recur_count= recur_count + ret_dict['recur_count']
    
    return {'recur_count': recur_count}
  else:
    set_evidence_in_AC(graph, BN, path, leaf_list)
    
    update= False
    if mode == 'error':
      curr_err, curr_ac_val= eval_val(graph, head_node, [], {}, {}, {} , mode)
      #curr_err, curr_ac_val= eval_val(graph, head_node, [], dict.fromkeys(ac_node_list, False), dict.fromkeys(ac_node_list, 0), dict.fromkeys(ac_node_list, 0), mode)
      if curr_err > lb_arr[0]:
        update= True
        m[0]= copy.deepcopy(path)
        lb_arr[0]= curr_err
    
    elif mode== 'max':
      t_start= time.time()
      curr_ac_val= ac_eval.ac_eval(graph, head_node)
      #print 'IN1:', time.time() - t_start
      if curr_ac_val > lb_arr[0]:
        update= True
        m[0]= copy.deepcopy(path)
        lb_arr[0]= curr_ac_val
    
    elif mode== 'min':
      curr_ac_val= ac_eval.ac_eval(graph, head_node)
      if curr_ac_val < lb_arr[0]:
        update= True
        m[0]= copy.deepcopy(path)
        lb_arr[0]= curr_ac_val
    
    return {'recur_count': 1}

def eval_val(graph, curr_node, M, done_nodes, error_val, ac_val, mode):
  """ This function corresponds to the algorithm 2- "eval(C,M)" of the paper 'Solving MAP exactly by searching in Compiled Arithmetic Circuits'
  @param: error_val: stored the val of done nodes. For top-level call, it is initialized to False for every key
  """
  mode_list= ['error', 'max', 'min'] 
  assert mode in mode_list, 'Mode is not supported'
  
  observable_BN_nodes= M
  
  if done_nodes.get(curr_node,  False) == True: 
    return error_val[curr_node], ac_val[curr_node]
  
  # This is a leaf node
  if graph[curr_node].operation_type == common_classes.OPERATOR.LEAF:
    done_nodes[curr_node]= True
    
    if graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_WEIGHT:
      error_val[curr_node] = 2**-graph[curr_node].bits
    elif graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_INDICATOR:
      error_val[curr_node] = 0
    else:
      print("Unsupported leaf type")
      exit(1)
    
    #ac_val[curr_node]= ac_eval.ac_eval(graph, curr_node)
    ac_val[curr_node]= graph[curr_node].curr_val
    
    return error_val[curr_node], ac_val[curr_node]
  
  # If not leaf node and not evaluated yet, compute the error
  child_err_list=[]
  child_ac_list=[]
  child_bit_list=[]
  child_key_list=[]
  for child in graph[curr_node].child_key_list:
    child_err_val, child_ac_val= eval_val(graph,child, observable_BN_nodes, done_nodes, error_val, ac_val, mode) 
    child_err_list.append(child_err_val)
    child_ac_list.append(child_ac_val)
    child_bit_list.append(graph[child].bits)
    child_key_list.append(child)

  if graph[curr_node].operation_type == common_classes.OPERATOR.PRODUCT:
    # Multiply all child val
    for val in child_err_list:
      curr_err_val= child_ac_list[0]*child_err_list[1] + child_ac_list[1]*child_err_list[0] + child_err_list[0]*child_err_list[1]
      curr_ac_val= child_ac_list[0] * child_ac_list[1]

  if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
    # Determine if this sum-operator corresponds to a observable_BN_node
    elim_flag= False
    for var in graph[curr_node].elim_BN_var:
      if var in observable_BN_nodes:
        elim_flag= True
    
    if elim_flag: 
      curr_err_val= 0.0
      for val in child_err_list:
        if val > curr_err_val:
          curr_error_val= val
      
      # For elim_flag true, the sum-operator acts as min or max depending on mode
      if mode == 'error' or mode== 'max':
        curr_ac_val= 0.0
        for val in child_ac_list:
          if val > curr_ac_val:
            curr_ac_val= val
      elif mode == 'min':
        curr_ac_val= 1000.0
        for val in child_ac_list:
          if val < curr_ac_val:
            curr_ac_val= val
        
    else: # Do not correspond to observable_BN_nodes
      curr_err_val=0.0
      for val in child_err_list:
        curr_err_val= curr_err_val + val
      
      curr_ac_val=0.0
      for val in child_ac_list:
        curr_ac_val= curr_ac_val + val
  
  # Error added due to curr operation (Only in product)
  if graph[curr_node].operation_type == common_classes.OPERATOR.PRODUCT:
    curr_err_val= curr_err_val + 2**-graph[curr_node].bits
  if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
    if graph[curr_node].bits < child_bit_list[0] or graph[curr_node].bits < child_bit_list[1]:
      curr_err_val= curr_err_val + 2**-graph[curr_node].bits

  done_nodes[curr_node] = True
  error_val[curr_node]= curr_err_val
  ac_val[curr_node]= curr_ac_val

  return curr_err_val, curr_ac_val

def set_err_to_zero(graph):
  for key, obj in list(graph.items()):
    obj.abs_error_val= 0.0

def find_OverOptimized_nodes(analysis_obj, graph, BN, leaf_list, head_node, map_BinaryOpListKey_To_OriginalKey, worst_case_query):
  """ Tries to find the nodes that are important but are allocated less number of bits
  """
  
  # Reset abs_error_val field of every ac_node
  set_err_to_zero(graph)
  
  # Set worst case evidence in AC
  set_evidence_in_AC(graph, BN, worst_case_query, leaf_list)
 
  # Run error_eval to populate abs_error_val with fresh values
  
  ac_eval.error_eval(analysis_obj, 'fixed', head_node, custom_bitwidth= False, verb=False)
  
  # Print the culprit nodes 
  open_set= queue.Queue()
  open_set.put(head_node) 
  
  done_nodes= set()
  
  culprit_nodes= []
  while not open_set.empty():
    curr_node= open_set.get()
    if curr_node in done_nodes:
      continue
    
    if len(graph[curr_node].child_key_list) > 0:
      child_0= graph[curr_node].child_key_list[0]
      child_1= graph[curr_node].child_key_list[1]
    
      #tolerance_factor= (graph[curr_node].abs_error_val)/4
      tolerance_factor= 0.5 

      own_error=  2**-graph[curr_node].bits
      
      if graph[curr_node].operation_type == common_classes.OPERATOR.PRODUCT:
        child_0_err= graph[child_1].curr_val * graph[child_0].abs_error_val
        child_1_err= graph[child_0].curr_val * graph[child_1].abs_error_val
      if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
        child_0_err= graph[child_0].abs_error_val
        child_1_err= graph[child_1].abs_error_val
      
      if child_0_err >= child_1_err * tolerance_factor and child_0_err >= own_error * tolerance_factor:
        open_set.put(child_0)
      if child_1_err >= child_0_err * tolerance_factor and child_1_err >= own_error * tolerance_factor:
        open_set.put(child_1)
      if own_error >= child_0_err * tolerance_factor and own_error >= child_1_err * tolerance_factor:
        culprit_nodes.append(curr_node)
        #print curr_node,
        #print 'own:' , 2**-graph[curr_node].bits, '|',
        #print 'ac_idx:', map_BinaryOpListKey_To_OriginalKey[curr_node]
    else:
      culprit_nodes.append(curr_node)
      #print curr_node,
      #print 'own:' , 2**-graph[curr_node].bits, '|',
      #print 'ac_idx:', map_BinaryOpListKey_To_OriginalKey[curr_node]

      
    done_nodes.add(curr_node)

    #if 2**-graph[curr_node].bits > err_thresh:
    #  print curr_node,
    #  print 'own:' , 2**-graph[curr_node].bits, '|',
    #  print 'ac_idx:', map_BinaryOpListKey_To_OriginalKey[curr_node]
    #   

    #  if graph[curr_node].operation_type == common_classes.OPERATOR.PRODUCT:
    #    if graph[child_0].curr_val * graph[child_1].abs_error_val > err_thresh:
    #      open_set.put(child_1)
    #    if graph[child_1].curr_val * graph[child_0].abs_error_val > err_thresh:
    #      open_set.put(child_0)
    #  if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
    #    if graph[child_1].abs_error_val > err_thresh:
    #      open_set.put(child_1)
    #    if graph[child_0].abs_error_val > err_thresh:
    #      open_set.put(child_0)
  
  return culprit_nodes

def modify_bits_in_tree_below(graph, head_node, bits, mode='REMOVE'):
  """ To remove or add bits in the enitre tree below given curr node
  Some of the functionalities is a duplicate of src.ac_eval.set_ac_node_bits()
  """
  modes= ['REMOVE', 'REMOVE_SATURATE' ,'ADD', 'SET']
  assert mode in modes, "Invalid mode passed"
  
  open_set= queue.Queue()
  open_set.put(head_node)

  done_nodes=set()
  
  while not open_set.empty():
    curr_node= open_set.get()
    if curr_node in done_nodes:
      continue
    
    curr_obj= graph[curr_node]

    if mode == 'REMOVE':
      curr_obj.bits= curr_obj.bits- bits
      assert curr_obj.bits >=0 , "Number of bits become negative. Use 'REMOVE_SATURATE' mode to keep it 0 in case of negative result"
    elif mode == 'REMOVE_SATURATE':
      curr_obj.bits= curr_obj.bits- bits
      if curr_obj.bits < 0.0:
        curr_obj.bits= 0
    elif mode == 'ADD':
      curr_obj.bits= curr_obj.bits+ bits
    elif mode == 'SET':
      curr_obj.bits= bits
    else:
      assert False, "Mode not implemented"

    for childs in curr_obj.child_key_list:
      open_set.put(childs)
    
    done_nodes.add(curr_node)

def avg_ac_bits(graph):
  total_bits = 0
  for key, obj in list(graph.items()):
    total_bits = total_bits + obj.bits
  
  avg_bits= (total_bits* 1.0)/len(graph) 
  
  return avg_bits

def fload_add_opt_exhaust(graph, BN, BN_evidence, head_node, global_var, leaf_list, ac_node_list, net):
  bits= 23
  lowest_bits= 0

  #BN_evidence= {'HRBP': 'HIGH', 'PAP': 'NORMAL', 'HRSAT': 'HIGH', 'EXPCO2': 'HIGH', 'MINVOL': 'HIGH', 'HYPOVOLEMIA': 'FALSE'}
   
  var_name_lst= list(BN.keys())
  print('Total BN nodes:', len(var_name_lst))
  random.shuffle(var_name_lst)

  obs_var_len=5
  for i in range(obs_var_len):
    BN_evidence[var_name_lst[i]]= None
  
  #BN_evidence= {'TRY24': 'false', 'SYSTEM18': 'false', 'GOAL_142': 'false', 'GOAL_103': 'false', 'RApp9': 'false'} 
  BN_evidence= {'TRY24': 'false', 'SYSTEM18': 'false', 'GOAL_142': 'false'}

  set_BN_evidence(BN, BN_evidence, net, global_var , 'exhaustive_init')
  print(BN_evidence)

  opt_possible= dict.fromkeys(ac_node_list, True)
  reduce_bits= dict.fromkeys(ac_node_list, bits)
  
  opt_done= dict.fromkeys(ac_node_list, False)
  
  for iter in range(2**obs_var_len):
    modify_bits_in_tree_below(graph, head_node, bits, mode='SET')
    
    set_BN_evidence(BN, BN_evidence, net, global_var , 'exhaustive')
    ac_eval.ac_eval_with_evidence(graph, BN, head_node, BN_evidence, leaf_list)

    open_set= queue.Queue()
    open_set.put(head_node)
    
    done_nodes= set()
    graph= graph
  
    opt_node_track= dict.fromkeys(ac_node_list, False)
    
    while not open_set.empty():
      curr_node= open_set.get()
      if curr_node in done_nodes:
        continue
      
      done_nodes.add(curr_node)

      obj= graph[curr_node]
      for childs in obj.child_key_list:
        open_set.put(childs)
      
      BITS_SET= False
      if obj.operation_type == common_classes.OPERATOR.SUM:
        if opt_possible[curr_node] == True:
          child_0= obj.child_key_list[0]
          child_1= obj.child_key_list[1]
          
          child_0_obj= graph[child_0]
          child_1_obj= graph[child_1]

          child_0_val= child_0_obj.curr_val
          child_1_val= child_1_obj.curr_val

          if child_0_val < child_1_val:
            if opt_done[curr_node] == child_1:
              opt_possible[curr_node] = False
            
            else:
              BITS_SET= True
              child_1_obj.bits= obj.bits
              opt_node_track[child_0]= True
              opt_done[curr_node] = child_0
              
              if child_0_val != 0:
                diff_bits= math.log( float(child_1_val)/float(child_0_val), 2)
                if diff_bits < reduce_bits[curr_node]:
                  reduce_bits[curr_node] = diff_bits
                  child_0_obj.bits= obj.bits - diff_bits
                else:
                  child_0_obj.bits= obj.bits - reduce_bits[curr_node]
              else:
                child_0_obj.bits= obj.bits - reduce_bits[curr_node]
              

          elif child_1_val < child_0_val:
            if opt_done[curr_node] == child_0:
              opt_possible[curr_node] = False
            
            else:
              BITS_SET= True
              child_0_obj.bits= obj.bits
              opt_node_track[child_1]= True
              opt_done[curr_node] = child_1
              
              if child_1_val !=0:
                diff_bits= math.log( float(child_0_val)/float(child_1_val), 2)
                if diff_bits < reduce_bits[curr_node]:
                  reduce_bits[curr_node] = diff_bits
                  child_1_obj.bits= obj.bits - diff_bits
                else:
                  child_1_obj.bits= obj.bits - reduce_bits[curr_node]
              else:
                child_1_obj.bits= obj.bits - reduce_bits[curr_node]
              

      if not BITS_SET:
        for child in obj.child_key_list:
          graph[child].bits= obj.bits
          opt_node_track[child]= False

    if iter % 1 == 0:
      ##--- make sure the child always has as many bits as the most important parent
      open_set= queue.Queue()
      open_set.put(head_node)

      done_nodes= set()
      
      for curr_node in range(head_node, -1, -1):
      #while not open_set.empty():
      #   curr_node= open_set.get()
        if curr_node in done_nodes:
          continue

        curr_obj= graph[curr_node]
        for childs in curr_obj.child_key_list:
          open_set.put(childs)
        
        # Make sure current node is not optimized
        if not opt_node_track[curr_node]:
          for parent in curr_obj.parent_key_list:
            if graph[parent].bits > curr_obj.bits:
              curr_obj.bits= graph[parent].bits
        
        done_nodes.add(curr_node)
       
      avg_bits= avg_ac_bits(graph)
      if avg_bits > lowest_bits:
        lowest_bits= avg_bits
      print('AVG bits after optimization_2: ', lowest_bits)
      
      print('Reduction: ', bits- lowest_bits)

def cust_precision_db_sim(graph, BN, final_node, BN_nodes_obs ,evid_db, leaf_list, **kwargs):
  """
  Custom precision simulation for a database of evidences (most probably simulated by SAMIAM)
  
  Inputs:
    A database of evidences: A list of dictionaries. Each dict is a var_name to var_state map
    BN_nodes_obs: A list of nodes that are to be used for ac evaluation
    **kwargs : 
      Example:
        kwargs= {'precision': 'CUSTOM', 'arith_type' : 'FIXED', 'int': 8, 'frac': 23}
      Read more in docstrings of ac_eval.ac_eval()
      
  Outputs:
    A list of ac_eval of each of these
  """
  assert kwargs is not None

  ac_val_lst= []
  print(list(evid_db[0].keys()))

  for evidence_dict in evid_db:
    final_evid_dict= {}
    for node in BN_nodes_obs:
      final_evid_dict[node]= evidence_dict[node]

    val= ac_eval.ac_eval_with_evidence(graph, BN, final_node, final_evid_dict, leaf_list, elim_op= 'SUM', **kwargs)
    ac_val_lst.append(val)  
    if len(ac_val_lst) % 100 == 0:
      print(len(ac_val_lst))

  return ac_val_lst
