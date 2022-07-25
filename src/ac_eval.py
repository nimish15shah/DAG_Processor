from collections import defaultdict
import queue
import math 
import time
import networkx as nx

#**** imports from our codebase *****
from . import common_classes
from . import reporting_tools
from . import graph_init
from . import evidence_analysis
from . import FixedPointImplementation 
from enum import Enum

import logging

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PrecisionConfig():
  def __init__(self):
    self.arith_type_enum= Enum('ARITH_TYPE', 'DEFAULT, FIXED, FLOAT, POSIT', module=__name__)
    self.arith_type= self.arith_type_enum["DEFAULT"]

    self.sign_bits= None
    self.exp_bits= None
    self.frac_bits= None
    self.int_bits= None
    self.total_bits= None
  
  def set_arith_type(self, arith_type):
    """
      first set the relevant exp_bits, frac_bits, etc. before calling this function
    """
    self.arith_type = self.arith_type_enum[arith_type]
    if self.arith_type == self.arith_type_enum.DEFAULT:
      pass
    elif self.arith_type == self.arith_type_enum.FIXED:
      assert self.int_bits != None
      assert self.frac_bits != None
    elif self.arith_type == self.arith_type_enum.FLOAT:
      assert self.exp_bits != None
      assert self.frac_bits != None
    elif self.arith_type == self.arith_type_enum.POSIT:
      assert self.exp_bits != None
      assert self.total_bits != None
      FixedPointImplementation.set_posit_env(self.total_bits, self.exp_bits)
    else:
      assert 0


  def mul(self, in_0, in_1):
    if self.arith_type == self.arith_type_enum.DEFAULT:
      result= in_0 * in_1
    elif self.arith_type == self.arith_type_enum.FIXED:
      result= FixedPointImplementation.fix_mul(in_0, in_1, self.int_bits, self.frac_bits)
    elif self.arith_type == self.arith_type_enum.FLOAT:
      result= FixedPointImplementation.flt_mul(in_0, in_1, self.exp_bits, self.frac_bits, denorm=False)        
    elif self.arith_type == self.arith_type_enum.POSIT:
      result= FixedPointImplementation.posit_mul(in_0, in_1)
    else:
      assert 0

    return result

  def add(self, in_0, in_1):
    if self.arith_type == self.arith_type_enum.DEFAULT:
      result= in_0 + in_1
    elif self.arith_type == self.arith_type_enum.FIXED:
      result= FixedPointImplementation.fix_add(in_0, in_1, self.int_bits, self.frac_bits)
    elif self.arith_type == self.arith_type_enum.FLOAT:
      # result= FixedPointImplementation.flt_add(in_0, in_1, self.exp_bits, self.frac_bits, denorm=False)        
      result= FixedPointImplementation.flt_add_signed(in_0, in_1, self.exp_bits, self.frac_bits, denorm=False)        
    elif self.arith_type == self.arith_type_enum.POSIT:
      result= FixedPointImplementation.posit_add(in_0, in_1)
    else:
      assert 0

    return result

  def to_custom(self, in_0):
    if self.arith_type == self.arith_type_enum.DEFAULT:
      result= in_0
    elif self.arith_type == self.arith_type_enum.FIXED:
      result= FixedPointImplementation.FloatingPntToFixedPoint(in_0, self.int_bits, self.frac_bits)
    elif self.arith_type == self.arith_type_enum.FLOAT:
      result= FixedPointImplementation.flt_to_custom_flt(in_0, self.exp_bits, self.frac_bits, denorm=False)        
    elif self.arith_type == self.arith_type_enum.POSIT:
      result= FixedPointImplementation.flt_to_posit(in_0)
    else:
      assert 0

    return result

  def from_custom(self, in_0):
    if self.arith_type == self.arith_type_enum.DEFAULT:
      result= in_0
    elif self.arith_type == self.arith_type_enum.FIXED:
      result= FixedPointImplementation.FixedPoint_To_FloatingPoint(in_0, self.int_bits, self.frac_bits)
    elif self.arith_type == self.arith_type_enum.FLOAT:
      result= FixedPointImplementation.custom_flt_to_flt(in_0, self.exp_bits, self.frac_bits, denorm=False)        
    elif self.arith_type == self.arith_type_enum.POSIT:
      result= FixedPointImplementation.posit_to_flt(in_0)
    else:
      assert 0

    return result

def ac_eval_with_evidence(graph, BN, final_node, evidence_dict, leaf_list, elim_op= 'SUM', **kwargs):
  evidence_analysis.set_evidence_in_AC(graph, BN, evidence_dict, leaf_list)
  return ac_eval(graph, final_node, elim_op, **kwargs)

def ac_eval_non_recurse(graph, graph_nx, final_node= None, precision_obj= PrecisionConfig()):
  topological_list= list(nx.algorithms.dag.topological_sort(graph_nx))

  for key in topological_list:
    obj= graph[key]
    if obj.is_sum():
      result= precision_obj.to_custom(0.0)
      for child in obj.child_key_list:
        result= precision_obj.add(result, graph[child].curr_val)
      
      obj.curr_val = result

    if obj.is_prod():
      result= precision_obj.to_custom(1.0)
      for child in obj.child_key_list:
        result= precision_obj.mul(result, graph[child].curr_val)
      
      obj.curr_val = result

  if final_node != None:
    return graph[final_node].curr_val
  else:
    return None
  
def ac_eval(graph, final_node, elimop= 'SUM', **kwargs):
  
  assert elimop in ['SUM', 'MAX', 'MIN', 'PROD_BECOMES_SUM'], "Elimination operation provided for AC_eval is invalid"
 
  precision= 'FULL'
  if kwargs is not None:
    if 'precision' in kwargs:
      assert kwargs['precision'] in ['FULL', 'CUSTOM']
      if kwargs['precision'] == 'CUSTOM':
        precision= 'CUSTOM'

    if precision == 'CUSTOM':
      assert 'arith_type' in kwargs, "arith_type has to be passed if precision is CUSTOM"
      assert kwargs['arith_type'] in ['FIXED', 'FLOAT']
      arith_type= kwargs['arith_type']

      if arith_type == 'FIXED':
        assert 'int' in kwargs, "number of int bits has to be passed if arith_type is FIXED"
        assert kwargs['int'] < 50 and kwargs['int'] > 0, "number of int bits should be in range (0,50)"
        int_bits= kwargs['int']
        
        assert 'frac' in kwargs, "number of frac bits has to be passed if arith_type is FIXED"
        assert kwargs['frac'] < 50 and kwargs['frac'] > 0, "number of frac bits should be in range (0,50)"
        frac_bits= kwargs['frac']
        
      if arith_type == 'FLOAT':
        assert 'exp' in kwargs, "number of exp bits has to be passed if arith_type is FLOAT"
        assert kwargs['exp'] < 50 and kwargs['exp'] > 0, "number of exp bits should be in range (0,50)"
        exp_bits= kwargs['exp']
        
        assert 'mant' in kwargs, "number of mant bits has to be passed if arith_type is FLOAT"
        assert kwargs['mant'] < 50 and kwargs['mant'] > 0, "number of mant bits should be in range (0,50)"
        mant_bits= kwargs['mant']

  #ac_node_list= list(graph.keys())

  # A dict to keep track of evaluated nodes
  # Key: Ac node number
  # Val: By default: False
  #done_nodes= dict.fromkeys(ac_node_list, False)
  done_nodes = {}

  #done_nodes= defaultdict(lambda: False, done_nodes) # All keys initialized to 0
  
  # Stores the value of the smallest non-zero number in the entire network for this query
  global_min_val = [1.0]
  
  ac_query_val= ac_eval_recurse(final_node, graph, done_nodes, global_min_val, elimop, **kwargs)
  
  if precision== 'CUSTOM':
    if arith_type == 'FIXED':
      ac_query_val= FixedPointImplementation.FixedPoint_To_FloatingPoint(ac_query_val, int_bits, frac_bits)  
    elif arith_type == 'FLOAT':
      ac_query_val= FixedPointImplementation.custom_flt_to_flt(ac_query_val, exp_bits, mant_bits, denorm= False)  
  
  #print "Smallest val in this query: ", global_min_val[0], 'Exp_bits: ', math.log(abs(math.log(global_min_val[0],2)),2)

  return ac_query_val


def ac_eval_recurse(curr_node, graph, done_nodes, global_min_val, elimop, **kwargs):
  """ kwargs:
    'precision' = FULL or CUSTOM
    'arith_type' = FIXED or FLOAT
    
    Only valid if 'arith_type' is FIXED:
      'int' = Number of integer bits
      'frac' = Number of fraction bits
    
    Only valid if 'arith_type' is FLOAT:
      'exp' = Number of exponent bits
      'mant' = Number of mantissa bits

    Example:
      kwargs= {'precision': 'CUSTOM', 'arith_type' : 'FIXED', 'int': 8, 'frac': 23}
  """

  #--- Process arguments
  assert elimop in ['SUM', 'MAX', 'MIN', 'PROD_BECOMES_SUM'], "Elimination operation provided for AC_eval is invalid"

  precision= 'FULL'
  if kwargs is not None:
    if 'precision' in kwargs:
      assert kwargs['precision'] in ['FULL', 'CUSTOM']
      if kwargs['precision'] == 'CUSTOM':
        precision= 'CUSTOM'

    if precision == 'CUSTOM':
      assert 'arith_type' in kwargs, "arith_type has to be passed if precision is CUSTOM"
      assert kwargs['arith_type'] in ['FIXED', 'FLOAT']
      arith_type= kwargs['arith_type']

      if arith_type == 'FIXED':
        assert 'int' in kwargs, "number of int bits has to be passed if arith_type is FIXED"
        assert kwargs['int'] < 50 and kwargs['int'] > 0, "number of int bits should be in range (0,50)"
        int_bits= kwargs['int']
        
        assert 'frac' in kwargs, "number of frac bits has to be passed if arith_type is FIXED"
        assert kwargs['frac'] < 50 and kwargs['frac'] > 0, "number of frac bits should be in range (0,50)"
        frac_bits= kwargs['frac']
        
      if arith_type == 'FLOAT':
        assert 'exp' in kwargs, "number of exp bits has to be passed if arith_type is FLOAT"
        assert kwargs['exp'] < 50 and kwargs['exp'] > 0, "number of exp bits should be in range (0,50)"
        exp_bits= kwargs['exp']
        
        assert 'mant' in kwargs, "number of mant bits has to be passed if arith_type is FLOAT"
        assert kwargs['mant'] < 50 and kwargs['mant'] > 0, "number of mant bits should be in range (0,50)"
        mant_bits= kwargs['mant']
      

  #---- Functionality starts here------

  if done_nodes.get(curr_node, False) == True:    
    return graph[curr_node].curr_val
  
  # This is a leaf node
  if graph[curr_node].is_leaf():
    #done_nodes[curr_node]= True
    if precision == 'FULL':
      result = graph[curr_node].curr_val
    elif precision == 'CUSTOM':
      if arith_type == 'FIXED':
        result = FixedPointImplementation.FloatingPntToFixedPoint(graph[curr_node].curr_val, int_bits, frac_bits)
      elif arith_type == 'FLOAT':
        result = FixedPointImplementation.flt_to_custom_flt(graph[curr_node].curr_val, exp_bits, mant_bits, denorm= False)
        
    return result 

  # If not leaf node and not evaluated yet
  if graph[curr_node].is_prod():
    if elimop == 'PROD_BECOMES_SUM':
      result = 0
    else:
      result = 1
  elif graph[curr_node].is_sum():
    if elimop == 'SUM' or 'PROD_BECOMES_SUM':
      result = 0
    elif elimop == 'MIN':
      result = 1000
    elif elimop == 'MAX':
      result = 0

  if precision == 'CUSTOM':
    if arith_type == 'FIXED':
      result = FixedPointImplementation.FloatingPntToFixedPoint(result, int_bits, frac_bits)
    elif arith_type == 'FLOAT':
      result = FixedPointImplementation.flt_to_custom_flt(result, exp_bits, mant_bits, denorm=False)
      
  for child in graph[curr_node].child_key_list:
    child_val= ac_eval_recurse(child, graph, done_nodes, global_min_val, elimop, **kwargs)

    if graph[curr_node].is_prod():
      if precision == 'FULL':
        if elimop == 'PROD_BECOMES_SUM':
          result = result + child_val
        else:
          result = result * child_val

      elif precision == 'CUSTOM':
        if arith_type == 'FIXED':
          result= FixedPointImplementation.fix_mul(result, child_val, int_bits, frac_bits)
        elif arith_type == 'FLOAT':
          result= FixedPointImplementation.flt_mul(result, child_val, exp_bits, mant_bits, denorm=False)        

    elif graph[curr_node].is_sum():
      if precision == 'FULL':
        if elimop == 'SUM' or 'PROD_BECOMES_SUM':
          result = result + child_val
        elif elimop == 'MIN':
          if child_val < result:
            result = child_val
        elif elimop == 'MAX':
          if child_val > result:
            result = child_val
      
      elif precision == 'CUSTOM':
        if elimop == 'SUM' or 'PROD_BECOMES_SUM':
          if arith_type == 'FIXED':
            result= FixedPointImplementation.fix_add(result, child_val, int_bits, frac_bits)
          elif arith_type == 'FLOAT':
            # result= FixedPointImplementation.flt_add(result, child_val, exp_bits, mant_bits, denorm=False)        
            result= FixedPointImplementation.flt_add_signed(result, child_val, exp_bits, mant_bits, denorm=False)        
        else:
          assert 0, "Not implemented yet."
  
  if elimop == 'MIN':
    graph[curr_node].min_val= result
    
  done_nodes[curr_node] = True
  graph[curr_node].curr_val= result
  
  # Update gloabl_min_val
  if result < global_min_val[0] and result != 0:
    global_min_val[0] = result
  
  # assert result < 2**63

  return result

def print_ac_val(analysis_obj, print_type= 'curr_val', write_csv= False):
  assert print_type in ['curr_val', 'min_val', 'max_val', 'all'], 'printy_type should be one from the list'
  
  curr_val_list= []
  
  for key, obj in list(analysis_obj.graph.items()):
    print('key=',key, end=' ')
    if print_type == 'curr_val' or print_type == 'all':
      print(',curr_val:',obj.curr_val, end=' ')
      curr_val_list.append(obj.curr_val)
    if print_type == 'min_val' or print_type == 'all':
      print(',min_val:',obj.min_val, end=' ')
    if print_type == 'max_val' or print_type == 'all':
      print(',max_val:',obj.max_val, end=' ')
    print('')

  reporting_tools.reporting_tools._write_csv('./REPORTS/ALARM/alarm_max_val', curr_val_list)
  
def copy_curr_to_max(analysis_obj):
  for key, obj in list(analysis_obj.graph.items()):
    obj.max_val= obj.curr_val

def error_propogate_recurse(analysis_obj, curr_node, done_nodes, arith_type, verb):
  """
  Usage from graph_analysis.py:
    For float:
    #print "Min query val: ", src.ac_eval.ac_eval(self.graph, self.head_node, 'MIN')
    #error= src.ac_eval.error_propogate_recurse(self, self.head_node, dict.fromkeys(self.ac_node_list, False), 'float', bits)
    #error= error -1
    For fixed:
    print  "No evidence AC eval: ", src.ac_eval.ac_eval(self.graph, self.head_node)
    src.ac_eval.copy_curr_to_max(self)
    error= src.ac_eval.error_propogate_recurse(self, self.head_node, dict.fromkeys(self.ac_node_list, False), 'fixed')
  """
  assert arith_type in ['float', 'fixed'], "Invalid arith_type passed tp error propogate"
  
  graph= analysis_obj.graph
  if done_nodes.get(curr_node, False) == True: 
    if arith_type == 'float':
      return graph[curr_node].rel_error_val
    elif arith_type == 'fixed':
      return graph[curr_node].abs_error_val


  # This is a leaf node
  if graph[curr_node].is_leaf():
    done_nodes[curr_node]= True
    
    if arith_type == 'float':
      if graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_WEIGHT:
        graph[curr_node].rel_error_val= (1 + 2**-(graph[curr_node].bits+1))
      elif graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_INDICATOR:
        graph[curr_node].rel_error_val= 1 
      else:
        print("Unsupported leaf type")
        exit(1)
      return graph[curr_node].rel_error_val
    
    elif arith_type == 'fixed':
      if graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_WEIGHT:
        graph[curr_node].abs_error_val = 2**-(graph[curr_node].bits+1)
      elif graph[curr_node].leaf_type== graph[curr_node].LEAF_TYPE_INDICATOR:
        graph[curr_node].abs_error_val = 0
      else:
        print("Unsupported leaf type")
        exit(1)
      return graph[curr_node].abs_error_val
  

  # If not leaf node and not evaluated yet, compute the error
  
  if arith_type == 'float': 
    if graph[curr_node].is_prod():
      result = 1.0
    elif graph[curr_node].is_sum():
      result = 0.0
    
    child_err_list=[]
    child_min_list=[]
    child_max_list=[]
    for child in graph[curr_node].child_key_list:
      child_val= error_propogate_recurse(analysis_obj, child, done_nodes, arith_type, verb)
      child_err_list.append(child_val)
      child_min_list.append(graph[child].min_val)
      child_max_list.append(graph[child].max_val)

      if graph[curr_node].is_prod():
        result = result * child_val
      
      if graph[curr_node].is_sum():
        if child_val > result:
          result= child_val

    #if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
    #  if child_err_list[0] > child_err_list[1]:
    #    if child_err_list[1] != 1:
    #      print "%.2f" % math.log( (child_err_list[0]-1)/(child_err_list[1]-1) ,2),
    #  else:  
    #    if child_err_list[0] != 1:
    #      print "%.2f" % math.log( (child_err_list[1]-1)/(child_err_list[0]-1) ,2),
      
    #if graph[curr_node].operation_type == common_classes.OPERATOR.SUM:
      #denorm_sum= sum(child_min_list)
      #for idx, err  in enumerate(child_err_list):
      #  numer= child_max_list[idx]
      #  denorm= denorm_sum - child_min_list[idx] + child_max_list[idx]
      #  print numer, denorm, err*(numer/denorm), err
      #  result= result + err*(numer/denorm)
    
    result= result * (1+ 2**-(graph[curr_node].bits+1))
    
    done_nodes[curr_node] = True
    graph[curr_node].rel_error_val= result 
    
    return result

  elif arith_type == 'fixed':
    result = 0.0
    
    child_err_list=[]
    child_min_list=[]
    child_max_list=[]
    child_bit_list=[]
    child_key_list=[]
    for child in graph[curr_node].child_key_list:
      child_val= error_propogate_recurse(analysis_obj, child, done_nodes, arith_type, verb)
      child_err_list.append(child_val)
      child_min_list.append(graph[child].min_val)
      child_max_list.append(graph[child].max_val)
      child_bit_list.append(graph[child].bits)
      child_key_list.append(child)

    # Error propogated from children
    # Following code assumes 2 children, i.e., AC is binarized
    if graph[curr_node].is_prod():
      result= child_max_list[0]*child_err_list[1] + child_max_list[1]*child_err_list[0] + child_err_list[0]*child_err_list[1]
    if graph[curr_node].is_sum():
      result= child_err_list[0] + child_err_list[1]
    
    # Error added due to curr operation (Only in product)
    if graph[curr_node].is_prod():
      result= result + 2**-(graph[curr_node].bits+1)
    if graph[curr_node].is_sum():
      if graph[curr_node].bits < child_bit_list[0] or graph[curr_node].bits < child_bit_list[1]:
        result= result + 2**-(graph[curr_node].bits+1)
  
    # Prints to find culprit node
    if verb:
      print(curr_node, end=' ')
      print(result, '|', end=' ')
      if graph[curr_node].is_prod():
        print('own:' , 2**-(graph[curr_node].bits+1), '|', end=' ')
      if graph[curr_node].is_sum():
        if graph[curr_node].bits < child_bit_list[0] or graph[curr_node].bits < child_bit_list[1]:
          print('own:' , 2**-(graph[curr_node].bits+1), '|', end=' ')
        else:
          print('own:', 0, '|', end=' ')
      
      print(child_key_list[0], end=' ')
      if graph[curr_node].is_prod():
        print(child_err_list[0] * child_max_list[1], end=' ')
      if graph[curr_node].is_sum():
        print(child_err_list[0], end=' ')
      print('|', end=' ')

      print(child_key_list[1], end=' ')
      if graph[curr_node].is_prod():
        print(child_err_list[1] * child_max_list[0], end=' ')
      if graph[curr_node].is_sum():
        print(child_err_list[1], end=' ')
      print('|', end=' ') 
      
      if graph[curr_node].is_prod():
        print(child_err_list[1] * child_err_list[0], end=' ')
      print(' ')

    done_nodes[curr_node] = True
    graph[curr_node].abs_error_val= result 
    
    return result

def error_eval(self, arith_type, node, custom_bitwidth= False, verb= False):
  """ In case of error_eval for uniform bit width, set bit_width of each node using set_ac_node_bits method before calling this method
  Usage from graph_analysis.py:
    src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= False)
    OR
    src.ac_eval.set_ac_node_bits(self, bits= 21)
    src.ac_eval.error_eval(self, 'fixed', self.head_node, custom_bitwidth= True)
  """
  assert arith_type in ['float', 'fixed'], "Invalid arith_type passed tp error propogate"
  #assert bits > 0 and bits < 54, "bits should be greater than 0 and less than 54 (max limit due to double-precision float limit)"
  
  if custom_bitwidth:
    bit_content= open(self.global_var.BITWIDTH_FILE, 'r').readlines()
    graph_init.read_custom_bits_file(self, bit_content)
    

  if arith_type == 'float':
    #ac_eval(self.graph, node, 'MIN')
    error= error_propogate_recurse(self, node, dict.fromkeys(self.ac_node_list, False), 'float', verb)
    error= error -1
  
  elif arith_type =='fixed': 
    #print "AC val with given evidence:", 
    ac_eval(self.graph, node)
    copy_curr_to_max(self)
    error= error_propogate_recurse(self, node, dict.fromkeys(self.ac_node_list, False), 'fixed', verb)
  
  return error

def set_ac_node_bits(analysis_obj, bits):
  assert bits > 0 and bits < 54, "bits should be greater than 0 and less than 54 (max limit due to double-precision float limit)"
  for key, obj in list(analysis_obj.graph.items()):
    obj.bits= bits
  

