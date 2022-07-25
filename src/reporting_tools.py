  
import csv
import queue
from collections import defaultdict
import random
import itertools
from math import sqrt
import math
from statistics import mean

import pydot
#import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np

import logging

logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#**** imports from our codebase *****
from . import common_classes
from . import ac_eval
from . import useful_methods

def _write_csv(file_name, list_name, mode= 'wb+'):
  with open(file_name, mode) as f:
    wr = csv.writer(f)
    wr.writerow(list_name)

def _read_csv(file_name):
  with open(file_name) as f:
    rd= csv.reader(f, delimiter=',')
    data= [row for row in rd]
    return data

def geo_mean(iterable):
  a = np.array(iterable)
  return a.prod()**(1.0/len(a))

def geo_mean_overflow(iterable):
  """
    avoids overflow
  """
  a = np.log(iterable)
  return np.exp(a.mean())

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
  
#def create_dot_file(graph, BN, ac_node_list, dot_file_name, option, ac_node):
def create_dot_file(**kw_args):
  """
  Converting .dot file: dot -Tpng alarm_BN_var_under_AC_node.dot > alarm_BN_var_under_AC_node.png
  Viewing .dot files:
    command: zgrviewer
  """
  #graph= analysis_obj.graph

  option_list=['mark_computed', 'BN_var_details', 'color_according_to_bits', 'color_partition']
  option= kw_args['option']

  assert isinstance(option, str), "option should be of string type"
  assert (option in option_list), "Unrecognized option" 
  
  dot_file_name= kw_args['dot_file_name']
  f= open(dot_file_name, 'w')
  
  f.write('digraph G {\n')
  
  if option == 'mark_computed': 
    for node in graph:
      f.write( str(node) + ' [shape=ellipse, label=' + str(node) + ' ,style=\"filled\", fillcolor=\"')
      if (graph[node].computed):
        if (graph[node].operation_type == src.common_classes.OPERATOR.LEAF):
          f.write('green\"]\n')
        else:
          f.write('red\"]\n')
      else:
        f.write('azure\"]\n')

      for parent in graph[node].parent_key_list:
        f.write(str(parent) + ' ->' + str(node) + ' [dir=none]\n')
  
  elif option == 'BN_var_details':
    #done_nodes= dict.fromkeys(analysis_obj.ac_node_list, False)
    #BN= analysis_obj.BN
    done_nodes= dict.fromkeys(ac_node_list, False)

    open_set= queue.Queue()
    open_set.put(ac_node) 
    
    while not open_set.empty():
      curr_node= open_set.get()
      if done_nodes[curr_node]:
        continue
      
      curr_obj= graph[curr_node]
      
      f.write( str(curr_node) + ' [shape=ellipse, label=\"')
      if (curr_obj.operation_type == common_classes.OPERATOR.PRODUCT) or (curr_obj.operation_type == common_classes.OPERATOR.SUM):
        if (curr_obj.operation_type == common_classes.OPERATOR.PRODUCT):
          f.write('* ::'+ str(curr_node) )
        elif (curr_obj.operation_type == common_classes.OPERATOR.SUM):
          f.write('+ ::'+ str(curr_node) )
        
        BN_var_state_dict= defaultdict(list)
        for BN_var_state in curr_obj.BN_node_list:
          assert '$' in BN_var_state, "FIXME: This code has to be modified as the convention in evidence_analysis.populate_BN_list has been modified"
          BN_var_state = BN_var_state.split('$')
          var= BN_var_state[0]
          state= BN_var_state[1]
          BN_var_state_dict[var].append(state)
        
        for BN_var, state_ls in list(BN_var_state_dict.items()):
          f.write('|' + BN_var)
          ## If not all the states are covered, put the name of all states
          if not len(BN[BN_var].states) == len(state_ls):
            f.write('(')
            for name in state_ls:
              f.write(',' + name)
            f.write(')')
        
        f.write('|ELIM:::')
        for BN_var in graph[curr_node].elim_BN_var:
          f.write(BN_var)

      elif (curr_obj.operation_type == common_classes.OPERATOR.LEAF):
        if curr_obj.leaf_type == curr_obj.LEAF_TYPE_INDICATOR:
          f.write(curr_obj.leaf_BN_node_name + '$' + curr_obj.leaf_BN_node_state)
        elif curr_obj.leaf_type == curr_obj.LEAF_TYPE_WEIGHT:
          f.write(str(curr_obj.curr_val))
      
      f.write('\"]\n')

      for child in curr_obj.child_key_list:
        f.write(str(curr_node) + ' ->' + str(child) + '\n')
        open_set.put(child)

      done_nodes[curr_node]= True
  
  elif option == 'color_according_to_bits':
    mbit= graph[ac_node].bits
    mbit_R= 0.666 * mbit
    mbit_G= 0.333 * mbit
    mbit_B= 0 * mbit
    mbit_res= 0.333 * mbit

    for node in graph:
      #f.write( str(node) + ' [shape=ellipse, label=\"' + str(node) + '_' + str(graph[node].operation_type)+ '_' + str("{0:.2f}".format(graph[node].bits)) + '_' +  str("{0:.2E}".format(graph[node].max_val)) + '_' + str("{0:.2E}".format(graph[node].min_val))  + '\" ,style=\"filled\", fillcolor=\"#')
      f.write( str(node) + ' [shape=ellipse, label=\"' + str(node) + '_' + str(graph[node].operation_type) + '\" ,style=\"filled\", fillcolor=\"#')
      curr_bits= graph[node].bits
      R=  max(0, curr_bits- mbit_R)
      R= int(R/mbit_res * 255.0)
      R= hex(R)[2:]

      G=  max(0, curr_bits- mbit_G)
      G= float(G/mbit_res)
      if (G > 1):
        G= 255
      else:
        G= G *255
      G= hex(int(G))[2:]

      B=  max(0, curr_bits- mbit_B)
      B= float(B/mbit_res)
      if (B > 1):
        B= 255
      else:
        B= B *255
      B= hex(int(B))[2:]
      
      if len(R) == 1:
        R= '0' + R
      if len(R) == 0:
        R= '00'

      if len(G) == 1:
        G= '0' + G
      if len(G) == 0:
        G= '00'
      
      if len(B) == 1:
        B= '0' + B
      if len(R) == 0:
        B= '00'
      
      if curr_bits > 2:   
        f.write(R + G  + B + '\"]\n')
      else:
        f.write(R + G  + B + '\"]\n')
        #f.write('FF0000\"]\n')
      
      for parent in graph[node].parent_key_list:
        f.write(str(parent) + ' ->' + str(node) + ' [dir=none]\n')
  elif option == 'color_partition':
    BB_graph= kw_args['BB_graph']
    part_dict= kw_args['part_dict']
    color_pallet= {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    for bb in BB_graph:
      f.write( str(bb) + ' [shape=ellipse, label=' + str(bb) + ' ,style=\"filled\", fillcolor=\"')
      f.write(color_pallet[part_dict[bb]] + '"]\n')

      for parent in BB_graph[bb].parent_bb_lst_duplicates:
        f.write(str(parent) + ' ->' + str(bb) + ' [dir=none]\n')
      
  else:
    assert False, 'Mode not supported'

  f.write('}\n')

def write_vect_c(file_name, graph, BB_graph):
  f= open(file_name, 'w') 
  sorted_bb= sorted(list(BB_graph.values()), key= lambda x:x.sch_level) 
  
  ptr_0_lst=[] 
  ptr_1_lst=[] 
  ptr_r_lst=[] 

  for bb in sorted_bb:
    for node in bb.out_list:
      ptr_r_lst.append(node)
      ptr_0_lst.append(graph[node].child_key_list[0])
      ptr_1_lst.append(graph[node].child_key_list[1])
  
  assert len(ptr_r_lst) == len(set(ptr_r_lst))
  # Rename to avoid scattering
  # Key: old
  # val: new
  rename_map= {}
  for idx, var in enumerate(ptr_r_lst): 
    rename_map[var] = idx
  
  # Rename leaves as well
  avail_key= len(ptr_r_lst)
  assert avail_key not in list(rename_map.values())
  for node in graph:
    if node not in rename_map:
      #assert graph[node].operation_type == common_classes.OPERATOR.LEAF
      rename_map[node] = avail_key
      avail_key += 1
  
  assert len(list(rename_map.values())) == len(set(rename_map.values()))
  
  # REplace in original ptr lsits
  ptr_0_lst_new= [rename_map[node] for node in ptr_0_lst]
  ptr_1_lst_new= [rename_map[node] for node in ptr_1_lst]
  ptr_r_lst_new= [rename_map[node] for node in ptr_r_lst]

  assert len(set(ptr_0_lst)) == len(set(ptr_0_lst_new))
  assert len(set(ptr_1_lst)) == len(set(ptr_1_lst_new))
  assert len(set(ptr_r_lst)) == len(set(ptr_r_lst_new))
  
  ptr_r_lst= ptr_r_lst_new
  ptr_0_lst= ptr_0_lst_new
  ptr_1_lst= ptr_1_lst_new

  f.write('#include <stdint.h>\n')
  f.write('#include <immintrin.h>\n')
  
  assert len(ptr_0_lst) == len(ptr_1_lst)
  assert len(ptr_0_lst) == len(ptr_r_lst)

  f.write('int main() {\n')
  f.write('  int a['+ str(len(graph)) +'];\n')
  f.write('  int ptr_0_lst[' + str(len(ptr_0_lst)) + ']= {')
  for idx, val in enumerate(ptr_0_lst):
    if idx != 0:
      f.write(', ' + str(val))
    else:
      f.write(str(val))
  f.write('};\n')
  
  f.write('  int ptr_1_lst[' + str(len(ptr_1_lst))  + ']= {')
  for idx, val in enumerate(ptr_1_lst):
    if idx != 0:
      f.write(', ' + str(val))
    else:
      f.write(str(val))
  f.write('};\n')

  #f.write('  int ptr_r_lst[' + str(len(ptr_r_lst)) + ']= {')
  #for idx, val in enumerate(ptr_r_lst):
  #  if idx != 0:
  #    f.write(', ' + str(val))
  #  else:
  #    f.write(str(val))
  #f.write('};\n')
  #f.write('\n')
  
  f.write('  int i,j,k;\n')
  f.write('  for (j=0;j<1000; j++) {\n')
  f.write('#pragma ivdep\n')
  f.write('    for (i=0;i<' + str(len(ptr_r_lst)) + '; i++) {\n')
  #f.write('       a[ptr_r_lst[i]] = a[ptr_0_lst[i]] + a[ptr_1_lst[i]];\n') 
  f.write('       a[i] = a[ptr_0_lst[i]] + a[ptr_1_lst[i]];\n') 
  f.write('    }\n')
  f.write('  }\n')
  f.write('}\n')

def write_c_for_intel_cpu(file_name, graph):  
  f= open(file_name, 'w') 

  for key, node in list(graph.items()):
    if node.operation_type == common_classes.OPERATOR.LEAF:
      f.write('float val_' + str(key) + ' = ' + str(node.curr_val) + ";")
  
  f.write("\n\n")
  f.write('float compute();\n')
  f.write('int main(void) { \n')
  f.write('  int i,j,k;\n')
  f.write('  for (j=0;j<100000; j++) {\n')
  f.write('    compute();')
  f.write('  }\n')
  f.write('}\n')

  f.write('float compute(void) { \n')
  
  # Declaration
  f.write('  float ')
  for key, node in list(graph.items()):
    if not node.is_leaf:
      f.write('val_' + str(key) + ', ')
  f.write('val_dummy')
  f.write(';\n')

  # Arithmetic
  for key, node in list(graph.items()):
    #if node.operation_type == common_classes.OPERATOR.LEAF:
    #  f.write(" val_" + str(key) + " = " + str(int(node.curr_val*100)) + ";\n")
    
    if node.is_prod():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) + " * ")

    elif node.is_sum():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) +  " + ")
    else:
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) +  " * ")

  f.write(" return val_" + str(len(graph)-1) + ";\n")
  f.write('}\n')


def write_c_for_asip(file_name, graph):
  f= open(file_name, 'w') 

  for key, node in list(graph.items()):
    if node.operation_type == common_classes.OPERATOR.LEAF:
      f.write('int val_' + str(key) + ' = ' + str(int(node.curr_val*100)) + ";")
  
  f.write("\n\n")
  f.write('int compute();\n')
  f.write('int main(void) { \n')
  f.write('  int i,j,k;\n')
  f.write('  for (j=0;j<100000; j++) {\n')
  f.write('    compute();')
  f.write('  }\n')
  f.write('}\n')

  f.write('int compute(void) { \n')
  
  # Declaration
  f.write('  int ')
  for key, node in list(graph.items()):
    if not node.is_leaf:
      f.write('val_' + str(key) + ', ')
  f.write('val_dummy')
  f.write(';\n')

  # Arithmetic
  for key, node in list(graph.items()):
    #if node.operation_type == common_classes.OPERATOR.LEAF:
    #  f.write(" val_" + str(key) + " = " + str(int(node.curr_val*100)) + ";\n")
    
    if node.is_prod():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) + " * ")

    elif node.is_sum():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) +  " + ")
    else:
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          f.write(" val_"+ str(child) + ";\n")
        else:
          f.write(" val_"+ str(child) +  " * ")

  f.write(" return val_" + str(len(graph)-1) + ";\n")
  f.write('}\n')

def write_c_for_asip_1(file_name, BB_graph, graph, vect_len):
  f= open(file_name, 'w')

  f.write('int chess_storage(DM%VSIZE) A[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) B[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) T[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) R[' + str(len(graph)) +'];\n')
  
  f.write('int main () {\n')
  f.write('   vint *v1, *v2, *vR;\n')
  f.write('   int *r;')
  f.write('   v1 = (vint *) A;')
  f.write('   v2 = (vint *) B;')
  f.write('   vR = (vint *) T;')

  sorted_bb= sorted(list(BB_graph.values()), key= lambda x:x.sch_level) 
  map_bb_to_in_list= {}
  map_bb_to_out_list= {}

  for bb_obj in sorted_bb:
    input_list= []
    output_list= []
    for unit in bb_obj.set_of_decompose_units:
      assert len(unit.all_nodes) == 3
      input_list += unit.input_list
      output_list += list(unit.uncomputed_nodes)
      
    assert len(input_list) == 2*len(output_list)
    
    for extra in range(vect_len - len(output_list)):
      output_list.append(None)
      input_list.append(None)
      input_list.append(None)
     
    assert len(input_list) == 2*len(output_list)
    assert vect_len == len(output_list)
    
    map_bb_to_in_list[bb_obj] = list(input_list)
    map_bb_to_out_list[bb_obj] = list(output_list)
    
    for idx, in_var in enumerate(input_list):
      if in_var == None:
        continue

      if idx % 2 == 0:
        f.write('   A[' +str(idx/2) + '] = R[' + str(in_var) + '];\n'  )
      else: 
        f.write('   B[' +str(idx/2 + 1) + '] = R[' + str(in_var) + '];\n'  )    
    f.write('\n')
    
    f.write('   *vR= *v1 + *v2;\n')
    f.write('   r= (int *)vR;\n')
    for idx, out_var in enumerate(output_list):
      if out_var == None:
        continue

      f.write('   R[' + str(out_var)+'] = *(r+' + str(idx) + ');\n')
  
  f.write('\n')
  f.write(' chess_report(R);')
  f.write('}')    

def write_c_for_asip_2(file_name, BB_graph, graph, vect_len, leaf_list):
  f= open(file_name, 'w')

  f.write('int chess_storage(DM%VSIZE) A[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) B[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) T[VSIZE];\n')
  f.write('int chess_storage(DM%VSIZE) IN[' + str(len(leaf_list))+ '];\n')
  
  f.write('int chess_storage(DM%VSIZE) R[' + str(len(BB_graph)) +'][VSIZE];\n')
  
  f.write('int main () {\n')
  f.write('   vint *v1, *v2, *vR;\n')
  f.write('   int *r;')
  f.write('   v1 = (vint *) A;')
  f.write('   v2 = (vint *) B;')
  f.write('   vR = (vint *) T;')

  sorted_bb= sorted(list(BB_graph.keys()), key= lambda x:BB_graph[x].sch_level) 
  map_bb_to_in_list= {}
  map_bb_to_out_list= {}

  for bb in sorted_bb:
    input_list= []
    output_list= []
    for unit in BB_graph[bb].set_of_decompose_units:
      assert len(unit.all_nodes) == 3
      input_list += unit.input_list
      output_list += list(unit.uncomputed_nodes)
      
    assert len(input_list) == 2*len(output_list)
    
    for extra in range(vect_len - len(output_list)):
      output_list.append(None)
      input_list.append(None)
      input_list.append(None)
     
    assert len(input_list) == 2*len(output_list)
    assert vect_len == len(output_list)
    
    map_bb_to_in_list[bb] = list(input_list)
    map_bb_to_out_list[bb] = list(output_list)
    
    for idx, in_var in enumerate(input_list):
      if in_var == None:
        continue
      
      if graph[in_var].operation_type == common_classes.OPERATOR.LEAF:
        RHS= 'IN[' + str(in_var) + '];\n'
      else:
        storing_bb= graph[in_var].storing_builblk
        storing_idx= map_bb_to_out_list[storing_bb].index(in_var)

        RHS= 'R[' + str(storing_bb) + '] [ ' + str(storing_idx)+'];\n'


      if idx % 2 == 0:
        f.write('   A[' +str(idx/2) + '] = ' + RHS  )
      else: 
        f.write('   B[' +str(idx/2 + 1) + '] = ' + RHS  )    
    f.write('\n')
    
    f.write('   vR= (vint*) R[' +str(bb) + '];')
    f.write('   *vR= *v1 + *v2;\n')
    #f.write('   r= (int *)vR;\n')
    #for idx, out_var in enumerate(output_list):
    #  if out_var == None:
    #    continue

    #  f.write('   R[' + str(out_var)+'] = *(r+' + str(idx) + ');\n')
  
  f.write('\n')
  f.write(' chess_report(R);')
  f.write('}')    

def write_c_for_gpu_cuda(file_name, graph):
  f= open(file_name, 'w+') 

  for key, node in list(graph.items()):
    if node.operation_type == common_classes.OPERATOR.LEAF:
      f.write('int val_' + str(key) + ' = ' + str(int(node.curr_val*100)) + ";")
  
  f.write("\n\n")
  f.write('int compute();\n')
  f.write('int main(void) { \n')
  f.write('  int i,j,k;\n')
  f.write('  for (j=0;j<100000; j++) {\n')
  f.write('    compute();')
  f.write('  }\n')
  f.write('}\n')

  f.write('int compute(void) { \n')
  
  # Declaration
  f.write('  int ')
  for key, node in list(graph.items()):
    if not node.is_leaf():
      f.write('val_' + str(key) + ', ')
  f.write('val_dummy')
  f.write(';\n')

  # Arithmetic
  for key, node in list(graph.items()):
    #if node.operation_type == common_classes.OPERATOR.LEAF:
    #  f.write(" val_" + str(key) + " = " + str(int(node.curr_val*100)) + ";\n")
    
    if node.is_prod():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          if graph[child].is_leaf():
            f.write(" A[idx_off + "+ str(child) + "];\n")
          else:
            f.write(" val_"+ str(child) + ";\n")
        else:
          if graph[child].is_leaf():
            f.write(" A[idx_off + "+ str(child) + "] * ")
          else:
            f.write(" val_"+ str(child) + " * ")

    elif node.is_sum():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if child == node.child_key_list[-1]:
          if graph[child].is_leaf():
            f.write(" A[idx_off + "+ str(child) + "];\n")
          else:
            f.write(" val_"+ str(child) + ";\n")

        else:
          if graph[child].is_leaf():
            f.write(" A[idx_off + "+ str(child) +  "] + ")
          else:
            f.write(" val_"+ str(child) +  " + ")

  f.write(" A[idx_off]= val_" + str(len(graph)-1) + ";\n")
  f.write('}\n')


def write_c_for_gpu_cuda_1(file_name, graph):
  f= open(file_name, 'w+') 

  for key, node in list(graph.items()):
    if node.operation_type == common_classes.OPERATOR.LEAF:
      f.write('int val_' + str(key) + ' = ' + str(int(node.curr_val*100)) + ";")
  
  f.write("\n\n")
  f.write('int compute();\n')
  f.write('int main(void) { \n')
  f.write('  int i,j,k;\n')
  f.write('  for (j=0;j<100000; j++) {\n')
  f.write('    compute();')
  f.write('  }\n')
  f.write('}\n')

  f.write('int compute(void) { \n')
  
  # Declaration
  f.write('  int ')
  for key, node in list(graph.items()):
    if not node.is_leaf():
      f.write('val_' + str(key) + ', ')
  f.write('val_dummy')
  f.write(';\n')
  
  # Distinguish between weight and indicators
  map_leaf_to_weight_id= {}
  map_leaf_to_ind_id= {}
  ind_cnt=0
  weigh_cnt= 0
  for key, node in list(graph.items()):
    if node.is_weight():
      map_leaf_to_weight_id[key]= weigh_cnt
      weigh_cnt += 1 

    if node.is_indicator():
      map_leaf_to_ind_id[key]= ind_cnt
      ind_cnt += 1 

  # Arithmetic
  for key, node in list(graph.items()):
    #if node.operation_type == common_classes.OPERATOR.LEAF:
    #  f.write(" val_" + str(key) + " = " + str(int(node.curr_val*100)) + ";\n")
    
    if node.is_prod() or node.is_sum():
      f.write(" val_" + str(key) + " = ")
      for child in node.child_key_list:
        if graph[child].is_leaf():
          if graph[child].is_indicator():
            f.write(" A[idx_off + "+ str(map_leaf_to_ind_id[child]) + "]")
          else:
            f.write(" B["+ str(map_leaf_to_weight_id[child]) + "]")
        else:
          f.write(" val_"+ str(child))

        if child == node.child_key_list[-1]:
          f.write(";\n")
        else:
          if node.is_prod():
            f.write(" * ")
          elif node.is_sum():
            f.write(" + ")

  f.write(" A[idx_off] += val_" + str(len(graph)-1) + ";\n")
  f.write('}\n')

def write_c_for_gpu_cuda_2(file_name, graph, graph_nx, final_node, N_THREADS=64):
  """
    with multiple threads
  """
  
  # preprocess leaf nodes
  map_node_to_id= {}
  curr_id= 0
  A= []  # list that stores the values of the input
  for node, obj in list(graph.items()):
    if obj.is_leaf():
      map_node_to_id[node]= curr_id
      obj.curr_val= random.uniform(0.5,1)
      A.append(obj.curr_val)
      curr_id += 1
  
  # Make length of A multiple of threads
  for i in range(N_THREADS - (len(A)%N_THREADS)):
    A.append(50000.0)
  assert len(A)% N_THREADS == 0

  
  assign_reverse_level_graph(graph, graph_nx)
  map_lvl_to_nodes= defaultdict(list)
  for node, obj in list(graph.items()):
    map_lvl_to_nodes[obj.reverse_level].append(node)
  
  # delete leaf nodes
  del map_lvl_to_nodes[0]

  B=[]  # first input idx
  C=[]  # second input idx
  Op= [] # operation
  sync_chunk_idx= []

  overall_chunk_idx= len(A)/N_THREADS
  ac_chunk_idx= 0
  for lvl in sorted(list(map_lvl_to_nodes.keys())):
    nodes= map_lvl_to_nodes[lvl]
    
    # Break list into chunks of size N_THREADS
    node_chunks= [nodes[i:i+N_THREADS] for i in range(0, len(nodes), N_THREADS)]
    
    B.append([])
    C.append([])
    Op.append([])
  
    for chunks in node_chunks:
      B[-1].append([])
      C[-1].append([])
      Op[-1].append([])

      assert len(chunks) <= N_THREADS
      for node_idx, node in enumerate(chunks):
        assert overall_chunk_idx * N_THREADS + node_idx not in list(map_node_to_id.values())
        assert node not in map_node_to_id
        map_node_to_id[node] = overall_chunk_idx * N_THREADS + node_idx

        obj= graph[node]
        child_0= obj.child_key_list[0]
        child_1= obj.child_key_list[1]

        B[-1][-1].append(map_node_to_id[child_0])
        C[-1][-1].append(map_node_to_id[child_1])

        if obj.is_sum():
          Op[-1][-1].append('sum')
        elif obj.is_prod():
          Op[-1][-1].append('prod')
        else:
          assert 0

      overall_chunk_idx += 1      
      ac_chunk_idx += 1

    sync_chunk_idx.append(ac_chunk_idx)
  
  # Golden ref
  print("Noraml:", ac_eval.ac_eval(graph, final_node))
#  print [{map_node_to_id[node]: obj.curr_val} for node,obj in graph.items() if not obj.is_leaf()]

  # Print to cuda file
  f= open(file_name, 'w+') 

  B_ls= []
  C_ls= []
  Op_ls= []
  for lvl in range(len(B)):
    B_chunks= B[lvl]
    C_chunks= C[lvl]
    Op_chunks= Op[lvl]

    for chunk_idx in range(len(B_chunks)):
      B_indices= B_chunks[chunk_idx] 
      C_indices= C_chunks[chunk_idx] 
      Op_indices= Op_chunks[chunk_idx] 

      for idx in range(N_THREADS):
        if idx < len(B_indices):
          B_idx= B_indices[idx]
          C_idx= C_indices[idx]
          Op_idx= Op_indices[idx]
        else:
          B_idx= idx  # To avoid bank conflict
          C_idx= idx
          Op_idx= 'sum'
        
        B_ls.append(B_idx)
        C_ls.append(C_idx)

        if Op_idx == 'sum':
          Op_ls.append(0)
        elif Op_idx == 'prod':
          Op_ls.append(1)
        else:
          assert 0
  
  # Initialize constant lists, i.e. inputs and indices
  f.write("float h_A[]= {\n")
  f.write(str(A)[1:-1])
  f.write("};\n")

  f.write("int h_B[]= {\n")
  f.write(str(B_ls)[1:-1])
  f.write("};\n")

  f.write("int h_C[]= {\n")
  f.write(str(C_ls)[1:-1])
  f.write("};\n")

  f.write("bool h_Op[]= {\n")
  f.write(str(Op_ls)[1:-1])
  f.write("};\n")
  
  f.write("#define THREADS_PER_BLOCK " + str(N_THREADS) + '\n')
  f.write("#define BLOCKS_PER_GRID " + str(1) + '\n')
  f.write("#define SIZE_OF_IN " + str(len(A)) + '\n')
  f.write("#define SIZE_OF_AC " + str(len(B_ls)) + '\n')

  input_chunks= [A[i:i+N_THREADS] for i in range(0, len(A), N_THREADS)]

  # Actual kernel
  f.write("__device__ void\n")
  f.write("ac(float *A, const int *B, const int *C, const bool *Op, int n_iter) { \n")
  f.write("  int i= blockDim.x * blockIdx.x + threadIdx.x;\n")
  f.write("  __shared__  float R[" + str(sync_chunk_idx[-1] + len(input_chunks)) + "*THREADS_PER_BLOCK];\n")
  f.write("  const int t= THREADS_PER_BLOCK;\n")
  f.write("  __shared__ float final;\n")
  f.write("  final=0;\n")
  
  # Copy inputs
  for idx, chunk in enumerate(input_chunks):
    f.write("  R[i + " + str(idx)+ "*t] = A[i + " + str(idx)+ "*t];\n")
  f.write('  __syncthreads();\n')
    
  # Actual compute
  f.write('  for (int iter=0; iter< n_iter; iter++) {\n')
  for i in range(sync_chunk_idx[-1]):
    if i in sync_chunk_idx:
      f.write('    __syncthreads();\n')
    
    j= i + len(input_chunks)

    f.write("    R[i + " + str(j)+ "*t] = Op[i + " + str(i)+ "*t] ? ")
    f.write("R[B[i + " + str(i) + "*t]] * ")
    f.write("R[C[i + " + str(i) + "*t]] : ")
    f.write("R[B[i + " + str(i) + "*t]] + ")
    f.write("R[C[i + " + str(i) + "*t]];\n")

  f.write("    if (i==0) { final += R[" + str(sync_chunk_idx[-1] -1 + len(input_chunks))+ "*t]; }\n")
  f.write("    __syncthreads();\n")
  f.write("  }\n")
  f.write("  if (i==0) { A[0]= final;} \n")
  f.write("}\n")


def write_c_for_gpu_cuda_3(file_name, graph, graph_nx, final_node, N_THREADS=64, SHARED= True):
  """
    with multiple threads
  """
  
  assert N_THREADS % 32 == 0

  # preprocess leaf nodes
  curr_id= 0
  input_nodes= []
  A= []  # list that stores the values of the input
  for node, obj in list(graph.items()):
    if obj.is_leaf():
      obj.curr_val= random.uniform(0.5,1)
      A.append(obj.curr_val)
      curr_id += 1
      input_nodes.append(node)
  
  # Make length of A multiple of threads
  for i in range(N_THREADS - (len(A)%N_THREADS)):
    A.append(50000.0)
  assert len(A)% N_THREADS == 0

  input_chunks= [input_nodes[i:i+N_THREADS] for i in range(0, len(input_nodes), N_THREADS)]
  
  assign_reverse_level_graph(graph, graph_nx)
  map_lvl_to_nodes= defaultdict(list)
  for node, obj in list(graph.items()):
    map_lvl_to_nodes[obj.reverse_level].append(node)
  
  # delete leaf nodes
  del map_lvl_to_nodes[0]

  # Create list of chunks
  full_chunk_list=[]
  sync_chunk_idx= []
  ac_chunk_idx= 0
  for lvl in sorted(list(map_lvl_to_nodes.keys())):
    nodes= map_lvl_to_nodes[lvl]
    # Break list into node_chunks of size N_THREADS
    node_chunks= [nodes[i:i+N_THREADS] for i in range(0, len(nodes), N_THREADS)]
    for chunk in node_chunks:
      full_chunk_list.append(chunk)
      ac_chunk_idx += 1
    sync_chunk_idx.append(ac_chunk_idx)
  
  full_chunk_list, input_chunks = avoid_gpu_bank_conflicts(graph, full_chunk_list, input_chunks, N_THREADS)
  A=[]
  for chunk in input_chunks:
    A_chunk= []
    for node in chunk:
      if node == None:
        A_chunk.append(-1.0)
      else:
        A_chunk.append(graph[node].curr_val)
    assert len(A_chunk) == N_THREADS
    A += A_chunk

  map_node_to_id= {node: chunk_idx*N_THREADS + node_idx for chunk_idx, chunk in enumerate(input_chunks) for node_idx, node in enumerate(chunk) if node != None}

  # Assing chunks to pos in shared mem, 
  # by finding when nodes become fully consumed
  map_node_to_chunk= {node: chunk_idx for chunk_idx, node_ls in enumerate(full_chunk_list) for node in node_ls if node != None}
  par_set_d= {node:set(obj.parent_key_list) for node, obj in list(graph.items()) if not obj.is_leaf()}
  free_pos= set(range(12*1024/N_THREADS)) # depends on GPU SM shared_mem size
  consumed_ls= defaultdict(set) #key: pos , val: non-fully consumed nodes
  map_chunk_to_pos= {} # key: chunk idx, val: pos it has to be stored
  base_pos= len(input_chunks)  

  for chunk_idx, chunk in enumerate(full_chunk_list):
    pos= min(free_pos)
    free_pos.remove(pos)
    map_chunk_to_pos[chunk_idx]= pos
    consumed_ls[pos]= set(chunk) - set([None])

    for node_idx, node in enumerate(chunk):
      if node == None:
        continue

      map_node_to_id[node]= (base_pos + pos) * N_THREADS + node_idx
      for child in graph[node].child_key_list:
        if not graph[child].is_leaf():
          par_set_d[child].remove(node) # Remove current node from list of parents
          if len(par_set_d[child]) == 0: # All parents computed, this nodes is fully consumed
            del par_set_d[child]
            ch_chunk= map_node_to_chunk[child]
            ch_pos= map_chunk_to_pos[ch_chunk]
            consumed_ls[ch_pos].remove(child)
            
            if len(consumed_ls[ch_pos]) == 0: # All nodes at this pos are fully consumed
              free_pos.add(ch_pos)
              del consumed_ls[ch_pos]
  
  # Sanity
  assert len(consumed_ls) == 1 # Only 1 consumed pos for the final output
  
  # Create B,C, and Op list
  B=[]
  C=[]
  Op=[]
  for chunk in full_chunk_list:
    B.append([])
    C.append([])
    Op.append([])
    for node_idx, node in enumerate(chunk):
      if node == None:
        B[-1].append(node_idx)
        C[-1].append(node_idx)
        Op[-1].append('sum')
      else:
        obj= graph[node]
        child_0= obj.child_key_list[0]
        child_1= obj.child_key_list[1]

        B[-1].append(map_node_to_id[child_0])
        C[-1].append(map_node_to_id[child_1])

        if obj.is_sum():
          Op[-1].append('sum')
        elif obj.is_prod():
          Op[-1].append('prod')
        else:
          assert 0
      
  B_ls= []
  C_ls= []
  Op_ls= []
  for chunk_idx, B_indices in enumerate(B):
    C_indices= C[chunk_idx]
    Op_indices= Op[chunk_idx]
    for idx in range(N_THREADS):
      if idx < len(B_indices):
        B_idx= B_indices[idx]
        C_idx= C_indices[idx]
        Op_idx= Op_indices[idx]
      else:
        B_idx= idx  # To avoid bank conflict
        C_idx= idx
        Op_idx= 'sum'
      
      B_ls.append(B_idx)
      C_ls.append(C_idx)

      if Op_idx == 'sum':
        Op_ls.append(0)
      elif Op_idx == 'prod':
        Op_ls.append(1)
      else:
        assert 0
  
  # Golden ref
  print("Noraml:", ac_eval.ac_eval(graph, final_node))
#  print [{map_node_to_id[node]: obj.curr_val} for node,obj in graph.items() if not obj.is_leaf()]

  # Print to cuda file
  f= open(file_name, 'w+') 
  # Initialize constant lists, i.e. inputs and indices
  f.write("float h_A[]= {\n")
  f.write(str(A)[1:-1])
  f.write("};\n")

  f.write("int h_B[]= {\n")
  f.write(str(B_ls)[1:-1])
  f.write("};\n")

  f.write("int h_C[]= {\n")
  f.write(str(C_ls)[1:-1])
  f.write("};\n")

  f.write("bool h_Op[]= {\n")
  f.write(str(Op_ls)[1:-1])
  f.write("};\n")
  
  f.write("#define THREADS_PER_BLOCK " + str(N_THREADS) + '\n')
  f.write("#define BLOCKS_PER_GRID " + str(1) + '\n')
  f.write("#define SIZE_OF_IN " + str(len(A)) + '\n')
  f.write("#define SIZE_OF_AC " + str(len(B_ls)) + '\n')


  # Actual kernel
  f.write("__device__ void\n")
  
  if SHARED:
    f.write("ac(float *A, const int *B, const int *C, const bool *Op, int n_iter) { \n")
  else:
    f.write("ac(float *A, const int *B, const int *C, const bool *Op, float *R, int n_iter) { \n")

  f.write("  int i= blockDim.x * blockIdx.x + threadIdx.x;\n")

  R_size= max(map_chunk_to_pos.values()) + base_pos + 1

  if SHARED:
    f.write("  __shared__  float R[" + str(R_size) + "*THREADS_PER_BLOCK];\n")
  f.write("  const int t= THREADS_PER_BLOCK;\n")
  f.write("  __shared__ float final;\n")
  f.write("  final=0;\n")
  
  # Copy inputs
  for idx, chunk in enumerate(input_chunks):
    f.write("  R[i + " + str(idx)+ "*t] = A[i + " + str(idx)+ "*t];\n")
  f.write('  __syncthreads();\n')
    
  base_pos= len(input_chunks)  
  # Actual compute
  f.write('  for (int iter=0; iter< n_iter; iter++) {\n')
  for i in range(len(full_chunk_list)):
    if i in sync_chunk_idx:
      f.write('    __syncthreads();\n')

    
#    j= i + len(input_chunks)
    j= map_chunk_to_pos[i] + base_pos

    f.write("    R[i + " + str(j)+ "*t] = Op[i + " + str(i)+ "*t] ? ")
    f.write("R[B[i + " + str(i) + "*t]] * ")
    f.write("R[C[i + " + str(i) + "*t]] : ")
    f.write("R[B[i + " + str(i) + "*t]] + ")
    f.write("R[C[i + " + str(i) + "*t]];\n")
  
  final_pos= map_chunk_to_pos[map_node_to_chunk[final_node]] + base_pos

  f.write("    __syncthreads();\n")
  f.write("    if (i==0) { final += R[" + str(map_node_to_id[final_node]) + "]; }\n")
  f.write("    __syncthreads();\n")
  f.write("  }\n")
  f.write("  if (i==0) { A[0]= final;} \n")
  f.write("}\n")

def write_c_for_gpu_cuda_4(file_name, graph, graph_nx, final_node):
  """
    Single thread
  """
  
  f= open(file_name, 'w+') 

  input_ls=[node for node, obj in list(graph.items()) if obj.is_leaf()]
  map_node_to_id={node: idx for idx, node in enumerate(input_ls)}
  for node in input_ls:
    graph[node].curr_val= random.uniform(0.5, 1)
  
  print("Normal:", ac_eval.ac_eval(graph, final_node))

  A= [graph[node].curr_val for node in input_ls] 
  f.write("float h_A[]= {\n")
  f.write(str(A)[1:-1])
  f.write("};\n")

  # Children before parent
  topological_list= nx.algorithms.dag.topological_sort(graph_nx)

  topological_list= [node for node in topological_list if not graph[node].is_leaf()]
  
  for idx, node in enumerate(topological_list):
    map_node_to_id[node]= idx

  f.write("#define THREADS_PER_BLOCK " + str(1) + '\n')
  f.write("#define BLOCKS_PER_GRID " + str(1) + '\n')
  f.write("#define SIZE_OF_IN " + str(len(A)) + '\n')

  # Actual kernel
  f.write("__device__ void\n")
  f.write("ac(float *A, int n_iter) { \n")
  
  node_str= ["val_" + str(map_node_to_id[node]) for node in topological_list]
  f.write("  float ")
  seperator= ','
  f.write(seperator.join(node_str))
  f.write(';\n')
  
  # Copy inputs to shared mem
#  f.write('  __shared__ float A[SIZE_OF_IN];\n')
#  f.write('  for (int j=0; j < n_iter; j++) {\n')
#  f.write('    A[j]=x_A[j];\n')
#  f.write('  }\n')  

  f.write('  float res=0;\n')

  f.write('  for (int j=0; j < n_iter; j++) {\n')
  for node in topological_list:
    obj= graph[node]
    f.write('    val_' + str(map_node_to_id[node]) + ' = ')
    
    ch_0= obj.child_key_list[0]
    ch_id= str(map_node_to_id[ch_0])
    if graph[ch_0].is_leaf():
      f.write('A[' + ch_id + '] ')    
    else:
      f.write('val_' + ch_id + ' ')    

    if obj.is_sum():
      f.write('+ ')
    elif obj.is_prod():
      f.write('* ')
    else:
      assert 0

    ch_1= obj.child_key_list[1]
    ch_id= str(map_node_to_id[ch_1])
    if graph[ch_1].is_leaf():
      f.write('A[' + ch_id + '] ')    
    else:
      f.write('val_' + ch_id + ' ')    
    f.write(";\n")
  
  f.write('    res += val_' + str(map_node_to_id[final_node]) + ';\n')
  f.write('    __syncthreads();\n')
  f.write("  }\n")

  f.write('  A[0] = res;\n')
  f.write("}\n")



def avoid_gpu_bank_conflicts(graph, full_chunk_list, input_chunks, N_THREADS):
  """
    avoids shared memory bank conflicts 
  """

  N_BANKS= 32 # shared memory in SM has 32 banks
  
  # break everything into chunk of size N_BANKS
  new_input_chunks= []
  for chunk in input_chunks:
    new_chunks= [chunk[i:i+N_BANKS] for i in range(0, len(chunk), N_BANKS)]
    new_input_chunks += new_chunks    
  
  new_full_chunk_list= []
  for chunk in full_chunk_list:
    new_chunks= [chunk[i:i+N_BANKS] for i in range(0, len(chunk), N_BANKS)]
    new_full_chunk_list += new_chunks    
  
  # Create IO graph
  IO_graph= nx.Graph()
  
  # Add nodes
  for chunk in new_full_chunk_list:
    assert len(chunk) <= N_BANKS
    IO_graph.add_nodes_from(chunk)
  for chunk in new_input_chunks:
    assert len(chunk) <= N_BANKS
    IO_graph.add_nodes_from(chunk)
  
  # Add edges
  for chunk in new_input_chunks:
    edges_obj= itertools.combinations(chunk ,2)
    IO_graph.add_edges_from(edges_obj)
  
  for chunk in new_full_chunk_list:
    #output conflict
    edges_obj= itertools.combinations(chunk ,2)
    IO_graph.add_edges_from(edges_obj)

    #input conflict
    input_0_ls= []
    input_1_ls= []
    for node in chunk:
      input_0_ls.append(graph[node].child_key_list[0])
      input_1_ls.append(graph[node].child_key_list[1])
    edges_obj= itertools.combinations(input_0_ls ,2)
    IO_graph.add_edges_from(edges_obj)
    edges_obj= itertools.combinations(input_1_ls ,2)
    IO_graph.add_edges_from(edges_obj)
  
  # Allocate bank
  avail_banks= {node: set(range(N_BANKS)) for node in IO_graph.nodes()}
  bank_d= {}

  conflict = 0
  while len(avail_banks) != 0:
    curr_node= min(list(avail_banks.keys()), key= lambda x: len(avail_banks[x]))

    if len(avail_banks[curr_node]) != 0:
      picked_bank= random.choice(list(avail_banks[curr_node]))
    else:
      picked_bank= random.choice(list(range(N_BANKS)))
      conflict += 1
    
    bank_d[curr_node] = picked_bank

    # remove banks from neighbors
    for node in IO_graph.neighbors(curr_node):
      if node in avail_banks:
        avail_banks[node] -= set([picked_bank])
    
    del avail_banks[curr_node]
  
  print('conflict:', conflict)

  # modify chunks according to picked banks
  final_input_chunks=[]
  for chunk in input_chunks:
    new_chunk= [None] * N_THREADS
    avail_threads= set(range(N_THREADS))
    for node in chunk:
      bank= bank_d[picked_bank]
      possible_threads= set([i*N_BANKS + bank for i in range(N_THREADS/N_BANKS)])
      possible_threads &= avail_threads

      if len(possible_threads) != 0:
        picked_thread= random.choice(list(possible_threads))  
      else:
        picked_thread= random.choice(list(avail_threads))  
      
      new_chunk[picked_thread]= node
      avail_threads.remove(picked_thread)
    
    final_input_chunks.append(new_chunk)

  final_full_chunk_list=[]
  for chunk in full_chunk_list:
    new_chunk= [None] * N_THREADS
    avail_threads= set(range(N_THREADS))
    for node_idx, node in enumerate(chunk):
      bank= bank_d[picked_bank]
      possible_threads= set([i*N_BANKS + bank for i in range(N_THREADS/N_BANKS)])

      start= (node_idx/N_BANKS) * N_BANKS
      end= ((node_idx/N_BANKS)+1) * N_BANKS
      possible_threads &= set(range(start, end))
      assert len(possible_threads) == 1

      possible_threads &= avail_threads

      if len(possible_threads) != 0:
        picked_thread= random.choice(list(possible_threads))  
      else:
        picked_thread= random.choice(list(avail_threads))  
      
      new_chunk[picked_thread]= node
      avail_threads.remove(picked_thread)
    
    final_full_chunk_list.append(new_chunk)
  
  return final_full_chunk_list, final_input_chunks
    

def write_psdd_dot(graph):
  G= pydot.Dot(graph_type= "digraph", shape= 'circle', overlap='compress', overlap_shrink= True, ratio='compress', splines= 'curved')

  cnt=0
  for node, obj in graph.items():
    parent_len= len(obj.parent_key_list)
    if parent_len > 1 and not obj.is_leaf():
      cnt += 1

    if obj.is_sum():
      node= pydot.Node(str(node), label=str('+'), shape= 'ellipse', style='filled',fillcolor= 'indianred1',\
#          width=0.08, height= 0.05,\
          fontsize=20, penwidth= 3, margin=0, ranksep= 0.5, nodesep= 0.5)
    elif obj.is_prod():
      node= pydot.Node(str(node), label=str('x'), shape= 'ellipse', style='filled',\
#          width=0.08, height= 0.05,\
          fontsize=20,fillcolor= 'palegreen1', penwidth= 3, margin=0, ranksep= 0.5, nodesep= 0.5)
    else:
      node= pydot.Node(str(node), label=str(''), shape= 'none', style='filled',fillcolor= 'white', width=0, height=0, margin=0, penwidth= 0, ranksep= 0.5, nodesep= 0.5)
#    node= pydot.Node(str(node), label= '+',shape= 'circle', style='filled',fillcolor= 'white', penwidth= 3, ranksep= 0.2, nodesep= 0.2)
    G.add_node(node)
      
    for child in obj.child_key_list:
      edge= pydot.Edge(node, child, weight=1.2, color= 'black', splines= 'curved', dir='back')
      G.add_edge(edge)
  
  print("cnt:", cnt)
  G.write_png("./psdd_img.png")
  G.write_dot("./psdd_img.dot")


def write_asm(global_var, operation, operands):
  """ Prints assembly to a file
  operation: name of the operation like ld, st, cp etc.
  operands: Name of the registers, constants etc as a list of strings
  """

  fp= open(global_var.ASM_FILE, 'a+' )
  
  fp.write(operation)

  for opr in operands:
    fp.write(' ')
    fp.write(str(opr))
  
  fp.write('\n')

  fp.close()

class video_maker():
  def __init__(self, vid_path, img_path, *args, **kw ):
    self.vid_path= vid_path
    self.img_path= img_path
    self.IMG_WRITTEN= False
    
    self.vid_obj= None
    self.FRAME_RATE= 5

  def create_a_frame_scheduling(self, BB_graph, score_inputs={}):
    G= pydot.Dot(graph_type= "digraph", shape= 'box')
    
    for bb, obj in list(BB_graph.items()):
      
      if obj.status == obj.COMPUTED:
        COLOR= 'red'
      elif obj.status == obj.INPUTS_IN_VECTOR:
        if score_inputs:
          if score_inputs[bb] == len(obj.in_list_unique):
            COLOR= 'green'
          else:
            COLOR= 'blue'
        else:
          COLOR= 'blue'
      elif obj.status == obj.RECENT_INPUTS:
        COLOR= 'blue'
      elif obj.status == obj.INPUTS_IN_SCALAR:
        if score_inputs:
          if score_inputs[bb] == len(obj.in_list_unique):
            COLOR= 'green'
          else:
            COLOR= 'yellow'
        else:
          COLOR= 'yellow'
      else:
        COLOR= 'white'
      
      node= pydot.Node(str(bb), shape= 'box', style='filled',fillcolor= COLOR, penwidth= 3)
      G.add_node(node)
        
      for child in obj.child_bb_lst:
        edge= pydot.Edge(bb, child)
        G.add_edge(edge)
    
    G.write_png(self.img_path)
    self.IMG_WRITTEN= True
  
  def create_a_frame_ac(self, graph):
    G= pydot.Dot(graph_type= "digraph", shape= 'box', rankdir= 'BT')
    
    COLOR= 'white'
    BORDER_COLOR= 'dodgerblue'

    for key, obj in list(graph.items()):
      node= pydot.Node(str(key), label= ' ', shape= 'circle', style='filled',fillcolor= COLOR, color= BORDER_COLOR ,penwidth= 5)
    
      G.add_node(node)
      
      for child in obj.child_key_list:
        edge= pydot.Edge(child, key, color= 'darkturquoise', penwidth= 3)
        G.add_edge(edge)
    
    G.write_png(self.img_path)
    self.IMG_WRITTEN= True

  def create_vid_obj(self):
    assert self.IMG_WRITTEN, "Create atleast one frame before creating vid object"
      
    frame= cv2.imread(self.img_path)
    height, width, layers = frame.shape
      
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    
    self.vid_obj = cv2.VideoWriter(self.vid_path, fourcc, self.FRAME_RATE, (width,height))
   
  def add_frame(self):
    assert self.IMG_WRITTEN, "Previously created frame already consumed. Create a new frame again before adding it to video"
    assert self.vid_obj is not None, "Create a video object first by calling create_vid_obj()"
    
    frame= cv2.imread(self.img_path)
    self.vid_obj.write(frame)
    
    self.IMG_WRITTEN= False

  def write_vid(self):
    assert self.vid_obj is not None, "Create a video object first by calling create_vid_obj()"
    
    cv2.destroyAllWindows()
    self.vid_obj.release()
    
def plot_graph(G, **kw):
  """
    Plots a networkx graph

    parameters: G (a networkx graph)
    **kw : optional keyword arguments to specifcy attributes like node color etc.
  """

  useful_methods.plot_graph(G)

def plot_graph_nx_graphviz(G, path= "./temp.png"):
  #A= nx.nx_agraph.to_agraph(graph_nx) # convert to a graphviz graph
  #A.layout() # neato layout
  #A.draw() # write postscript in k5.ps with neato layout 
  
  #nx.drawing.nx_pydot.write_dot(G, "./LOG/temp.dot")
  dot_graph= nx.drawing.nx_pydot.to_pydot(G)
  dot_graph.set('rankdir', 'BT')
  dot_graph.write_png(path) 
  
    
def plot_dot_example():
  G = nx.Graph()

  G.add_node(1,color='red',style='filled',fillcolor='blue',shape='square')
  G.add_node(2,color='blue',style='filled')
  G.add_edge(1,2,color='green')
  G.nodes[2]['shape']='circle'
  G.nodes[2]['fillcolor']='red'

def show_image(path= "./temp.png"):
  img = mpimg.imread(path)
  imgplot = plt.imshow(img)
  plt.show()
  
def show_matrix(A):
  plt.spy(A, marker= "s", markersize= 3, markerfacecolor='black', markeredgecolor='black')
  plt.show()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_workload_balancing_info(name_ls, n_threads_ls, plot_d):
  for name in name_ls:
    for threads in n_threads_ls:
      fig_dims = (2, 2)
      fig, ax = plt.subplots(figsize=fig_dims) 

      list_of_partitions, node_w, config_obj, graph_nx, graph = plot_d[(name,threads)]

      N_PE= len(list_of_partitions)
      n_layers= len(list_of_partitions[0])

      x= list(range(n_layers))
      x= [a+1 for a in x]

      y_array= [[sum([node_w[n] for n in list_of_partitions[pe][l]]) for l in range(n_layers)] for pe in range(N_PE)]

      # replace 0 with 1 for log scale
      y_array = [[y_array[pe][l] if y_array[pe][l] > 0 else 1 for l in range(n_layers)] for pe in range(N_PE)]

      print(y_array)

      for pe in range(N_PE):
        ax.plot(x, y_array[pe], 'o', color='C0')

      ax.plot(x, [1 for _ in x], 'o', alpha=0)

      ax.set_yscale('log')
      plt.xticks(x)

      plt.tight_layout()
      # plt.show()

      path= './no_backup/' + name.replace('/', '_') + f'_workload_balance_{threads}.png'
      plt.savefig(path, dpi= 300, format='png')

def plot_workload_balancing_info_multiple(plot_d):
  dims= int(sqrt(len(plot_d)))
  fig_dims = (16, 16)
  # fig, ax = plt.subplots(dims, dims, figsize=fig_dims) 
  fig, ax = plt.subplots(dims, dims) 

  name_ls= list(plot_d.keys())

  for d_x in range(dims):
    for d_y in range(dims):
      print(d_x, d_y)

      name= name_ls[d_x*dims + d_y]
      list_of_partitions, node_w= plot_d[name]

      N_PE= len(list_of_partitions)
      n_layers= len(list_of_partitions[0])

      x= list(range(n_layers))
      x= [int(a) for a in x]
      
      y_array= [[sum([node_w[n] for n in list_of_partitions[pe][l]]) for l in range(n_layers)] for pe in range(N_PE)]

      # replace 0 with 1 for log scale
      y_array = [[y_array[pe][l] if y_array[pe][l] > 0 else 1 for l in range(n_layers)] for pe in range(N_PE)]

      print(y_array)

      for pe in range(N_PE):
        ax[d_x, d_y].plot(x, y_array[pe], 'o')
        ax[d_x, d_y].set_yscale('log')
        ax[d_x, d_y].set_title(name)
  
  # plt.xticks(x)
  
  # for ax in axs.flat:
  #     ax.set(xlabel='x-label', ylabel='y-label')

  # # Hide x labels and tick labels for top plots and y ticks for right plots.
  # for ax in axs.flat:
  #     ax.label_outer()

  plt.tight_layout()
  plt.show()

