
import random
import logging as log

from . import ac_eval
from . import common_classes
from . import files_parser
from . import psdd

# from .verif import pru_async as verif_pru_async
from . import pru_sync  as verif_pru_sync

def init_leaf_val(graph, mode="all_1s"):
  """
    Initializes values of input nodes
  """
  
  assert mode in ['random', 'all_1s']

  for node, obj in list(graph.items()):
    if obj.is_leaf():
      if mode == 'random':
        val= random.uniform(0.2, 1.0)
      elif mode == 'all_1s':
        val= 1
      else:
        assert 0
      
      obj.curr_val= val

def pru_sync(graph, prefix, hw_details, instr_ls, map_param_to_addr, map_output_to_addr, write_files):
  init_leaf_val(graph, mode="random")
  map_addr_to_val = verif_pru_sync.main(graph, prefix, hw_details, instr_ls, map_param_to_addr, map_output_to_addr, write_files)
  return map_addr_to_val

def pru_async(global_var, graph, graph_nx, final_output_nodes, status_dict, list_of_schedules, config_obj):
#  init_leaf_val(graph, mode="random")

  if config_obj.targe_app == config_obj.target_app_enum.SPN:
    print('instantiating SPN leafs')
    dataset= files_parser.read_dataset(global_var, config_obj.name, 'test')
    psdd.instanciate_literals(graph, dataset[0])
  
  simulated_res_dict= verif_pru_async.simulate_instr_async(graph, global_var, final_output_nodes, status_dict, list_of_schedules, config_obj)
  
  # log.warning("Checker is disbaled here. Will not check if HW simulations match ac_eval")
  checker(graph, graph_nx, final_output_nodes, simulated_res_dict)

  verif_pru_async.main(global_var, graph, status_dict, list_of_schedules, config_obj)

  return simulated_res_dict

def checker(graph, graph_nx, final_output_nodes, simulated_res_dict):
  # print('Golden:', ac_eval.ac_eval_non_recurse(graph, graph_nx))
  total_error= []

  for n in list(final_output_nodes)[:10]:
    print (f'{n}, golden: {graph[n].curr_val}, simulated: {simulated_res_dict[n]}')
    err = abs(graph[n].curr_val - simulated_res_dict[n])
    total_error.append(err)
  
  print(f'total_error: {sum(total_error)}')
  total_val= sum(list(simulated_res_dict.values()))
  if total_val !=0:
    print(f'normalized error: {sum(total_error)/total_val}')
  else:
    print('normalized error: Div by zero')
