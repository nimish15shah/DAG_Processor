
from collections import defaultdict
import itertools
import networkx as nx
from . import useful_methods

from . import reporting_tools

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def relabel_nodes_with_contiguous_numbers(graph_nx, start= 0):
  """
    Creates a shallow copy
  """
  mapping= {n : (idx + start) for idx, n in enumerate(list(graph_nx.nodes()))}

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping
  
def replace_leaves_with_x(graph_nx_original, x):
  
  # Copy
  ## NOTE: only a shallow copy is created
  # Ok for graph structure, but be careful if attributes are there
  graph_nx= graph_nx_original.copy(as_view= False)

  # remove all leaves
  leaves= useful_methods.get_leaves(graph_nx)
  graph_nx.remove_nodes_from(leaves)

  # Rename rest of the nodes from 1 to max_cnt
  graph_nx, _ = relabel_nodes_with_contiguous_numbers(graph_nx, start = x + 1)

  # Add 0 as a common leaf to all the nodes
  leaves= useful_methods.get_leaves(graph_nx)
  new_edges= [(x, l) for l in leaves]
  graph_nx.add_edges_from(new_edges)
  
  return graph_nx
    
def global_partition(fpath, graph_nx_original, D):
  # replace leaves and a single node
  # Also, sort nodes number in contiguos fashion starting from 0
  graph_nx= replace_leaves_with_x(graph_nx_original, x = 0)

  nodes= list(graph_nx.nodes())
  min_node= min(nodes)
  max_node= max(nodes)

  print (min_node)
  print (max_node)
  
  assert min_node == 0
  assert set(nodes) == set(list(range(max_node+1))), (set(nodes) - set(list(range(max_node + 1))))
  successors_str= "successors= ["
  for n in range(max_node + 1):
    successors_str += "{"
    for succ in graph_nx.successors(n):
      successors_str += str(succ + 1)
      successors_str += ','

    if len(list(graph_nx.successors(n))) != 0:
      successors_str = successors_str[:-1]
      successors_str += "}, "
    else:
      successors_str += "}];\n"

  predecessors_str= "predecessors= ["
  for n in range(max_node + 1):
    predecessors_str += "{"
    for succ in graph_nx.predecessors(n):
      predecessors_str += str(succ + 1)
      predecessors_str += ','

    if len(list(graph_nx.predecessors(n))) != 0:
      predecessors_str = predecessors_str[:-1]

    predecessors_str += "},"

  predecessors_str = predecessors_str[:-1]
  predecessors_str += "];\n"

#  V_str= "V = " + str(max_node + 1) + ";\nTMAX= " + str(max_node//18 + 1) + ";\n"
  misc_str= ""
  misc_str += "V = " + str(max_node + 1) + ";\n"
  print (fpath)
  
  e_str = "E= " + str(graph_nx.number_of_edges()) + ";\n"
  e_str+= "edges= ["
  edge_list= list(graph_nx.edges())
  map_n_to_out_edges= defaultdict(set)
  map_n_to_in_edges= defaultdict(set)
  for edge_idx, n_tup in enumerate(edge_list):
    n1= n_tup[0]
    n2= n_tup[1]
    map_n_to_out_edges[n1].add(edge_idx)
    map_n_to_in_edges[n2].add(edge_idx)
    e_str += "|" + str(n1 + 1) + "," + str(n2+1) 
  e_str += "|];\n"

  out_edges_str= "out_edges= ["
  for n in range(max_node + 1):
    out_edges_str += "{"
    for edge_idx in map_n_to_out_edges[n]:
      out_edges_str += str(edge_idx + 1)
      out_edges_str += ','

    if len(map_n_to_out_edges[n]) != 0:
      out_edges_str = out_edges_str[:-1]

    out_edges_str += "},"
  out_edges_str = out_edges_str[:-1]
  out_edges_str += "];\n"


  in_edges_str= "in_edges= ["
  for n in range(max_node + 1):
    in_edges_str += "{"
    for edge_idx in map_n_to_in_edges[n]:
      in_edges_str += str(edge_idx + 1)
      in_edges_str += ','

    if len(map_n_to_in_edges[n]) != 0:
      in_edges_str = in_edges_str[:-1]

    in_edges_str += "},"
  in_edges_str = in_edges_str[:-1]
  in_edges_str += "];\n"


  # max slack allowed in T of successor nodes
  map_v_to_reverse_lvl = useful_methods.compute_reverse_lvl(graph_nx)
  slack_dict= {}
  topological_list= nx.algorithms.dag.topological_sort(graph_nx)
  graph_reverse= graph_nx.reverse()
  for v in topological_list:
    curr_lvl= map_v_to_reverse_lvl[v]
    successors= list(graph_nx.successors(v))
    predecessors= list(graph_nx.predecessors(v))
    max_slack = 1

    for s in successors:
      slack= (map_v_to_reverse_lvl[s] - curr_lvl + 1*D) // D 
      if max_slack < slack:
        max_slack = slack
    
    bfs_predecessors = dict(nx.algorithms.traversal.breadth_first_search.bfs_successors(graph_reverse, v, D))
    bfs_predecessors = list(bfs_predecessors.values())
    bfs_predecessors_ls= []
    for b in bfs_predecessors:
      bfs_predecessors_ls += b
    bfs_predecessors = set(bfs_predecessors_ls)

    for p in bfs_predecessors:
      if slack_dict[p] < max_slack:
        slack_dict[p] = max_slack

    slack_dict[v] = max_slack
  
  misc_str += "SLACK_DICT = ["
  for n in range(max_node + 1):
    misc_str += str(slack_dict[n]) + ","
  misc_str= misc_str[:-1]
  misc_str += "];\n"
#  misc_str += "MAX_SLACK= " + str(max_slack) + ";\n"

  # TMAX
  TMAX = len(nx.algorithms.dag.dag_longest_path(graph_nx))
  TMAX //= D
  TMAX += 4 # because optimal solution is generally not at lowest possible TMAX
  misc_str += "TMAX =" + str(TMAX) + ";\n"
  
  with open(fpath, 'w+') as fp:
    fp.write(misc_str)
    fp.write(successors_str)
    fp.write(predecessors_str)
    fp.write(e_str)
    fp.write(out_edges_str)
    fp.write(in_edges_str)


def custom_dot_file(graph_nx):
#  T = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 2, 5, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 3, 2, 2, 3, 4, 4, 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 5, 5, 5, 5, 5]
#  T = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]

  T= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 1, 4, 4, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

  f= "visualize_2.dot"

  f= open(f, 'w+')
  f.write('digraph G {\n')
  
  colors= ['red', 'green', 'blue', 'yellow', 'pink', 'white']
  for node, t in enumerate(T):
    f.write( str(node) + ' [shape=ellipse, label=' + str(node) + ' ,style=\"filled\", fillcolor=\"')
    f.write(''+ colors[t] + '\"]\n')

    for parent in graph_nx.successors(node):
      f.write(str(parent) + ' ->' + str(node) + ' [dir=none]\n')


def minizinc_decompose(fpath, graph_nx):
  nodes= list(graph_nx.nodes())
  min_node= min(nodes)
  max_node= max(nodes)

  print (min_node)
  print (max_node)
  
  assert min_node == 0

  assert set(nodes) == set(list(range(max_node+1))), (set(nodes) - set(list(range(max_node + 1))))
  
  first= []
  second= []
  successors_str= "successors= ["
  dummy_node= max_node + 2
  for n in range(max_node + 1):
    pred= sorted(list(graph_nx.predecessors(n)), reverse= True)
    if len(pred) != 0:
      first.append(pred[0] + 1)
      second.append(pred[1] + 1)
    else:
      first.append(dummy_node)
      second.append(dummy_node)

    successors_str += "{"
    for succ in graph_nx.successors(n):
      successors_str += str(succ + 1)
      successors_str += ','

    if len(list(graph_nx.successors(n))) != 0:
      successors_str = successors_str[:-1]
    else: # final node
      successors_str += str(dummy_node)

    successors_str += "}, "
  
  first.append(dummy_node)
  second.append(dummy_node)

  successors_str += "{"+ str(dummy_node) +"} ];\n"
  
  
  assert len(first) == dummy_node , len(first)
  assert len(second) == dummy_node , len(second)

  first_str= "first= ["
  for f in first:
    first_str += str(f) + ',' 
  first_str = first_str[:-1]
  first_str += '];\n'

  second_str= "second= ["
  for f in second:
    second_str += str(f) + ',' 
  second_str = second_str[:-1]
  second_str += '];\n'
  
  V_str= "V = " + str(dummy_node) + ";\n"
  print (fpath)
  
  dfs_G= graph_nx
  dfs_G= dfs_G.reverse()
  dfs_topological_list= list(nx.algorithms.traversal.depth_first_search.dfs_postorder_nodes(dfs_G, max_node))
  map_v_to_dfs_idx= {v: idx for idx,v in enumerate(dfs_topological_list)}

  dfs_lvl= "dfs_lvl = array1d(0..V, [0, "
  for n in range(max_node + 1):
    dfs_lvl += str(map_v_to_dfs_idx[n]) + ','
  dfs_lvl += '0]);\n'
    
  with open(fpath, 'w+') as fp:
    fp.write(V_str)
    fp.write(first_str)
    fp.write(second_str)
    fp.write(successors_str)
    fp.write(dfs_lvl)

def minizinc_systolic(fpath, original_graph_nx, max_size= 500):
  logger.info(f"Total number of nodes: {len(original_graph_nx)}")
  graph_nx= original_graph_nx.copy(as_view= False)

  largest_cc = max(nx.weakly_connected_components(graph_nx), key= len)
  graph_nx= graph_nx.subgraph(largest_cc)
  
  if len(graph_nx) > max_size:
    topo_list= useful_methods.dfs_topological_sort(graph_nx)
    topo_list= topo_list[: max_size]

    # ensure all the predecessors are still in the graph
    subg= set([])
    for n in topo_list:
      subg.add(n)
      pred= set(list(graph_nx.predecessors(n)))
      assert len(pred) == 2 or len(pred) == 0
      subg |= pred

    graph_nx = graph_nx.subgraph(subg)
  
  logger.info(f"Number of nodes considered: {len(graph_nx)}")

  out_str= f'V= {len(graph_nx)};\n'

  leaves= useful_methods.get_leaves(graph_nx)
  V_internal_last = len(graph_nx) - len(leaves)
  out_str += f'V_internal_last= {V_internal_last};\n'

  V_leaf_start =  V_internal_last + 1
  out_str += f'V_leaf_start= {V_leaf_start};\n'

  graph_nx_internal= graph_nx.copy(as_view= False)
  graph_nx_internal.remove_nodes_from(leaves)

  graph_nx_internal, mapping = relabel_nodes_with_contiguous_numbers(graph_nx_internal, 1)
  reverse_mapping= {new_n: n for n, new_n in mapping.items()}  
  assert V_leaf_start not in graph_nx_internal

  iterator= itertools.count(V_leaf_start)
  for l in leaves:
    assert l not in mapping
    mapping [l] = next(iterator)

  assert next(iterator) == len(graph_nx) + 1

  pred_0= []
  pred_1= []
  reverse_mapping= {new_n: n for n, new_n in mapping.items()}  
  for new_n in range(1, V_internal_last+1):
    n= reverse_mapping[new_n]
    pred= list(graph_nx.predecessors(n))
    assert len(pred) == 2

    pred= [mapping[p] for p in pred]
    for p in pred:
      assert p != 0
    
    pred_0.append( pred[0] )
    pred_1.append( pred[1] )

  out_str += 'pred_0= ['
  for p in pred_0:
    out_str += f'{p},'
  out_str = out_str[: -1] # remove last comma
  out_str += "];\n"

  out_str += 'pred_1= ['
  for p in pred_1:
    out_str += f'{p},'
  out_str = out_str[: -1] # remove last comma
  out_str += "];\n"

  # interesting_nodes= [10, 22, 8, 65, 0, 0, 0, 0, 20, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 5, 6, 0, 0, 0, 46, 42, 59, 0, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0, 0, 0, 61, 0, 0, 57, 0, 0, 0, 0, 0, 69]
  # interesting_nodes = reversed([reverse_mapping[i] if (i !=0)  else 0 for i in interesting_nodes ])
  # for idx, i in enumerate(interesting_nodes):
  #   if idx % 8 == 0:
  #     print("")
  #   print(str(i).zfill(3), end= "   ")
  # exit(1)

  with open(fpath, 'w+') as fp:
    fp.write(out_str)

def minizinc_pe_bank_allocate(fpath, IO_graph, n_mem_banks):
  
  n_str = "N= " + str(IO_graph.number_of_nodes()) + ";\n"
  n_str += "NODES= {"
  for n in IO_graph.nodes():
    n_str += "n" + str(n) + ","
  n_str = n_str[:-1]
  n_str += "};\n"

  e_str = "E= " + str(IO_graph.number_of_edges()) + ";\n"
  e_str+= "edges= ["
  for n1, n2 in IO_graph.edges():
    e_str += "|n" + str(n1) + ",n" + str(n2) 
  e_str += "|];\n"

  reg_str= "MAX_R= " + str(n_mem_banks + 1) + ";\n"
#  reg_str= "REG = {"
#  for r in range(n_mem_banks):
#    reg_str += "r" + str(r) + ","
#  reg_str += "r_dummy};\n"

  with open(fpath, 'w+') as fp:
    fp.write(n_str)
    fp.write(e_str)
    fp.write(reg_str)
  
