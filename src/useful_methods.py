import inspect
import networkx as nx
import itertools
from collections import defaultdict
from typing import Mapping, MutableMapping, MutableSequence, Sequence, Iterable, List, Set, Dict
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    black   = '\u001b[30m'
    red     = '\u001b[31m'
    green   = '\u001b[32m'
    yellow  = '\u001b[33m'
    blue    = '\u001b[34m'
    magenta = '\u001b[35m'
    cyan    = '\u001b[36m'
    white   = '\u001b[37m'

    bblack  = '\u001b[30;1m'
    bred    = '\u001b[31;1m'
    bgreen  = '\u001b[32;1m'
    byellow = '\u001b[33;1m'
    bblue   = '\u001b[34;1m'
    bmagenta= '\u001b[35;1m'
    bcyan   = '\u001b[36;1m'
    bwhite  = '\u001b[37;1m'


    reset= '\u001b[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

        self.black   ='' 
        self.red     ='' 
        self.green   ='' 
        self.yellow  ='' 
        self.blue    ='' 
        self.magenta ='' 
        self.cyan    ='' 
        self.white   ='' 
        
        self.bblack   ='' 
        self.bred     ='' 
        self.bgreen   ='' 
        self.byellow  ='' 
        self.bblue    ='' 
        self.bmagenta ='' 
        self.bcyan    ='' 
        self.bwhite   ='' 
        

    def color(self, name= 'reset'):
      if name == 'red':
        return self.red 
      
      if name == 'black':
        return self.black

      if name == 'green':
        return self.green 

      if name == 'yellow':
        return self.yellow

      if name == 'blue':
        return self.blue 
      
      if name == 'magenta':
        return self.magenta 

      if name == 'cyan':
        return self.cyan

      if name == 'white':
        return self.white

      if name == 'bred':
        return self.bred 
      
      if name == 'bblack':
        return self.bblack

      if name == 'bgreen':
        return self.bgreen 

      if name == 'byellow':
        return self.byellow

      if name == 'bblue':
        return self.bblue 
      
      if name == 'bmagenta':
        return self.bmagenta 

      if name == 'bcyan':
        return self.bcyan

      if name == 'bwhite':
        return self.bwhite
  

      if name == 'reset':
        return self.reset

      return self.reset

def indices(a, func):
  """ Returns list of indices where func evals to True 
  @param a: list to be checked
  @param func: Test to be performed
  
  @example: indices(a, lambda x: x==1) returns all the indices of the list that contains value 1
  """ 
  return [i for (i, val) in enumerate(a) if func(val)]

def printlog(msg, color='reset'):
  callerframerecord = inspect.stack()[1]    # 0 represents this line
                                            # 1 represents line at caller
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  filename= info.filename
  filename= filename.split('/')[-1]
  print(filename, ':', end=' ')                      # __FILE__     -> Test.py
  #print info.function, '::',                      # __FUNCTION__ -> Main
  print(info.lineno, ':', end=' ')                       # __LINE__     -> 13
  
  color_obj= bcolors()
  print(color_obj.color(color), end=' ')
  print(msg, end=' ')
  print(color_obj.color('reset'))

def printcol(msg, color= 'reset'):
  color_obj= bcolors()
  print(color_obj.color(color), end=' ')
  print(msg, end=' ')
  print(color_obj.color('reset'))
  

def create_nx_graph_from_node_graph(graph):
  """
    Creates an equivalent of self.graph (the default AC graph) in a networkx format.
    This is in an attempt to move everything to networkx graphs
  """

  G= nx.DiGraph()
  for node,obj in list(graph.items()):
    for parent in obj.parent_key_list:
      G.add_edge(node, parent)
  
  return G  

def check_if_only_one_root(graph):
  if isinstance(graph, dict):
    head_ls= [node for node, obj in list(graph.items()) if len(obj.parent_key_list) == 0]
  else: #graph_nx
    head_ls= [node for node in graph.nodes() if len(list(graph.successors(node))) == 0]

  assert len(head_ls) == 1, [head_ls]
  return head_ls[0]

def get_head_ls(graph):
  if isinstance(graph, dict):
    head_ls= [node for node, obj in list(graph.items()) if len(obj.parent_key_list) == 0]
  else: #graph_nx
    head_ls= [node for node in graph.nodes() if len(list(graph.successors(node))) == 0]

  return head_ls

def clog2(num):
  """
    clog2(2) = 1
    clog2(32) = 5
  """
  assert num > 0
  return len(bin(num-1)) - 2

def isPowerOfTwo(n): 
    if (n == 0): 
        return False
    while (n != 1): 
            if (n % 2 != 0): 
                return False
            n = n // 2
    return True

def format_hex(num, L):
  format_str= '{:0' + str(L//4) + 'x}'
  return format_str.format(num)

def get_leaves(graph_nx):
  return set([n for n in graph_nx.nodes() if len(list(graph_nx.predecessors(n))) == 0])

def get_non_leaves(graph_nx):
  return set([n for n in graph_nx.nodes() if len(list(graph_nx.predecessors(n))) != 0])

def get_non_leaves_subgraph(graph_nx):
  non_leaves= get_non_leaves(graph_nx)

  return graph_nx.subgraph(non_leaves)

def neither_ancestors_nor_descendents(graph_nx, node):
  ancestors= nx.algorithms.dag.ancestors(graph_nx, node)
  descendants= nx.algorithms.dag.descendants(graph_nx, node)
  
  return_nodes = set(graph_nx.nodes) - set([node])
  return_nodes -= set(ancestors)
  return_nodes -= set(descendants)
  
  return return_nodes

def compute_lvl(graph_nx):
  """ all leaves are NOT zero, Only the leaves of the longest path are 0
  """
  topological_list= list(nx.algorithms.dag.topological_sort(graph_nx))

  map_v_to_lvl= {}
  
  # head= check_if_only_one_root(graph_nx)
  # map_v_to_lvl[head]= 1000 # very high number
  
  # topological_list.remove(head)

  for node in reversed(list(topological_list)):
    succ= list(graph_nx.successors(node))
    if len(succ) == 0:
      curr_lvl= 10000000 # very high number
    else:
      curr_lvl = min([map_v_to_lvl[s] for s in graph_nx.successors(node)]) - 1    
    map_v_to_lvl[node]= curr_lvl
  
  min_lvl= min(list(map_v_to_lvl.values()))
  
  assert min_lvl >= 0, "very high number of head is not high enough for this graph"

  map_v_to_lvl= {k: lvl- min_lvl for k, lvl in map_v_to_lvl.items()}

  assert min(list(map_v_to_lvl.values())) == 0
  assert max(list(map_v_to_lvl.values())) == nx.algorithms.dag.dag_longest_path_length(graph_nx) 
  
  return map_v_to_lvl

def compute_reverse_lvl(graph_nx):
  """ All leaves are at zero
  """
  topological_list= nx.algorithms.dag.topological_sort(graph_nx)
  
  map_v_to_reverse_lvl= {}

  for node in topological_list:
    curr_lvl= 0

    for p in graph_nx.predecessors(node):
      if curr_lvl <= map_v_to_reverse_lvl[p]:
        curr_lvl = map_v_to_reverse_lvl[p] + 1
    
    map_v_to_reverse_lvl[node]= curr_lvl

  return map_v_to_reverse_lvl

def relabel_nodes_with_contiguous_numbers(graph_nx, start= 0):
  """
    Creates a shallow copy
  """
  mapping= {n : (idx + start) for idx, n in enumerate(list(graph_nx.nodes()))}

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping
  
def relabel_nodes_with_contiguous_numbers_leaves(graph_nx, start= 0):
  """
    Creates a shallow copy
    leaves from start, start+len(leaves)
    compute from start+len(leaves) + 1, ...
  """
  id_iter= itertools.count(start)

  leaves= get_leaves(graph_nx)
  mapping= {n : next(id_iter) for n in leaves}

  non_leaves= get_non_leaves(graph_nx)
  for n in non_leaves:
    mapping[n]= next(id_iter)

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping
  
def ls_to_str(ls):
  ls= str(ls)
  ls= ls[1:-1]
  return ls

def dfs_topological_sort_single_head(graph_nx, source_node= None, depth_limit= None):
  if source_node == None:
    # use head as source node
    head= check_if_only_one_root(graph_nx)
    source_node= head
  
  graph_nx_revese= graph_nx.reverse(copy=True)
  dfs_topological_list= nx.algorithms.traversal.depth_first_search.dfs_postorder_nodes(graph_nx_revese, source_node, depth_limit)

  return list(dfs_topological_list)

def dfs_topological_sort_recurse(graph_nx, curr_node, 
    node_list: List, 
    done_node: MutableMapping,
    map_v_to_reverse_lvl):
  
  if done_node[curr_node]:
    return
  
  if map_v_to_reverse_lvl != None:
    sorted_preds= sorted(graph_nx.predecessors(curr_node), key= lambda x: map_v_to_reverse_lvl[x], reverse= True)
  else:
    sorted_preds = graph_nx.predecessors(curr_node)

  for pred in sorted_preds:
    dfs_topological_sort_recurse(graph_nx, pred, node_list, done_node, map_v_to_reverse_lvl)

  node_list.append(curr_node)
  done_node[curr_node] = True

  return


def dfs_topological_sort(graph_nx, source_nodes= None):
  """
    NOTE: a recursive algorithm, DO NOT use for large graphs
  """
  if source_nodes == None:
    # use head as source node
    heads= get_head_ls(graph_nx)
    source_nodes= heads
  
  node_list= []
  done_node= defaultdict(lambda: False)
  for head in source_nodes:
    dfs_topological_sort_recurse(graph_nx, head, node_list, done_node, None)

  return node_list    
  

def plot_graph(G, **kw):
  """
    Plots a networkx graph

    parameters: G (a networkx graph)
    **kw : optional keyword arguments to specifcy attributes like node color etc.
  """

  pos= nx.drawing.nx_pydot.graphviz_layout(G)
#  pos= nx.drawing.nx_pydot.pydot_layout(G)
  nx.draw_networkx(G,  pos, **kw)
  plt.show()
  #app= Viewer(G)
  #app.mainloop()

def relabel_nodes_with_contiguous_numbers(graph_nx, start= 0):
  """
    Creates a shallow copy
  """
  mapping= {n : (idx + start) for idx, n in enumerate(list(graph_nx.nodes()))}

  return nx.relabel.relabel_nodes(graph_nx, mapping, copy= True), mapping

