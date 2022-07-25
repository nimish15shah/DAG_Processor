
from collections import defaultdict
import networkx as nx
import itertools
import math

from . import useful_methods
from .useful_methods import clog2

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OPERATOR():
  PRODUCT= 1
  SUM= 2
  LEAF= 3
  DIV= 4
    
class GraphClass():
  def __init__(self, id_iter= itertools.count()):
    self.graph={}
    self.graph_nx= None
    self.id_iter= id_iter

  def create_graph_nx(self):
    assert len(self.graph) != 0
    self.graph_nx= useful_methods.create_nx_graph_from_node_graph(self.graph)

  def create_2input_node(self, child_0, child_1, operator):
    assert operator in ['sum', 'prod']
    assert child_0 in self.graph
    assert child_1 in self.graph
    
    if operator == 'sum':
      key= self.create_sum_node()
    elif operator == 'prod':
      key= self.create_prod_node()
    else:
      assert 0

    self.add_parent_child_edge(key, child_0)  
    self.add_parent_child_edge(key, child_1)  

    return key

  def create_real_valued_leaf_node(self, val=None, mode="force_new", map_real_val_to_node_id= None):
    assert mode in ["force_new", "new_if_unique"]
    
    if mode == "force_new":
      create= True
    elif mode == 'new_if_unique':
      assert val != None
      assert map_real_val_to_node_id != None
      if not val in map_real_val_to_node_id:
        create= True
      else:
        create= False
    else:
      assert 0

    if create:
      node_obj= node(next(self.id_iter))
      node_obj.set_leaf_weight()
      node_obj.curr_val= val

      assert not node_obj.key in self.graph
      self.graph[node_obj.key]= node_obj

    if mode == 'new_if_unique':
      map_real_val_to_node_id[val]= node_obj

    return node_obj.key

  def create_indicator_leaf_node(self, var_id, map_var_id_to_node_id):
    if not var_id in map_var_id_to_node_id:
      node_obj= node(next(self.id_iter))
      node_obj.set_leaf_indicator()
      node_obj.psdd_literal_id = var_id

      map_var_id_to_node_id[var_id]= node_obj
      
      assert not node_obj.key in self.graph
      self.graph[node_obj.key]= node_obj

    return map_var_id_to_node_id[var_id].key

  def create_sum_node(self):
    key= next(self.id_iter)
    node_obj= node(key)
    node_obj.set_sum()
    self.graph[key]= node_obj
    return key

  def create_prod_node(self):
    key= next(self.id_iter)
    node_obj= node(key)
    node_obj.set_prod()
    self.graph[key]= node_obj
    return key

  def create_dummy_node(self, key= None):
    if key == None:
      key= next(self.id_iter)
    node_obj= node(key)
    self.graph[key]= node_obj
    return key

  def add_parent_child_edge(self, parent, child):
    self.graph[parent].child_key_list.append(child)
    self.graph[child].parent_key_list.append(parent)

  def del_parent_child_edge(self, parent, child):
    self.graph[parent].child_key_list.remove(child)
    self.graph[child].parent_key_list.remove(parent)

  def create_chain_of_nodes(self, input_node_list, operator, chain_nodes= set(), curr_graph_nx= nx.DiGraph()):
    assert operator in ['sum', 'prod']

    child_0= None
    for child_1 in input_node_list:
      assert child_1 != None
      if child_0 == None: # first element of the list
        child_0 = child_1
        continue

      if operator == 'sum':
        parent= self.create_sum_node()
      elif operator == 'prod': 
        parent= self.create_prod_node()
      else:
        assert 0
      
      chain_nodes.add(parent)
      self.add_parent_child_edge(parent, child_0)
      self.add_parent_child_edge(parent, child_1)

      curr_graph_nx.add_edge(child_0, parent)
      curr_graph_nx.add_edge(child_1, parent)

      child_0 = parent 
    
    return child_0
              
  def create_tree_of_nodes(self, input_node_list, operator, tree_nodes= set(), curr_graph_nx= nx.DiGraph()):
    assert operator in ['sum', 'prod']
    len_node_list= len(input_node_list)

    assert len_node_list > 0

    if len_node_list == 1:
      assert input_node_list[0] in self.graph
      return input_node_list[0]
    
    if len_node_list > 1:
      # Slicing index is the biggest power of 2 smaller than len_node_list
      biggest_power_2= int(math.log(len_node_list,2))
      slicing_idx= 2**biggest_power_2
      if len_node_list == slicing_idx:
        slicing_idx /= 2
        slicing_idx= int(slicing_idx)

      child_0= self.create_tree_of_nodes(list(input_node_list[ : slicing_idx]), operator, tree_nodes, curr_graph_nx)
      child_1= self.create_tree_of_nodes(list(input_node_list[slicing_idx : ]), operator, tree_nodes, curr_graph_nx)

      if operator == 'sum':
        key= self.create_sum_node()
      elif operator == 'prod': 
        key= self.create_prod_node()
      else:
        assert 0

      tree_nodes.add(key)
      self.add_parent_child_edge(key, child_0)
      self.add_parent_child_edge(key, child_1)

      curr_graph_nx.add_edge(child_0, key)
      curr_graph_nx.add_edge(child_1, key)

      return key

  def reset_compute_state(self):
    "set computed status of leaf nodes and clear for rest"
    for obj in list(graph.values()):
      obj.computed= obj.is_leaf()

class ConfigObjTemplates():
  def __init__(self):
    pass

  def graph_init(self, 
      name = "asia",
      cir_type = "ac",
      graph_mode= "fine",
    ):

    assert cir_type in ["ac", "psdd", "log", "sptrsv"]
    assert graph_mode in ["FINE", "COARSE"]

    self.cir_type= cir_type
    self.graph_mode= graph_mode
    self.name= name

#******    
#** Models a node in the AC 
#*****
class node():
  def __init__(self, node_key):
    self.key= node_key
    self.child_key_list= []
    self.parent_key_list= []
    self.operation_type= OPERATOR.PRODUCT
    self.depth_of_first_common_child= None
    
    #------Properties of BN-----------
    #---------------------------------
    #-- Leaf properteis (Only valid if operation_type=OPERATOR.LEAF)
    self.LEAF_TYPE_INVALID= 0 # Not a leaf
    self.LEAF_TYPE_INDICATOR= 1
    self.LEAF_TYPE_WEIGHT= 2
    self.leaf_type= self.LEAF_TYPE_INVALID
    #self.leaf_numeric_val=0.0 # Set to 1 for True/Don't care indicator and 0 for False indicator. Set to weight value if leaf type is LEAF_TYPE_WEIGHT
    self.leaf_BN_node_name= 'INVALID_BN_NODE' # Valid only if leaf_type == LEAF_TYPE_INDICATOR
    self.leaf_BN_node_state= 'INVALID_STATE' # Valid only if leaf_type == LEAF_TYPE_INDICATOR
    self.leaf_literal_val= 0 # Literal value is the value that is in the line of this leaf node in .net.ac file
    
    #-- Specify if BN_node corresponding to this leaf is evidence, hidden or query?
    self.BN_NODE_TYPE_INVALID= 0 # This node is not a leaf node
    self.BN_NODE_TYPE_EVIDENCE= 1
    self.BN_NODE_TYPE_HIDDEN= 2
    self.BN_NODE_TYPE_QUERY= 3
    self.leaf_BN_node_type= self.BN_NODE_TYPE_INVALID
    
    #---- AC eval and error details------
    #------------------------------------
    # Important value during evaluation
    self.curr_val=0.0
    self.min_val=0.0
    self.max_val=1.0
    
    # List of all BN nodes indicator under this AC node
    self.BN_node_list= [] # values are str
    self.elim_BN_var= [] # If this node is SUM node, this object tells us which BN var it is eliminating

    # Worst-case error instance for this node (depends on which BN var are declared as evidence)
    #self.BN_worst_case_inst= {}

    # Error modelling parameters
    self.rel_error_val= 2
    self.abs_error_val= 2
    self.bits= 32
    self.arith_type= 'fixed'

    #----- HW mapping details-----
    #------------------------------
    # state variables required for mapping to hw
    self.computed= 0 # indicate if the node is available in mem
    self.fully_consumed= False # Indicate that all it's parents are either computed or fully consumed
    self.n_consumed=0 # track the number of times this node is consumed
    self.n_parents_to_be_computed= 0

    # List of parent HW build-block that consume this node, if at all
    #--- Empty list mean that this node won't be stored to the memory and will be fully consumed in the building block itself
    self.parent_buildblk_lst= []

    # List of HW buildblocks of which this node will be a part of (as an intermediate node and not as an input)
    self.compute_buildblk_lst= []

    # Build_block that stores this node
    self.storing_builblk= None
    
    # Level and reverse level
    # level: computed top-down.
    # reverse_level: computed bottom up.
    # In reverse_level, all the leaf node will be at zero level
    # In level, the leaf nodes entering high up in the heirarchy will have a higher level.
    self.level= None
    self.reverse_level= None

    # Nodes below this node
    self.n_nodes_below= None
    
    # dfs level computed by a DFS traversal. Needed during decomposition
    self.dfs_level= None

    # psdd literal id
    self.psdd_literal_id= None


  def get_self_key(self):
    return self.key

  def add_child(self, child_key): # child_key is a key for the child to be added
    assert isinstance(child_key, int), "child_key should be of int type"    
    self.child_key_list.append(child_key)
  
  def get_child_list(self):
    return self.child_key_list
  
  def add_parent(self, parent_key):
    assert isinstance(parent_key, int), "parent_key should be of int type"    
    self.parent_key_list.append(parent_key)
  
  def get_parent_list(self):
    return self.parent_key_list

  def set_operation(self, operation):
    self.operation_type= operation
   
  def print_attr(self):
    print("Self key: ", self.get_self_key(), end=' ')
    print("Children list: ", self.get_child_list(), end=' ')
    print("Parent list: ", self.get_parent_list(), end=' ')
    if (self.operation_type == OPERATOR.PRODUCT):
      print("Operation: Product")
    elif (self.operation_type == OPERATOR.SUM):
      print("Operation: Sum")
    elif (self.operation_type == OPERATOR.LEAF):
      print("Operation: Leaf")
    else:
      print(' ')
  
  # Methods to check type of node
  def is_head(self):
    if len(self.parent_key_list) == 0:  return True
    else: return False

  def is_leaf(self):
    if self.operation_type == OPERATOR.LEAF:
      return True
    else:
      return False

  def is_sum(self):
    if self.operation_type == OPERATOR.SUM:
      return True
    else:
      return False

  def is_prod(self):
    if self.operation_type == OPERATOR.PRODUCT:
      return True
    else:
      return False
  
  def is_weight(self):
    if self.is_leaf():
      assert self.leaf_type != self.LEAF_TYPE_INVALID
      if self.leaf_type == self.LEAF_TYPE_WEIGHT:
        return True
    return False

  def is_indicator(self):
    if self.is_leaf():
      assert self.leaf_type != self.LEAF_TYPE_INVALID
      if self.leaf_type == self.LEAF_TYPE_INDICATOR:
        return True
    return False

  # Methods to set type of node
  def set_sum(self):
    self.operation_type= OPERATOR.SUM

  def set_prod(self):
    self.operation_type= OPERATOR.PRODUCT
  
  def set_leaf_weight(self):
    self.operation_type= OPERATOR.LEAF
    self.leaf_type= self.LEAF_TYPE_WEIGHT
    self.computed = True

  def set_leaf_indicator(self):
    self.operation_type= OPERATOR.LEAF
    self.leaf_type= self.LEAF_TYPE_INDICATOR

# **** Class to model actual hw operators
# ** contains diffent flags for HW properties. If set, it means that functionality is available
# ****
class HW_OPERATOR():
  def __init__(self):
    self.SUM= False       # Operator can perform sum
    self.PRODUCT= False   # Operator can perform product
    self.PASS_IN1= False  # Can pass in1 (instead of performing an arithmetic op)
    self.PASS_IN2= False  # Can pass in2
    self.LEAF= False      # s a leaf node
    self.OUT= False       # Output is stored to memory
  
  def reset_all(self):
    self.SUM= False
    self.PRODUCT=  False
    self.PASS_IN1= False
    self.PASS_IN2= False
    self.LEAF= False
    self.OUT= False       
  
  def print_attr(self):
    print("S_P_IN1_IN2_L_O=",self.SUM, self.PRODUCT, self.PASS_IN1, self.PASS_IN2, self.LEAF, self.OUT)

#****
# A class to model the hw_node in hw_struct
#****
class hw_node(node):
  def __init__(self, node_key):
    node.__init__(self, node_key)
    self.operation_type= HW_OPERATOR()

class BN_node():
  """ Class to model a BN node
  """
  def __init__(self, name):
    self.node_name= name
    self.states=[] # name str
    self.potential=[] 
    self.parent_list=[] # name str
    self.child_list=[] # name str
     
    #self.query_state= 'NO_STATE' # name of the state that we want to set lambda for in AC query
    
    # Node type
    self.HIDDEN_NODE= 0
    self.EVIDENCE_NODE= 1
    self.QUERY_NODE=2
    self.node_type= self.HIDDEN_NODE

    # Dictionary of leaf nodes in AC that corresponds to indicator
    # Key: state name (str)
    # Val: AC Leaf node key
    self.AC_leaf_dict= {}

class build_blk():
  """ This class is to model the details of the building blocks, like the details of inputs. outputs etc.
  """
  
  def __init__(self, blk_key):
    self.blk_key= blk_key
    
    # Parents and Children list (without duplicates)
    self.parent_bb_lst= []
    self.child_bb_lst=[]
    
    # parents and children list (with duplicates) (required for some graphical algorithms)
    self.parent_bb_lst_duplicates=[]
    self.child_bb_lst_duplicates=[]

    # List of AC nodes that are stored
    self.out_list= []
    
    # Map output nodes to PE
    # Key: node id, Val: tuple (t,lvl,pe)
    self.out_nodes_to_pe= {}
    
    # Map PEs to global nodes
    # Key: pe_tup (t, lvl, pe), Val: global node ids
    self.pe_to_node= {}

    # List of input AC nodes
    self.in_list= []
    self.in_list_unique=[]

    # List of internal nodes (including output nodes)
    self.internal_node_list=[]
    
    # Inputs that are leaf in AC
    self.leaf_inputs= []

    # Number of inputs that are leaf nodes in AC
    self.n_leaf_inputs= None

    # Set of decompose objects (of class decompose_unit) that would be mapped on this BB
    self.set_of_decompose_units= None
    
    # Dict to map a decompose unit to it's ordered input list
    # It is computed during bank allocation, in bank_allocate.py
    # Key: decompose_unit (a member of set_of_decompose_units)
    # Val: An ordered list of inputs. E.g. [3, 10, None, 4, 11, 12, None, None, None, 6]. This means 3 and 10, 11 and 12 are siblings at the lowest level of tree. 4 enters the tree at a higher level. 6 enters an an even higher level.
    self.map_decompose_unit_to_ordered_input_list= None
    
    # Information to help mapping BB to physical PEs
    # Describes which combination of tree is selected for scheduling. For example, trees [4,3,3] is chosen, or [4,4], or [2,2,2,2].
    # Type: integer (idx to use for hw_details generated by hw_struct_methods.py)
    self.selected_hw_set_idx= None

    self.level=-1000 # Leaf level is 0
    self.reverse_level= None

    self.sch_level= None

    # For creating visualization
    self.UNCOMPUTED= 0
    self.COMPUTED= 1
    self.INPUTS_IN_VECTOR= 2
    self.RECENT_INPUTS= 3
    self.INPUTS_IN_SCALAR= 4
    self.status= self.UNCOMPUTED
    
  def is_leaf(self):
    if len(self.child_bb_lst) == 0:
      return False
    else:
      return True
  
  def node_to_pe_lvl_and_unit(self, node):
    """
      returns set of levels of PE that stores this node
      "set" becuase there could be multiple PE's computing this node
    """
    assert node in self.out_list
    decompose_unit= None
    for unit in self.set_of_decompose_units:
      if decompose_unit != None:
        break
      if node in unit.uncomputed_nodes:
        for idx, ls in list(unit.child_dict.items()):
          if node in ls:
            lvl =  len(unit.child_dict)-idx-1 # First level PEs gets lvl=1, next lvl=2, and so on... 
            decompose_unit= unit
            decompose_unit.output_nodes.add(node)
            break
    
#    assert [unit_node for ls in decompose_unit.child_dict.values() for unit_node in ls].count(node) == 1
#    assert decompose_unit.child_dict[len(decompose_unit.child_dict) - lvl - 1].count(node) == 1, (decompose_unit.child_dict, node)
    
    assert decompose_unit != None

    return lvl, decompose_unit
  
  def print_unit_details(self):
    for unit in self.set_of_decompose_units:
      print(unit.parent, unit.child_dict, unit.output_nodes)
  
class decompose_unit_node():
  """
    Contains information for non-duplicate nodes in decompose_unit
  """
  def __init__(self, local_id, global_id):
    self.local_id= local_id
    self.global_id= global_id
    self.local_child_key_ls= set()
    self.global_child_key_ls= set()


class decompose_unit():
  """
    Contains information of connected sub-graph that has to be scheduled in a building blocks

    A bulding block (BB) contans several such decompose units
  """

  def __init__(self):
    # Top parent key
    self.parent= None

    # A normal graph to encode unit structure 
    # Key: Node key
    # Val: Children
    self.struct_graph= None

    # An NX graph to encode unit structure
    # NOTE: THIS MAY NOT BE A TREE
    self.struct_graph_nx= None
    
    # An NX graph that converts a struct_graph_nx to a tree, by using new local IDs instead of global IDs
    # Local id start with 0. Top node, i.e. self.parent is always assigned 0
    # Attr: 'details' is an object of type decompose_unit_node
    self.struct_graph_nx_tree= None

    # Child dict
    self.child_dict= None
    
    # Sets, no repeatetion
    self.input_list= None
    self.uncomputed_nodes= set()
    self.all_nodes= None

    # Uncomputed noes with repeatition
    self.uncomputed_nodes_repeated= []

    # NOTE: To be computed during bank allocation
    self.output_nodes= set()
    self.map_output_to_pe= {}
    self.map_output_to_local= {}

    # Total uncomputed nodes
    self.uncomputed_cnt= None

    # A dict to Map node to pe tuple
    # Key: local_id, Val: pe_tup
    self.map_local_to_pe= {}
    
    # A dict to map PE tuples to node
    # Key: PE tup, Val: Node
    self.map_pe_to_local= {}
    self.map_pe_to_global= {}

    # Flag for efficient decomposition
    self.SCHEDULABLE_FLAG= None
  
  def local_to_global(self, local_id):
    node_details= self.local_node_details()
    return node_details[local_id].global_id
  
  def local_node_details(self):
    return nx.get_node_attributes(self.struct_graph_nx_tree, 'details')

  def global_to_local(self, global_id):
    node_details= self.local_node_details()
    return set([local_id for local_id, details in list(node_details.items()) if details.global_id == global_id])
  
  def lvl_local(self, local_id):
    global_id= self.local_to_global(local_id)
    return self.lvl_ls(global_id)[0]

  def lvl_ls(self, node):
    lvl_ls= [idx for idx, ls in list(self.child_dict.items()) if node in ls]
    lvl_ls= [len(self.child_dict) - idx - 1 for idx in lvl_ls]
    
    assert len(set(lvl_ls)) == 1, "Assumption: All instances of a node in unit should be at the same level."
    return lvl_ls

  def parent_ls(self, node):
    parents= list(self.struct_graph_nx.successors(node))
    return parents 

  def children_ls(self, node):
    children= list(self.struct_graph_nx.predecessors(node))
    return children 

  def parent_ls_local(self, local_id):
    parent= self.struct_graph_nx_tree.successors(local_id)
    assert len(parent) == 1
    return parent[0] 

  def children_ls_local(self, local_id):
    children= list(self.struct_graph_nx_tree.predecessors(local_id))
    return children 
  
  def is_input_local(self,local_id):
    if len(self.children_ls_local(local_id)) == 0:
      assert len(list(self.struct_graph_nx_tree.predecessors(local_id))) == 0
      return True
    return False
  
  def is_input(self,node):
    if len(self.children_ls(node)) == 0:
      assert node in self.input_list
      return True
    return False
  
  def is_output(self, node):
    if node in self.output_nodes:
      return True
    return False
  
  def is_uncomputed(self, node):
    if node in self.uncomputed_nodes:
      return True
    return False

  def sibling_ls(self, node):
    """ returns the other child of the parent
    Only TOP node doesn't have a sibling
    Also see: uncomputed_sibling
    """
    parent_ls= self.parent_ls(node)

    ls= []
    for parent in parent_ls:
      ls += children_ls(parent)
    
    ls = list(set(ls) - set([node]))

    return ls

  def uncomputed_sibling_ls(self, node):  
    """ returns the other child of the parent if that node is an uncomputed node
    """
    parent_ls= self.parent_ls(node)

    ls= []
    for parent in parent_ls:
      ls += self.children_ls(parent)
    
    ls = list(set(ls) - set([node]))
    
    ls= [sib for sib in ls if self.is_uncomputed(sib)]
    
    return ls

  def update_pe_detail_local(self,local_id, pe):
    global_id= self.local_to_global(local_id)
    self.map_local_to_pe[local_id]= pe
    self.map_pe_to_local[pe]= local_id
    self.map_pe_to_global[pe]= global_id

    if local_id in list(self.map_output_to_local.values()): # This is an output node
      self.map_output_to_pe[global_id]= pe
      assert self.map_output_to_local[global_id] == local_id

  def output_to_local(self, output):
    assert output in self.output_nodes
    if output not in self.map_output_to_local:
      local_id= self.global_to_local(output).pop()
      self.map_output_to_local[output]= local_id
    
    return self.map_output_to_local[output]
  
  def descendants_of_output(self, output):
    """
      Decides descendants according to mapping to local_id
    """
    assert output in self.output_nodes
    descendants_local= nx.algorithms.dag.ancestors(self.struct_graph_nx_tree, self.map_output_to_local[output]) # NX ancestors are our descendants
    # remove leaf nodes
    descendants_local= set([node for node in descendants_local if not self.is_input_local(node)])
    descendants= set([self.local_to_global(local_id) for local_id in descendants_local])
    
    return descendants, descendants_local

  def ancestors_of_output(self, output):
    """
      Decides ancestors according to mapping to local_id
    """
    assert output in self.output_nodes
#      ancestors= list(nx.all_simple_paths(struct_graph_nx, node, unit.parent)) # Nodes on only one path to the Top node have to be updated
    ancestors_local= set(nx.algorithms.dag.descendants(self.struct_graph_nx_tree, self.map_output_to_local[output])) # NX ancestors are our descendants 
    ancestors= set([self.local_to_global(local_id) for local_id in ancestors_local])
    
    return ancestors, ancestors_local
  
  def common_ancestor_of_outputs(self, output_0, output_1):
    _, ancestors_local_0= self.ancestors_of_output(output_0)
    _, ancestors_local_1= self.ancestors_of_output(output_1)
    common_parent_local=  ancestors_local_0 & ancestors_local_1
    common_parent_local= min(common_parent_local, key= lambda x: self.lvl_local(x))
    
    return self.local_to_global(common_parent_local), common_parent_local

  def uncomputed_nodes_local(self):
    return set([node for node in self.struct_graph_nx_tree.nodes() if not self.is_input_local(node)])

class reg():
  def __init__(self):
    self.data= 0
  

class reg_file():
  def __init__(self, size= 0, name= ''):
    self.file_size= size
    self.name= name

    # Keep track of what was the highest size required during computation
    self.max_file_size= 0

    self.FREE= 'FREE'
    self.OCCUPIED= 'OCCUPIED'
    # Status of a register in file. If free or Occupied
    # key: reg position
    # val: FREE or OCCUPIED
    self.reg_status= {}
    
    # List of free regs
    self.free_pos_lst= list(range(size))

    # List of occupied regs
    self.occup_pos_lst= []
    
    # A dict to hold data and metadata about the
    # Key- Position
    # Val: Object of type register
    self.file_content= dict.fromkeys(self.free_pos_lst, None)

    # Map BB_key to reg positions
    # Key: App index (a number that application understands, typically Building Block key)
    # Val: List of Reg positions where data for AppIdx is stored (can be used as key for self.file_content)
    self.map_AppIdx_to_pos= defaultdict(list)
    
    
    # Map pos to BB_keys
    # Key: Reg position (can be used as key for self.file_content)
    # Val: List of App indices (a number that application understands, typically Building Block key)
    self.map_pos_to_AppIdx= defaultdict(list)

  def read_by_pos(self,pos):
    assert self.reg_status[pos] == self.OCCUPIED , "This position does not have valid data"
    return self.file_content[pos].data

  def read_by_AppIdx(self,AppIdx):
    pos_lst= self.map_AppIdx_to_pos[AppIdx]
    data_lst= []
    for pos in pos_lst:
      data_lst.append(self.file_content[pos].data)

    return {'data': data_lst, 'pos': pos_lst}
  
  def update_file_size(self):
    self.file_size = len(self.file_content)
    if self.max_file_size < self.file_size:
      self.max_file_size = self.file_size
  
  def allocate_AppIdx(self, AppIdx):
    #assert AppIdx not in self.map_AppIdx_to_pos, "AppIdx is already allocated to the file"
    if self.free_pos_lst:
      pos= self.free_pos_lst.pop()
    else:
      pos= len(self.occup_pos_lst) + 1
      
    assert pos not in self.occup_pos_lst, "This position is already occupied by some other AppIdx"

    assert pos not in self.map_AppIdx_to_pos[AppIdx], "Appendind pos that is already there {} {}".format(pos, AppIdx)
    assert AppIdx not in self.map_pos_to_AppIdx[pos], "Appendind pos that is already there {} {} ".format(pos, AppIdx)
    
    self.map_AppIdx_to_pos[AppIdx].append(pos)
    self.map_pos_to_AppIdx[pos].append(AppIdx)

    self.file_content[pos]= reg()
    self.occup_pos_lst.append(pos)
    self.file_size= len(self.file_content)
    
    if self.max_file_size < self.file_size:
      self.max_file_size = self.file_size
    

    self.update_file_size()

    return pos

  def deallocate_AppIdx(self, AppIdx, pos_lst=None):
    """
      Use pos_lst to specify a particular list of positions to deallocate for
    """
    if pos_lst == None:
      pos_lst= self.map_AppIdx_to_pos[AppIdx]
      self.map_AppIdx_to_pos.pop(AppIdx)
    else:
      for pos in pos_lst:
        self.map_AppIdx_to_pos[AppIdx].remove(pos)

    for pos in pos_lst:
      self.map_pos_to_AppIdx[pos].remove(AppIdx)
      
      if not self.map_pos_to_AppIdx[pos]: # List empty
        self.occup_pos_lst.remove(pos)
        self.free_pos_lst.append(pos)

        self.file_content.pop(pos)
  
  def write_by_AppIdx(self, data, AppIdx):
    """ Need to allocate before writting. This function does auto allocate
    """
    assert 0, "pos is a list not a single location"
    if AppIdx not in self.map_AppIdx_to_pos:
      self.allocate_AppIdx(AppIdx)

    pos= self.map_AppIdx_to_pos[AppIdx]

    self.file_content[pos].data= data
  
  def get_free_space(self):
    """ Returns the amount of free space available
    """
    return len(self.free_pos_lst)
  
  def get_occ_space(self):
    return len(self.occup_pos_lst)

  def get_AppIdx(self):
    """ returns list of AppIdx currently allocated in the reg_file
    """
    return list(self.map_AppIdx_to_pos.keys())
  
  def get_AppIdx_for_pos(self, pos):
    """
      Returns a list of all the AppIdx stored at pos
    """
    assert pos in self.map_pos_to_AppIdx, "pos not allocated"
    assert pos in self.occup_pos_lst, "pos not allocated"    
    
    return self.map_pos_to_AppIdx[pos] 

  def map_pos_to_new_AppIdx(self,pos, AppIdx):
    assert pos not in self.map_AppIdx_to_pos[AppIdx], "Appending pos that is already there {} {}".format(pos, AppIdx)
    assert AppIdx not in self.map_pos_to_AppIdx[pos], "Appending pos that is already there {} {} ".format(pos, AppIdx)

    self.map_pos_to_AppIdx[pos].append(AppIdx)
    self.map_AppIdx_to_pos[AppIdx].append(pos) 
  
class io_node_details():
  def __init__(self, node_id):
    self.node_id= node_id
    
    # reg_bank
    self.bank= None

    # reg_pos
    self.pos= None

  def print_details(self):
    print(f"node: {self.node_id}, bank, pos: {self.bank}, {self.pos}")

class instr():
  id_iter= itertools.count()

  def __init__(self,name):
    self.isa= ['bb', 'nop', 'nop_stall', 'ld', 'ld_sc', 'st', 'sh', 'st_4', 'st_8', 'sh_2', 'sh_4', 'sh_8']
    assert name in self.isa, 'Incorrect name check: ' + str(name)

    self.name= name
    self.id= next(self.id_iter)

    # Nodes on which this instr depend
    self.required_nodes= set()
    
    # Nodes produced by this instruction
    # instructions that use this produced nodes should be pipline distance away from this instruciton
    self.produced_nodes= set()
  
    """
    if instr.is_type('nop'):
    elif instr.is_type('nop_stall'):
    elif instr.is_type('ld') or instr.is_type('ld_sc'):
    elif instr.is_type('st'):
    elif instr.is_type('st_2'):
    elif instr.is_type('st_4'):
    elif instr.is_type('sh'):
    elif instr.is_type('sh_2'):
    elif instr.is_type('sh_4'):
    elif instr.is_type('sh_8'):
    elif instr.is_type('bb'):
    else:
      assert False
    """

  def is_type(self, name):
    assert name in self.isa, 'Incorrect name check: ' + str(name) + str(self.isa)
    
    if name == self.name:
      return True
    else:
      return False

  def print_details(self):
    logger.info(f"instruction type and id: {self.name} {self.id}")

class nop_instr(instr):
  def __init__(self):
   instr.__init__(self, 'nop') 

  def reg_input(self):
    return {}

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    instr_len = 4
    instr_len_wo_opt = instr_len 

    return instr_len, instr_len_wo_opt

class ld_sc_instr(instr):
  """
    Load from scratch memory
  """
  def __init__(self):
   instr.__init__(self, 'ld_sc') 

   self.dependent_bb_set= None
   self.node_set= None
   self.mem_addr= None

   #dict
  # key: node id, val: io_node_details object
   self.node_details_dict= None

  def reg_input(self):
    return {} 

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    instr_len = 4 + 32 + b
    instr_len_wo_opt = instr_len + b * clog2(r) 

    return instr_len, instr_len_wo_opt

class ld_instr(instr):
  def __init__(self):
   instr.__init__(self, 'ld') 

   self.dependent_bb_set= None
   self.node_set= None
   self.mem_addr= None

   #dict
  # key: node id, val: io_node_details object
   self.node_details_dict= None

  def reg_input(self):
    return {}

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    instr_len = 4 + 32 + b
    instr_len_wo_opt = instr_len + b * clog2(r) 

    return instr_len, instr_len_wo_opt


class st_instr(instr):
  def __init__(self):
   instr.__init__(self, 'st') 
   
   #self.source_bb_set= None
   self.node_set= None
   self.mem_addr= None
   
   # dict
  # key: node id, val: io_node_details object
   self.node_details_dict= None
  
  def reg_input(self):
    assert len(self.node_set) == len(list(self.node_details_dict.keys()))
    return {node: obj.bank for node, obj in list(self.node_details_dict.items())}

  def is_type(self, name):
    return_val= instr.is_type(self, name)

    if return_val:
      if name == 'st_4':
        assert len(self.node_set) <= 4
      elif name == 'st_8':
        assert len(self.node_set) <= 8

    return return_val

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    instr_len = 4 + 32 + b + b * clog2(r) 
    instr_len_wo_opt = instr_len 

    return instr_len, instr_len_wo_opt

class bb_instr(instr):
  def __init__(self):
    instr.__init__(self, 'bb') 
     
    self.bb_id= None

    
    # key: node id, val: io_node_details object
    self.in_node_details_dict= None
    self.out_node_details_dict= None

    # Set of nodes that has to be invalidated
    self.invalidate_node_set= set()

    self.output_to_pe= {}

    # Key: pe_tup
    # Val: Object of type pe_details
    self.pe_details= {}

  def reg_input(self):
    return {node: obj.bank for node, obj in list(self.in_node_details_dict.items())}

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    n_tree = int(b / (2**d))
    instr_len = 4 + b + b + b * clog2(r) + b * clog2(b) + b * clog2(d) + (n_tree * ((2**d) - 1)) * 2
    instr_len_wo_opt = instr_len + b * clog2(r)

    return instr_len, instr_len_wo_opt

  
  def print_details(self):
    logger.info(f"instruction type and id: {self.name} {self.id}")
    print(f"required_nodes: {self.required_nodes}")
    print(f"produced_nodes: {self.produced_nodes}")
    print(f"bb_id: {self.bb_id}")
    print(f"output_to_pe: {self.output_to_pe}")
    print(f"invalidate_node_set: {self.invalidate_node_set}")
    
    for n , obj in self.in_node_details_dict.items():
      print("in_node_details_dict:")
      obj.print_details()

    for n , obj in self.out_node_details_dict.items():
      print("out_node_details_dict:")
      obj.print_details()

    for pe_tup, obj in self.pe_details.items():
      if obj.node != None:
        obj.print_details()

class pe_details():
  def __init__(self, pe_tup):
    self.pe= pe_tup # (t, lvl, pe) # lvl starts from 1 and not 0
    self.node= None

    self.__SUM= '+'
    self.__PROD= '*'
    self.__PASS_0= '0->'
    self.__PASS_1= '1->'
    self.__op_type= None

    # TO be assigned a tuple (bank, pos), if applicable
    self.input_0_reg= None 
    self.input_1_reg= None 
      
    # TO be assigned a tuple (bank, pos), if applicable
    self.output_reg= None

  def set_op(self, operation):
    assert operation in ['sum', 'prod', 'pass_0', 'pass_1'], "'operation' argument should be one of these"
    if operation == 'sum':
      self.__op_type = self.__SUM
    if operation == 'prod':
      self.__op_type = self.__PROD
    if operation == 'pass_0':
      self.__op_type = self.__PASS_0
    if operation == 'pass_1':
      self.__op_type = self.__PASS_1

  def is_leaf(self):
    if self.pe[1] == 1: # PEs at level 1 are leaf nodes
      return True
    return False
    
  def is_sum(self):
    assert self.__op_type in [self.__SUM, self.__PROD, self.__PASS_0, self.__PASS_1]
    if self.__op_type == self.__SUM:
      return True
    return False

  def is_prod(self):
    assert self.__op_type in [self.__SUM, self.__PROD, self.__PASS_0, self.__PASS_1]
    if self.__op_type == self.__PROD:
      return True
    return False

  def is_pass_0(self):
    assert self.__op_type in [self.__SUM, self.__PROD, self.__PASS_0, self.__PASS_1]
    if self.__op_type == self.__PASS_0:
      return True
    return False
  
  def is_pass_1(self):
    assert self.__op_type in [self.__SUM, self.__PROD, self.__PASS_0, self.__PASS_1]
    if self.__op_type == self.__PASS_1:
      return True
    return False

  def print_details(self):
    print(f"pe_tup : {self.pe}")
    print(f"node : {self.node}")
    print(f"__op_type : {self.__op_type}")
    print(f"input_0_reg : {self.input_0_reg}")
    print(f"input_1_reg : {self.input_1_reg}")
    print(f"output_reg : {self.output_reg}")

class sh_instr(instr):
  def __init__(self, sh_type, instr_id):
    instr.__init__(self, 'sh') 
    self.corresponding_instr_id=  instr_id

    # shuffle dict for bank
    # Key: node
    # Val: (src_bank, dest_bank) 
    self.sh_dict_bank= {}
    
    # shuffle dict for pos
    # Key: node
    # Val: (src_pos, dst_pos)
    self.sh_dict_pos= {}

    # Set of nodes that has to be invalidated
    self.invalidate_node_set= set()

    # sh type
    self.__BB_IN= 'bb_in'
    self.__BB_OUT= 'bb_out'
    self.__LD= 'ld'
    self.sh_type= None
    
    assert sh_type in ['bb_in', 'bb_out', 'ld']
    if sh_type == 'bb_in':
      self.sh_type= self.__BB_IN
    elif sh_type == 'bb_out':
      self.sh_type= self.__BB_OUT
    elif sh_type == 'ld':
      self.sh_type= self.__LD
    else:
      assert 0
  
  def is_type(self, name):
    return_val= instr.is_type(self, name)

    if return_val:
      if name == 'sh_2':
        assert len(self.sh_dict_bank) <= 2
      elif name == 'sh_4':
        assert len(self.sh_dict_bank) <= 4
      elif name == 'sh_8':
        assert len(self.sh_dict_bank) <= 8

    return return_val

  def print_details(self):
    logger.info(f"instruction type and id: {self.name} {self.id}")
    logger.info("Nodes shifting: FROM src bank, pos TO dst bank, pos | whether to invalidate src register")
    for node in self.sh_dict_bank:
      src_bank, dst_bank = self.sh_dict_bank[node]
      src_pos, dst_pos = self.sh_dict_pos[node]
      invalid = (node in self.invalidate_node_set)
      logger.info(f"node: {node}, FROM {src_bank}, {src_pos} TO {dst_bank}, {dst_pos} | {invalid}")

    logger.info(f"invalidate_node_set nodes: {self.invalidate_node_set}")
    logging.info(f"sh_type : {self.sh_type}")


  def insert_sh_bank(self, target_node, src_bank, dst_bank):
    # Ensure there is no read conflict
    assert len([node for node, src_dst_bank_in_dict in list(self.sh_dict_bank.items()) if src_dst_bank_in_dict[0]== src_bank]) == 0, [target_node,src_bank, dst_bank, self.sh_dict_bank]

    # Ensure there is no write conflict
    assert len([node for node, src_dst_bank_in_dict in list(self.sh_dict_bank.items()) if src_dst_bank_in_dict[1]== dst_bank]) == 0, [target_node,src_bank, dst_bank, self.sh_dict_bank]
    self.sh_dict_bank[target_node]= (src_bank, dst_bank)
  
  def is_bb_in(self):
    assert self.sh_type in ['bb_in', 'bb_out', 'ld']
    if self.sh_type == self.__BB_IN:
      return True
    return False

  def is_bb_out(self):
    assert self.sh_type in ['bb_in', 'bb_out', 'ld']
    if self.sh_type == self.__BB_OUT:
      return True
    return False
  
  def is_ld(self):
    assert self.sh_type in ['bb_in', 'bb_out', 'ld']
    if self.sh_type == self.__LD:
      return True
    return False

  def reg_input(self):
    return {node: src_dst_tup[0] for node, src_dst_tup in list(self.sh_dict_bank.items())}

  def len_w_and_wo_auto_wr_addr(self, d, b, r):
    instr_len = 4 + b + 4 * clog2(b) + 4 * clog2(r)
    instr_len_wo_opt = instr_len + 4 * clog2(r) 

    return instr_len, instr_len_wo_opt

class instr_ls():
  def __init__(self):
    # list of instructions
    self.instr_ls= []
  
  def print_instr(self):
    instr_str= [ instr.name for instr in self.instr_ls]
    useful_methods.printlog("Instruction sequence:")
    print(instr_str)
  

class Mutable_level():
  max_lvl= 0

