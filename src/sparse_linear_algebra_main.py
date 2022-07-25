
from . import files_parser
import networkx as nx
from . import reporting_tools
from . import common_classes
from . import ac_eval
from . import useful_methods

import time
import math
import scipy.io
import matplotlib.pylab as plt
import scipy as sp
from scipy.sparse import linalg, csc_matrix, csr_matrix
import numpy as np
import itertools
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Coarse_node_mappings():
  def __init__(self, coarse_n, adder_tree_set, mul_set, final_prod_node, b_key):
    self.coarse_n= coarse_n
    self.adder_tree_set= adder_tree_set
    self.mul_set= mul_set
    self.final_prod_node= final_prod_node
    self.b_key= b_key
    self.all_nodes= self.adder_tree_set | self.mul_set | set([self.final_prod_node])
    # self.full_row_graph = graph_nx.subgraph(self.all_nodes)

    self.map_n_to_semi_coarse= {}
    self.map_semi_coarse_to_tup= {}

  def create_coarse_nodes(self, graph, graph_nx, curr_partition):
    mul_nodes= curr_partition & self.mul_set
    adder_nodes= curr_partition & self.adder_tree_set
    nodes_mapped = curr_partition & self.all_nodes

    # print(len(self.all_nodes))
    curr_graph= graph_nx.subgraph(nodes_mapped)
    topological_list= list(nx.algorithms.dag.topological_sort(curr_graph)) 
    topological_list_adders= [n for n in topological_list if graph[n].is_sum()]
    # these adders will be part of the mac operation
    # rest will be a separate adder-only tree operation
    adder_nodes_mac= set() 
    adder_nodes_pure_tree= set()
    for n in topological_list_adders:
      predecessors = set(graph_nx.predecessors(n))
      # remove b_key from predecessor
      predecessors &= self.all_nodes

      # either predecessor is not in nodes_mapped
      # or one of its predecessor is itself in adder_nodes_pure_tree
      if len(predecessors - nodes_mapped) or len(predecessors & adder_nodes_pure_tree):
        adder_nodes_pure_tree.add(n)

    adder_nodes_mac = adder_nodes - adder_nodes_pure_tree
    
    n_coarse_nodes = 0
    operation_nodes_tup_ls= []
    if self.final_prod_node in curr_partition:
      n_coarse_nodes += 1
      operation_nodes_tup_ls.append(('final', set([self.final_prod_node])))

    if len(mul_nodes) !=0:
      n_coarse_nodes += 1
      operation_nodes_tup_ls.append(('mac', mul_nodes | adder_nodes_mac))

    if len(adder_nodes_pure_tree) !=0:
      n_coarse_nodes += 1
      operation_nodes_tup_ls.append(('add', adder_nodes_pure_tree))
      assert len(nodes_mapped) != len(self.all_nodes)

    assert len(operation_nodes_tup_ls) != 0
    return n_coarse_nodes, nodes_mapped, operation_nodes_tup_ls
    
class Matrix_statistics():
  def __init__(self, name):
    self.name= name
    self.U_or_L= None
    self.nrows= None
    self.ncols= None
    self.nnz= None

    self.critical_path_len_coarse= None
    self.critical_path_len_fine_tree= None
    self.critical_path_len_fine_chain= None
    self.critical_path_len_fine_hybrid= None

    # following should be lists,
    # where each element shows the number of nodes in that layer
    self.greedy_layer_wise_coarse= None
    self.greedy_layer_wise_fine_tree= None
    self.greedy_layer_wise_fine_chain= None
    self.greedy_layer_wise_fine_hybrid= None
    # as last as possible
    self.ALAP_layer_wise_coarse= None
    self.ALAP_layer_wise_fine_tree= None
    self.ALAP_layer_wise_fine_chain= None
    self.ALAP_layer_wise_fine_hybrid= None

  def get_str(self):
    stat_str  = ""
    stat_str += self.name + ',' 
    stat_str += self.U_or_L + ',' 
    stat_str += f"{self.nrows},{self.ncols},{self.nnz},"
    stat_str += f"critical_path_len_coarse, {self.critical_path_len_coarse},"
    stat_str += f"critical_path_len_fine_tree, {self.critical_path_len_fine_tree},"
    stat_str += f"critical_path_len_fine_chain, {self.critical_path_len_fine_chain},"
    stat_str += f"critical_path_len_fine_hybrid, {self.critical_path_len_fine_hybrid},"

    return stat_str

class SparseTriangularSolve():
  def __init__(self, global_var, mtx_file_name, read_files= False, write_files= False, verify= False, graph_mode= 'FINE', output_mode= 'default', plot_matrix= False):
    assert output_mode in ['default', 'single_node']
    self.output_mode = output_mode

    self.global_var= global_var
    self.mtx_file_name= mtx_file_name

    try:
      self.A= files_parser.read_mat(global_var.DATA_PATH + mtx_file_name + '.mat')
    except:
      logger.warning(f"not able to read file for matrix: {mtx_file_name}")
      return

    self.nrows, self.ncols= self.A.shape
    self.nnz= self.A.count_nonzero()

    assert self.nrows == self.ncols

    # LU factors
    # Pr.T * L * U * Pc.T= A
    if not read_files:
      try:
        self.lu = linalg.splu(self.A)
      except:
        logger.warning(f"LU decomposition failed for matrix {mtx_file_name}")
        return

      self.L= self.lu.L # scipy csr matrix, self.L.A gives numpy array
      self.U= self.lu.U # scipy csr matrix
      # permutation matrices:
      self.Pr = csc_matrix((np.ones(self.nrows), (self.lu.perm_r, np.arange(self.nrows))))
      self.Pc = csc_matrix((np.ones(self.nrows), (np.arange(self.nrows), self.lu.perm_c)))

      if write_files:
        self.write_mm_file(self.L, self.mtx_file_name.replace('/', '_')+'_L')
        self.write_mm_file(self.U, self.mtx_file_name.replace('/', '_')+'_U')
        return

    else: # read the factors directly
      self.L= self.read_mm_file(self.mtx_file_name.replace('/', '_')+'_L')
      self.U= self.read_mm_file(self.mtx_file_name.replace('/', '_')+'_U')
      self.Pr= None
      self.Pc= None

    logger.info(f"matrix {mtx_file_name} loaded")

    # reporting_tools.show_matrix(self.A.A)
    # reporting_tools.show_matrix(self.L.A)
    # reporting_tools.show_matrix(self.U.A)
    # graph_obj= self.create_digraph(self.L, 'L')
    # reporting_tools.plot_graph_nx_graphviz(graph_obj.graph_nx)
    # reporting_tools.show_image()
    # exit(1)

    # all nonzero on diagonal
    # assert np.count_nonzero(self.L.diagonal()) == self.nrows
    # assert np.count_nonzero(self.U.diagonal()) == self.nrows
    logger.info(f"nrows: {self.nrows}, ncols:{self.ncols}, nnz:{self.nnz}")

    if graph_mode == 'FINE':
      self.L_graph_obj, self.L_map_x_to_node, self.L_map_b_to_node, self.L_map_nz_idx_to_node, self.L_map_r_to_nodes_info= \
          self.create_arithmetic_graph_directly('L', 'hybrid')
      logger.info(f"Fine critical path length: {nx.algorithms.dag.dag_longest_path_length(self.L_graph_obj.graph_nx)}")

    self.L_coarse_graph_obj= self.create_digraph(self.L, 'L')
    logger.info(f"Coarse critical path length: {nx.algorithms.dag.dag_longest_path_length(self.L_coarse_graph_obj.graph_nx)}")

    # self.U_coarse_graph_obj= self.create_digraph(self.U, 'U')
    # self.U_graph_obj, self.U_map_x_to_node, self.U_map_b_to_node, self.U_map_nz_idx_to_node, self.U_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('U', 'chain')

    if verify:
      b= np.array([1.0 for _ in range(self.ncols)], dtype= 'double')
      self.lin_solve(b, verify= True)
  
  def statistics(self, write_files= False, file_path= None):
    if write_files:
      assert file_path != None

    stat_obj_L= Matrix_statistics(self.mtx_file_name)
    stat_obj_L.U_or_L= 'L'
    stat_obj_L.nrows, stat_obj_L.ncols= self.L.shape
    stat_obj_L.nnz= self.L.count_nonzero()

    stat_obj_U= Matrix_statistics(self.mtx_file_name)
    stat_obj_U.U_or_L= 'U'
    stat_obj_U.nrows, stat_obj_U.ncols= self.U.shape
    stat_obj_U.nnz= self.U.count_nonzero()

    self.L_coarse_graph_obj= self.create_digraph(self.L, 'L')
    self.U_coarse_graph_obj= self.create_digraph(self.U, 'U')
    stat_obj_L.critical_path_len_coarse= nx.algorithms.dag.dag_longest_path_length(self.L_coarse_graph_obj.graph_nx)
    stat_obj_U.critical_path_len_coarse= nx.algorithms.dag.dag_longest_path_length(self.U_coarse_graph_obj.graph_nx)

    # self.L_graph_obj, self.L_map_x_to_node, self.L_map_b_to_node, self.L_map_nz_idx_to_node, self.L_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('L', 'tree')

    # self.U_graph_obj, self.U_map_x_to_node, self.U_map_b_to_node, self.U_map_nz_idx_to_node, self.U_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('U', 'tree')
    # stat_obj_L.critical_path_len_fine_tree= nx.algorithms.dag.dag_longest_path_length(self.L_graph_obj.graph_nx)
    # stat_obj_U.critical_path_len_fine_tree= nx.algorithms.dag.dag_longest_path_length(self.U_graph_obj.graph_nx)
    
    # self.L_graph_obj, self.L_map_x_to_node, self.L_map_b_to_node, self.L_map_nz_idx_to_node, self.L_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('L', 'chain')

    # self.U_graph_obj, self.U_map_x_to_node, self.U_map_b_to_node, self.U_map_nz_idx_to_node, self.U_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('U', 'chain')
    # stat_obj_L.critical_path_len_fine_chain= nx.algorithms.dag.dag_longest_path_length(self.L_graph_obj.graph_nx)
    # stat_obj_U.critical_path_len_fine_chain= nx.algorithms.dag.dag_longest_path_length(self.U_graph_obj.graph_nx)

    # self.L_graph_obj, self.L_map_x_to_node, self.L_map_b_to_node, self.L_map_nz_idx_to_node, self.L_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('L', 'hybrid')
    
    # b= np.array([1.0 for _ in range(self.ncols)], dtype= 'double')
    # self.tr_solve(b, 'L', verify= True)

    # self.U_graph_obj, self.U_map_x_to_node, self.U_map_b_to_node, self.U_map_nz_idx_to_node, self.U_map_r_to_nodes_info= \
    #     self.create_arithmetic_graph_directly('U', 'hybrid')
    # b= np.array([1.0 for _ in range(self.ncols)], dtype= 'double')
    # self.tr_solve(b, 'U', verify= True)
    
    stat_obj_L.critical_path_len_fine_hybrid= nx.algorithms.dag.dag_longest_path_length(self.L_graph_obj.graph_nx)
    # stat_obj_U.critical_path_len_fine_hybrid= nx.algorithms.dag.dag_longest_path_length(self.U_graph_obj.graph_nx)

    stat_str = self.mtx_file_name + ','
    stat_str += f"{self.nrows},{self.ncols},{self.nnz},"
    stat_str += stat_obj_L.get_str()
    stat_str += stat_obj_U.get_str()

    logger.info(stat_str)
    if write_files:
      with open(file_path, 'a') as f:
        print(stat_str, file=f)
  
  def unsigned_arithmetic(self, matrix):
    # nonzero() is a slower approach
    # convert to coo matrix for better speed
    rows,cols = matrix.nonzero()
    for row,col in zip(rows,cols):
      if row != col:
        matrix[row, col] = -abs(matrix[row, col])
      else:
        matrix[row, col] = abs(matrix[row, col])

  def create_digraph(self, matrix, U_or_L, start_from= None):
    """
      if start_from == None, node id will match row id
    """
    assert U_or_L in ['U', 'L']

    graph_nx= nx.convert_matrix.from_scipy_sparse_matrix(matrix)
    graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx))

    if start_from != None:
      graph_nx, _ = useful_methods.relabel_nodes_with_contiguous_numbers(graph_nx, start=start_from)
      # graph_obj= common_classes.GraphClass(id_iter= itertools.count(start_from))
    # else:
      # graph_obj= common_classes.GraphClass(id_iter= itertools.count(0))

    G= nx.DiGraph()
    G.add_nodes_from(graph_nx.nodes())
    for e in graph_nx.edges():
      e_new= sorted(list(e))
      G.add_edge(e_new[0], e_new[1])
    
    if U_or_L== 'U':
      G= G.reverse()
    
    topological_list= list(nx.algorithms.dag.topological_sort(G))

    graph_obj= common_classes.GraphClass(id_iter= None)
    for n in topological_list:
      graph_obj.create_dummy_node(n)
      for c in G.predecessors(n):
        graph_obj.add_parent_child_edge(n, c)
      
    graph_obj.graph_nx= G

    assert nx.algorithms.dag.is_directed_acyclic_graph(graph_obj.graph_nx)

    return graph_obj

  def read_mm_file(self, name):
    """
      NOTE: returns a scipy coo file and not a csr file
    """
    path= self.global_var.DATA_PATH + name
    logger.info(f"reading {name} matrix from an mtm file at {path}")
    m= scipy.io.mmread(path)
    m= m.tocsr()
    return m
    
  def write_mm_file(self, matrix, name):
    assert name != self.mtx_file_name
    path= self.global_var.DATA_PATH + name
    logger.info(f"writing {name} matrix to a mtm file at {path}")
    scipy.io.mmwrite(path, matrix)

  def lin_solve(self, b, verify= False):
    "finds x such that Ax=b via arithmetic graphs of L and U factors"

    logger.info("Linear solve")

    perm_b= (self.Pr * csr_matrix(b).transpose()).transpose()
    perm_b= perm_b.A[0]
    x= self.tr_solve(perm_b, 'L', verify)
    x= self.tr_solve(x, 'U', verify)
    x= (self.Pc * csr_matrix(x).transpose()).transpose()
    x= x.A[0]
    assert len(x) == len(b)

    if verify:
      x_golden= linalg.spsolve(self.A, b)
      x_golden_2= self.lu.solve(b)
      assert(np.allclose(b, self.A * csr_matrix(x).transpose().A))
      assert(np.allclose(x_golden, x_golden_2))
      assert(np.allclose(x_golden, x))

    logger.info(f"First 10 elements of the lin_solve solution: {x[:10]}")
    return x

  def tr_solve(self, b, U_or_L, verify= False):
    logger.info("Triangular solve")
    assert U_or_L in ['U', 'L']
    assert len(b) == self.nrows

    if U_or_L == 'U':
      target_M= self.U
      graph_obj= self.U_graph_obj
      map_x_to_node= self.U_map_x_to_node
      map_b_to_node= self.U_map_b_to_node
    elif U_or_L == 'L':
      target_M= self.L
      graph_obj= self.L_graph_obj
      map_x_to_node= self.L_map_x_to_node
      map_b_to_node= self.L_map_b_to_node
    else:
      assert 0

    self.instantiate_b(U_or_L, b)

    ac_eval.ac_eval_non_recurse(graph_obj.graph, graph_obj.graph_nx)

    # get output
    x=[]
    for idx in range(self.ncols):
      node= map_x_to_node[idx]
      x.append(graph_obj.graph[node].curr_val)
    x= np.array(x)

    # verify
    if verify:
      x_golden= linalg.spsolve_triangular(target_M, b, lower= (U_or_L == 'L'))
      assert(np.allclose(x_golden, x))

    return x

  def create_imbalanced_tree(self, r, map_v_to_slack, map_v_to_reverse_lvl, pred):
    curr_slack = map_v_to_slack[r]
    ref_lvl= map_v_to_reverse_lvl[r]
    sorted_pred= sorted(pred, key= lambda x: map_v_to_reverse_lvl[x], reverse= True)
    
    map_curr_pred_to_slack= {}
    list_of_tree_leaves_list= []

    while sorted_pred:
      critical_pred= sorted_pred[0]
      assert map_v_to_reverse_lvl[critical_pred] < ref_lvl

      slack= curr_slack + (ref_lvl - map_v_to_reverse_lvl[critical_pred]) - 1

      # slack may become negative, this because of the way map_v_to_slack is computed
      if slack < 0: 
        slack = 0
      
      # option_1: all pred with same lvl
      # same_lvl_pred= [p for p in sorted_pred if map_v_to_reverse_lvl[p] == map_v_to_reverse_lvl[critical_pred]]
      same_lvl_pred= []
      for p in sorted_pred:
        if map_v_to_reverse_lvl[p] == map_v_to_reverse_lvl[critical_pred]:
          same_lvl_pred.append(p)
        else:
          break
      
      # power of 2 as that is critical length of binary tree
      if len(same_lvl_pred) >= 2**slack:
        tree_preds= same_lvl_pred
      else:
        # option 2: preds according to slack
        tree_preds= sorted_pred[ : 2**slack]
        assert len(tree_preds) <= 2**slack

      # update variables to return
      list_of_tree_leaves_list.append(tree_preds)
      pred_slack= slack - useful_methods.clog2(len(tree_preds))
      for p in tree_preds:
        map_curr_pred_to_slack[p] = pred_slack
      
      # update sorted_pred for next iteration
      next_sorted_pred = sorted_pred[len(tree_preds) :]
      assert len(set(next_sorted_pred) | set(tree_preds)) == len(sorted_pred)
      assert len(set(next_sorted_pred) & set(tree_preds)) == 0
      sorted_pred = next_sorted_pred

      # the next tree in the chain will have to face one extra addition
      curr_slack -= 1

    return list_of_tree_leaves_list, map_curr_pred_to_slack


  def conservative_slack_on_every_node(self, map_v_to_reverse_lvl, graph_nx):
    """
      Defines slack on every node of coarse graph, 
      that can be used to unroll in a tree without affecting the critical length
    """

    critical_path_len= max(list(map_v_to_reverse_lvl.values()))
    assert critical_path_len == nx.algorithms.dag.dag_longest_path_length(graph_nx)
    
    topological_list= list(nx.algorithms.dag.topological_sort(graph_nx))

    map_v_to_slack= {n: critical_path_len - map_v_to_reverse_lvl[n] for n in graph_nx}

    # successors before predecessors
    for n in reversed(topological_list):
      _, map_curr_pred_to_slack= self.create_imbalanced_tree(n, map_v_to_slack, map_v_to_reverse_lvl, list(graph_nx.predecessors(n)))
      for p, s in map_curr_pred_to_slack.items():
        if s < map_v_to_slack[p]:
          map_v_to_slack[p] = s

    # resulting slack can be negative or positive bacause reverse_lvl does not take into 
    # account the numbers of MAC to be performed in each coarse node.
    # it just makes sure the lvl of successor is +1 lvl of all predecessors
    # making it 0 is complicated, because that would need another reassignment of slack with a new critical_path_len
    # Following cannot be asserted
    min_slack=  min(list(map_v_to_slack.values()))
    assert min_slack <= 0
    assert max([map_v_to_slack[n] for n in nx.algorithms.dag.dag_longest_path(graph_nx)]) <= 0

    return map_v_to_slack

  def create_arithmetic_graph_directly(self, mode, reduction_mode):
    logger.info(f"creating_arithmetic_graph for {mode} triangular matrix with reduction_mode: {reduction_mode}")
    assert mode in ['U', 'L'] # upper triangle or lower triangle
    assert reduction_mode in ['tree', 'chain', 'hybrid']

    # dict that maps non-zero matrix variable to input nodes of graph
    # key: (row, col)
    # val: node id in the graph
    map_nz_idx_to_node={}
    
    # map variables of b vector in Ax=b to a node in the graph
    # key: idx in b
    # val: node id in the graph
    map_b_to_node={}

    # map variables of x vector in Ax=b to a node in the graph
    # key: idx in x
    # val: node id in the graph
    # Thesre are the outputs of the computation
    map_x_to_node={}

    # map row/col to nodes
    # key: row/col idx depending on U or L
    # val: obj of class Coarse_node_mappings
    map_r_to_nodes_info= {}

    start= time.time()
    if mode == 'U':
      # take anti-transpose to convert to an equivalent L
      #target_M= np.rot90(self.U.A,2).T
      target_M= self.U
      col_ptrs_ls= target_M.tolil().rows
      # iterate on 'U' in a reverse order
      iterate_list= reversed(list(enumerate(col_ptrs_ls)))
    elif mode == 'L':
      target_M= self.L
      col_ptrs_ls= target_M.tolil().rows
      iterate_list= enumerate(col_ptrs_ls)
    else:
      assert 0
    # logger.info(f"A: {time.time() - start}")

    start= time.time()
    if reduction_mode == 'hybrid' or reduction_mode == 'chain':
      graph_nx= self.create_digraph(target_M, mode).graph_nx
      map_v_to_reverse_lvl= useful_methods.compute_reverse_lvl(graph_nx)
    else:
      map_v_to_reverse_lvl= None
    # logger.info(f"B1: {time.time() - start}")

    start= time.time()
    if reduction_mode == 'hybrid':
      map_v_to_slack= self.conservative_slack_on_every_node(map_v_to_reverse_lvl, graph_nx)
    else:
      map_v_to_slack= None
    # logger.info(f"B2: {time.time() - start}")

    # row-wise list of column indices that have non-zero values

    logger.info("build graph")
    start= time.time()
    graph_obj= common_classes.GraphClass(id_iter= itertools.count())
    time_matrix= [0]
    time_rest= [0]
    for r, col_ptrs in iterate_list:
      assert len(col_ptrs) != 0 # there is atleast one non-zero in this row
      if mode == 'U':
        assert min(col_ptrs) <= r # matrix is infact L
      elif mode == 'L':
        assert max(col_ptrs) <= r # matrix is infact L
      else:
        assert 0

      row_prod_nodes, reduction_nodes, x_key, b_key, _, _ = \
        self.create_arithmetic_graph_single_row(reduction_mode, graph_obj, r, target_M, map_x_to_node, map_b_to_node, map_nz_idx_to_node, time_matrix, time_rest,
                map_v_to_reverse_lvl, map_v_to_slack)

      coarse_node_info= Coarse_node_mappings(r, reduction_nodes, row_prod_nodes, x_key, b_key)
      map_r_to_nodes_info[r] = coarse_node_info

    # logger.info(f"C: {time.time() - start}")
    # logger.info(f"C1: {time_matrix[0]}")
    # logger.info(f"C2: {time_rest[0]}")

    if self.output_mode == 'single_node':
      logger.info(f"output_mode: {self.output_mode}, adding dummy nodes in case the graph does not have a single output node")

      # nodes with node parents
      output_nodes = [n for n, obj in graph_obj.graph.items() if len(obj.parent_key_list) == 0]
      assert len(output_nodes) != 0

      graph_len_before_dummy_nodes= len(graph_obj.graph)
      if len(output_nodes) > 1:
        graph_obj.create_tree_of_nodes(output_nodes, 'sum')

        logger.info(f"{len(graph_obj.graph) - graph_len_before_dummy_nodes} dummy nodes added because there were {len(output_nodes)} output_nodes earlier")
        logger.info(f"{len(graph_obj.graph) - graph_len_before_dummy_nodes} dummy nodes out of {len(graph_obj.graph)} total nodes. Percentage: {(len(graph_obj.graph) - graph_len_before_dummy_nodes) *100 / len(graph_obj.graph)}")
      else:
        logger.info("Dummy nodes not needed")

    graph_obj.create_graph_nx()

    logger.info(f"nnz of the {mode} triangular matrix: {target_M.count_nonzero()}, Length of the graph: {len(graph_obj.graph)}")

    for r, _ in iterate_list:
      assert map_r_to_nodes_info[r].final_prod_node == map_x_to_node[r]
    
    return graph_obj, map_x_to_node, map_b_to_node, map_nz_idx_to_node, map_r_to_nodes_info
  
  def create_arithmetic_graph_single_row(self, mode, graph_obj, r, matrix, map_x_to_node, map_b_to_node, map_nz_idx_to_node, time_matrix, time_rest, map_v_to_reverse_lvl= None, map_v_to_slack= None):
    assert mode in ['chain', 'tree', 'hybrid']
    
    b_key= graph_obj.create_real_valued_leaf_node()
    map_b_to_node[r]= b_key

    start= time.time()
    row_matrix= matrix.getrow(r)
    row_indices= row_matrix.indices
    row_data=  row_matrix.data
    time_matrix[0] += time.time() - start
    
    start= time.time()
    # key: column idx c
    row_prods={}
    row_prods_ls= [] # ls is needed because the order is important in create_chain_of_nodes

    for i, c in enumerate(row_indices):
      data= row_data[i]
      if r!= c: # except diagonal
        child_0= graph_obj.create_real_valued_leaf_node(val= -data) # negative of the actual val
        map_nz_idx_to_node[(r,c)] = child_0

        child_1= map_x_to_node[c]
        prod_node= graph_obj.create_2input_node(child_0, child_1, 'prod')
        row_prods[c]= prod_node
        row_prods_ls.append(prod_node)
      else:
        assert data != 0
        diag_elem= graph_obj.create_real_valued_leaf_node(val= 1.0/data) # inverse of diag 
        map_nz_idx_to_node[(r,r)]= diag_elem

    reduction_nodes= set()
    reduction_graph_nx= nx.DiGraph()
    if mode== 'tree':
      reduction_head= graph_obj.create_tree_of_nodes([b_key] + row_prods_ls, 'sum', reduction_nodes, reduction_graph_nx)
    elif mode == 'chain':
      assert map_v_to_reverse_lvl != None
      pred= [c for c in row_indices if r != c]
      sorted_pred= sorted(pred, key= lambda x: map_v_to_reverse_lvl[x], reverse= True)
      chain_leaves = [b_key] + [row_prods[p] for p in reversed(sorted_pred)]
      # if row_indices[0] == r: # U matrix
      #   # order is reversed so that the resulting chain 
      #   # will minimize the critical path of the triangular solve
      #   chain_leaves= [b_key] + list(reversed(row_prods_ls))
      # elif row_indices[-1] == r: # L matrix
      #   chain_leaves= [b_key] + row_prods_ls
      # else:
      #   assert 0
      reduction_head= graph_obj.create_chain_of_nodes(chain_leaves, 'sum', reduction_nodes, reduction_graph_nx)
      if len(pred) == 0:
        assert reduction_head == b_key

    elif mode == 'hybrid':
      assert map_v_to_reverse_lvl != None
      assert map_v_to_slack != None

      curr_slack = map_v_to_slack[r]
      ref_lvl= map_v_to_reverse_lvl[r]
      pred= [c for c in row_indices if r != c]

      list_of_tree_leaves_list, _ = self.create_imbalanced_tree(r, map_v_to_slack, map_v_to_reverse_lvl, pred)
      time_rest[0] += time.time() - start
      tree_heads= []
      for tree_preds in list_of_tree_leaves_list:
        tree_leaves= [row_prods[p] for p in tree_preds]
        tree_head= graph_obj.create_tree_of_nodes(tree_leaves, 'sum', reduction_nodes, reduction_graph_nx)
        tree_heads.append(tree_head)

      chain_leaves = [b_key] + list(reversed(tree_heads)) # reverse because of the way chain is created
      reduction_head= graph_obj.create_chain_of_nodes(chain_leaves, 'sum', reduction_nodes, reduction_graph_nx)

      if len(pred) == 0:
        assert reduction_head == b_key
    else:
      assert 0

    # prod with inverse of the diagonal element
    x_key= graph_obj.create_2input_node(reduction_head, diag_elem,'prod')
    map_x_to_node[r] = x_key
    
    row_prod_nodes= set(row_prods_ls)
    all_nodes= reduction_nodes | set([x_key]) | row_prod_nodes

    return row_prod_nodes, reduction_nodes, x_key, b_key, all_nodes, reduction_graph_nx

  def instantiate_b(self, U_or_L, b= None):
    assert U_or_L in ['U', 'L']

    if U_or_L == 'U':
      graph_obj= self.U_graph_obj
      map_b_to_node= self.U_map_b_to_node
    elif U_or_L == 'L':
      graph_obj= self.L_graph_obj
      map_b_to_node= self.L_map_b_to_node
    else:
      assert 0

    if b is None:
      b= [1.0 for r in range(self.nrows)]

    # instead of directly iterating over dict, 
    # generate fixed sequence of indices based on nrows
    for r in range(self.nrows):
      node= map_b_to_node[r]
      graph_obj.graph[node].curr_val= b[r]

  
  def coarsen_partitions(self, graph, graph_nx, list_of_partitions, map_r_to_nodes_info):
    logger.info("Coarsening partitions")
    N_PE= len(list_of_partitions)
    assert N_PE != 0
    
    n_layers= len(list_of_partitions[0])
    assert n_layers != 0

    total_nodes= set([n for pe in range(N_PE) for l in range(len(list_of_partitions[0])) for n in list_of_partitions[pe][l]])
    n_total_nodes= len(total_nodes) 

    print(len(total_nodes))
    print(len(useful_methods.get_non_leaves(graph_nx)))
    assert total_nodes == set(useful_methods.get_non_leaves(graph_nx))

    logger.info(f"lenght of partitions: {[len(list_of_partitions[pe][l]) for pe in range(N_PE) for l in range(n_layers)]}")

    # reverse map
    map_n_to_r= {}
    for r, coarse_node_info in map_r_to_nodes_info.items():
      for n in coarse_node_info.all_nodes:
        assert n not in map_n_to_r
        map_n_to_r[n] = r
    assert len(map_n_to_r) == n_total_nodes

    tot_coarse_nodes= 0
    id_iter= itertools.count(0)
    map_semi_coarse_to_tup= {}
    map_n_to_semi_coarse= {}
    total_fine_n_mapped= 0
    list_of_partitions_coarse= [[] for _ in range(N_PE)]
    done_nodes= set()
    for pe in range(N_PE):
      for l in range(n_layers):
        curr_partition= set(list_of_partitions[pe][l])
        semi_coarse_n_set= set()
        while len(curr_partition) != 0:
          n= curr_partition.pop() # remember this element is removed!
          r= map_n_to_r[n]
          n_coarse_nodes, nodes_mapped, operation_nodes_tup_ls= map_r_to_nodes_info[r].create_coarse_nodes(graph, graph_nx, curr_partition | set([n]))
          tot_coarse_nodes += n_coarse_nodes
          curr_partition -= nodes_mapped
          done_nodes |= nodes_mapped
          assert n_coarse_nodes == len(operation_nodes_tup_ls)

          nodes_mapped_assert= set()
          for tup in operation_nodes_tup_ls:
            semi_coarse_n= next(id_iter)
            semi_coarse_n_set.add(semi_coarse_n)

            map_semi_coarse_to_tup[semi_coarse_n] = tup
            map_r_to_nodes_info[r].map_semi_coarse_to_tup[semi_coarse_n] = tup
            for fine_n in tup[1]:
              nodes_mapped_assert.add(fine_n)
              assert fine_n not in map_n_to_semi_coarse
              map_n_to_semi_coarse[fine_n] = semi_coarse_n
              map_r_to_nodes_info[r].map_n_to_semi_coarse[fine_n] = semi_coarse_n

          assert nodes_mapped_assert == nodes_mapped
            
          total_fine_n_mapped += len(nodes_mapped)

        list_of_partitions_coarse[pe].append(semi_coarse_n_set)

    assert len(done_nodes) == n_total_nodes

    logger.info(f"total_fine_n_mapped: {total_fine_n_mapped} out of n_total_nodes : {n_total_nodes}")
    assert total_fine_n_mapped == n_total_nodes
    logger.info(f"Totol coarse nodes: {tot_coarse_nodes}, tot rows: {len(map_r_to_nodes_info)}")

    logger.info(f"lenght of partitions coarse: {[len(list_of_partitions_coarse[pe][l]) for pe in range(N_PE) for l in range(n_layers) ]}")

    # create semi coarse graph
    semi_coarse_g= nx.DiGraph()
    semi_coarse_g.add_nodes_from(list(map_semi_coarse_to_tup.keys()))
    done_edges= set()
    done_coarse_nodes= set()
    for u,v in graph_nx.edges():
      if not graph[u].is_leaf():
        coarse_u= map_n_to_semi_coarse[u]
        coarse_v= map_n_to_semi_coarse[v]
        if coarse_u != coarse_v:
          # print(coarse_u, u, v, map_semi_coarse_to_tup[coarse_u])
          ALLOWED= False
          if coarse_u in done_coarse_nodes:
            # print("repeat",coarse_u, u, v, map_semi_coarse_to_tup[coarse_u])
            if map_semi_coarse_to_tup[coarse_u][0] == 'final':
              assert (coarse_u, coarse_v) not in done_edges, f"{coarse_u, coarse_v}, len(done_edges)"
              ALLOWED = True
            else:
              assert map_semi_coarse_to_tup[coarse_v][0] != 'mac'
          else:
            ALLOWED= True

          if ALLOWED:
            semi_coarse_g.add_edge(coarse_u, coarse_v)
            done_edges.add((coarse_u, coarse_v))
            done_coarse_nodes.add(coarse_u)
      else:
        assert u not in map_n_to_semi_coarse

    # remove 0/1 input adder nodes
    removed_nodes= set()
    candidate_nodes= set([n for n in semi_coarse_g.nodes() if map_semi_coarse_to_tup[n][0] == 'add'])
    while candidate_nodes:
      n= candidate_nodes.pop()
      n_type= map_semi_coarse_to_tup[n][0]
      pred= list(semi_coarse_g.predecessors(n))
      if n==1482:
        print(n_type, pred)
      if n_type == 'add' and len(pred) <= 1:
        succ= list(semi_coarse_g.successors(n))
        assert len(succ) == 1
        succ= succ[0]

        if len(pred) == 1:
          semi_coarse_g.add_edge(pred[0], succ)
        else: # one of the predecessor of succ will be deleted, hence it now becomes a candidate node
          if map_semi_coarse_to_tup[succ][0] == 'add':
            candidate_nodes.add(succ)
        
        semi_coarse_g.remove_node(n)
        removed_nodes.add(n)


    # also remove the nodes from list_of_partitions_coarse
    for pe in range(N_PE):
      for l in range(n_layers):
        curr_partition= list_of_partitions_coarse[pe][l] - removed_nodes
        list_of_partitions_coarse[pe][l]= curr_partition

    logger.info(f"size of semi_coarse_g after removing useless add nodes: {len(semi_coarse_g)}")

    # assertions
    assert (len(semi_coarse_g)) != 0
    for coarse_u, coarse_v in semi_coarse_g.edges():
      u_type= map_semi_coarse_to_tup[coarse_u][0]
      v_type= map_semi_coarse_to_tup[coarse_v][0]

      assert not(u_type == 'mac' and v_type == 'mac')
      assert not(u_type == 'add' and v_type == 'mac')
      assert not(u_type == 'final' and v_type == 'add')

      if v_type == 'mac':
        assert u_type == 'final'

    for coarse_n in semi_coarse_g:
      n_type= map_semi_coarse_to_tup[coarse_n][0]
      if n_type == 'add' or n_type == 'mac':
        assert len(list(semi_coarse_g.successors(coarse_n))) != 0
      if n_type == 'add':
        pred= list(semi_coarse_g.predecessors(coarse_n))
        assert len(pred) > 1

    assert nx.algorithms.dag.is_directed_acyclic_graph(semi_coarse_g)

    logger.info(f"Non leaf coarse edges: {semi_coarse_g.number_of_edges()}")

    return list_of_partitions_coarse, semi_coarse_g, map_n_to_semi_coarse, map_r_to_nodes_info, map_n_to_r, map_semi_coarse_to_tup
