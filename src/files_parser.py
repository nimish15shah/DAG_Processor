
import pickle
import re
import scipy.io
import scipy.sparse

def read_mat(path):
  # print(path)
  mat = scipy.io.loadmat(path)
  # print(mat['Problem'])
  for i in range(len(mat['Problem'][0][0])):
    obj= mat['Problem'][0][0][i]
    if scipy.sparse.issparse(obj):
      return obj

def read_evid_db(dat_content, BN):
  """
    Returns a list of dictionaries.
    For each dictionary:
      Key: var_name
      Val: var_state
  """
  
  # remove \n
  dat_content= [i[:-1] for i in dat_content]
  
  dat_db= [] # List of dictionaries. Each dictionary is a map of var_name to var_state

  var_lst= dat_content[0]
  var_lst= var_lst.split(',')
  
  # Assert that var_name is indeed from our BN
  for var in var_lst:
    assert var in BN

  evid_lst= dat_content[1:]

  for evid in evid_lst:
    evid= evid.split(',')
    evid_dict= {}
    for idx, var in enumerate(var_lst):
      state= evid[idx]
      assert state in BN[var].states
      evid_dict[var] = state
    
    dat_db.append(evid_dict)
  
  return dat_db

def read_BB_graph_and_main_graph(global_var, hw_depth, hw_details_str):
  """
    Reads pickled file to exract BB_graph and AC graph
  """
  if len(hw_details_str) > 80:
    hw_details_str= hw_details_str[:80]

  with open(global_var.BB_FILE_PREFIX + str(hw_depth) + str(hw_details_str), 'rb') as f:
    BB_graph= pickle.load(f)
  with open(global_var.BB_FILE_PREFIX + 'nx'+ str(hw_depth) + str(hw_details_str), 'rb') as f:
    BB_graph_nx= pickle.load(f)
  with open(global_var.GRAPH_FILE_PREFIX + str(hw_depth) + str(hw_details_str), 'rb') as f:
    graph= pickle.load(f)
  with open(global_var.GRAPH_FILE_PREFIX + 'nx' + str(hw_depth) + str(hw_details_str), 'rb') as f:
    graph_nx= pickle.load(f)
  with open(global_var.GRAPH_FILE_PREFIX + 'misc' + str(hw_depth) + str(hw_details_str), 'rb') as f:
    misc= pickle.load(f)
  
  return BB_graph, graph, BB_graph_nx, graph_nx, misc

def write_BB_graph_and_main_graph(global_var, hw_depth, hw_details_str, graph, graph_nx, BB_graph, BB_graph_nx, misc):
  if len(hw_details_str) > 80:
    hw_details_str= hw_details_str[:80]
  
  with open(global_var.BB_FILE_PREFIX + str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(BB_graph, f)
  with open(global_var.BB_FILE_PREFIX + 'nx'+str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(BB_graph_nx, f)
  with open(global_var.GRAPH_FILE_PREFIX + str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(graph, f)
  with open(global_var.GRAPH_FILE_PREFIX + 'nx' + str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(graph_nx, f)
  with open(global_var.GRAPH_FILE_PREFIX + 'misc' + str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(misc, f)

def write_schedule(global_var, hw_depth, hw_details_str, instr_ls_obj, map_param_to_addr, map_output_to_addr):
  if len(hw_details_str) > 80:
    hw_details_str= hw_details_str[:80]
  
  with open(global_var.BB_FILE_PREFIX + 'SCH' + str(hw_depth) + hw_details_str, 'wb+') as f:
    pickle.dump(instr_ls_obj, f)

  with open(global_var.BB_FILE_PREFIX + '_param_addr_' + hw_details_str, 'wb+') as f:
    pickle.dump(map_param_to_addr, f)

  with open(global_var.BB_FILE_PREFIX + '_output_addr_' + hw_details_str, 'wb+') as f:
    pickle.dump(map_output_to_addr, f)



def read_schedule(global_var, hw_depth, hw_details_str):
  if len(hw_details_str) > 80:
    hw_details_str= hw_details_str[:80]
  
  with open(global_var.BB_FILE_PREFIX + 'SCH' + str(hw_depth) + hw_details_str, 'rb') as f:
    instr_ls_obj= pickle.load(f)

  with open(global_var.BB_FILE_PREFIX + '_param_addr_' + hw_details_str, 'rb') as f:
    map_param_to_addr = pickle.load(f)

  with open(global_var.BB_FILE_PREFIX + '_output_addr_' + hw_details_str, 'rb') as f:
    map_output_to_addr= pickle.load(f)

  return instr_ls_obj, map_param_to_addr, map_output_to_addr

def read_logistic_circuit(global_var):
  """
    Reads logistic circuit file
    Does some pre-processing to remove unnecessary material

    Returns: A list of lists.
    Each inner list has a format: 

    [T, id-of-true-literal-node, variable, parameter]
    [F, id-of-false-literal-node, variable, parameter]
    [D, id-of-or-gate, number-of-elements, (id-of-prime id-of-sub parameter)*]
    [B, bias_parameter]

    Note that AND-gates are not seperately encoded. They are an element in the description of OR-Gate (D)

  """
  with open(global_var.LOGISTIC_CIRCUIT_FILE, 'rb') as f:
    lines= f.readlines()

  # remove '\n'
  lines= [i[:-1] for i in lines]
  
  # Remove comments
  lines= [i for i in lines if i[0] != 'c']
  
  
  # Split according to brackets
  bracket_re= re.compile('\([^\)]*\)')
  
  for idx, i in enumerate(lines):
    # Convert strings to numbers in T and F
    if i[0] == 'T' or i[0] == 'F':
      i = i.split(' ')
      i[1] = int(i[1])
      #i[2] = int(i[2])  # Variable name should be string
      i[3] = float(i[3])
      lines[idx]= i

    if i[0] == 'D':
      AND_gates= bracket_re.findall(i)
      
      for idx_j, gates in enumerate(AND_gates):
        gates= gates.strip("()")
        gates= gates.split(" ")
        gates[0]= int(gates[0].strip(","))
        gates[1]= int(gates[1].strip(","))
        gates[2]= float(gates[2])
        AND_gates[idx_j]= gates

      rest= bracket_re.sub("", i)
      rest= rest.strip(" ")
      rest= rest.split(" ")
      
      rest[1]= int(rest[1])
      rest[2]= int(rest[2])

      full= rest + AND_gates

      assert full[2] == len(AND_gates), "Number of elements do not match with the AND gates"
      
      lines[idx]= full
    
    if i[0] == 'B':
      i= i.split(' ') 
      i[1] = float(i[1])
      lines[idx]= i
  
  return lines

def modify_logistic_circuit():
  """
    Preprocess new format to remove v_tree id
  """
  
  with open('../benchmarks/mnist.circuit_985', 'r') as f:
    lines= f.readlines()
  
  lines_m=[]
  for idx, i in enumerate(lines):
    if i[0] == 'T' or i[0] == 'F' or i[0] == 'D':
      i= i.split(' ')      
      del i[2]
      seperator= ' '
      i = seperator.join(i)

    lines_m.append(i)

    
  with open('../benchmarks/mnist_985.txt', 'w+') as f:
    f.writelines(lines_m)
  

def read_dataset(global_var, network, dataset_type):
  assert dataset_type in ['train', 'test', 'valid']

  if dataset_type == 'train':
    dataset_path= global_var.DATASET_PATH_PREFIX + network + '/' + network + '.train.data'
  elif dataset_type == 'test':
    dataset_path= global_var.DATASET_PATH_PREFIX + network + '/' + network + '.test.data'
  elif dataset_type == 'valid':
    dataset_path= global_var.DATASET_PATH_PREFIX + network + '/' + network + '.valid.data'
  else:
    assert 0

  with open(dataset_path, 'r') as f:
    dataset= f.readlines()
    dataset_formatted=[]
    for line in dataset:
      line= line.strip()
      line= line.split(",")
      line= [int(l) for l in line]
      dataset_formatted.append(line)
  
  dataset = dataset_formatted
  
  return dataset        
