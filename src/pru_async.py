import logging as log

from .useful_methods import clog2, format_hex, printcol
from .new_arch import memory_allocation
from .util import convert_data, custom_precision_operation


def main(global_var, graph, status_dict, list_of_schedules, config_obj):
  init_data_class_obj= init_data_c()

  output_data(global_var, graph, status_dict, init_data_class_obj, config_obj)

  output_program(global_var, list_of_schedules, status_dict, init_data_class_obj, config_obj)

  config_data(global_var, init_data_class_obj, config_obj)

  #print_instr_details(list_of_schedules, hw_details)

def print_instr_details(list_of_schedules, hw_details):
  N_PE = hw_details.N_PE
  
  for pe, pe_schedules in enumerate(list_of_schedules):
    printcol("New PE", 'green')
    for global_barrier_idx, pe_schedule in enumerate(pe_schedules):
      for instr in pe_schedule:
        print(pe, instr.operation, end=' ')
        if instr.to_load_0:
          print("load 0:", instr.load_0_addr.bank, instr.load_0_addr.pos, instr.load_0_addr.store_in_local_mem, end=' ')
        if instr.to_load_1:
          print("load 1:", instr.load_1_addr.bank, instr.load_1_addr.pos, instr.load_1_addr.store_in_local_mem, end=' ')
        if instr.to_store:
          print("store:", instr.store_addr.bank, instr.store_addr.pos, instr.store_addr.store_in_local_mem, end=' ')
        
        print(" ")
      printcol("Global barrier : " + str(global_barrier_idx), "red")

def simulate_instr_async(graph, global_var, final_output_nodes, status_dict, list_of_schedules, config_obj):
  hw_details= config_obj.hw_details
  N_PE = hw_details.N_PE
  
  #processor state
  reg = [{} for _ in range(N_PE)]
  local_mem = [{} for _ in range(N_PE)]
  global_mem = [{} for _ in range(N_PE)]

  debug_map_node_to_val= {}

#  print graph.keys()

  # load leaves
  for node, obj in list(graph.items()):
    if obj.is_leaf():
      val= convert_data(obj.curr_val, hw_details, mode='to')
      for idx, bank in enumerate(status_dict[node].bank):
        mem_obj= memory_allocation.create_mem_obj(node, status_dict, bank)
        #store(status_dict[node], local_mem, global_mem, val)
        store(mem_obj, local_mem, global_mem, val)
      debug_map_node_to_val[node]= val

  n_global_barriers= len(list_of_schedules[0])
  # Itegrate over all PEs before crossing global barriers
  for global_barrier_idx in range(n_global_barriers): 
    for pe, pe_schedule in enumerate(list_of_schedules):
      schedule = pe_schedule[global_barrier_idx]
      for instr in schedule:
        out= None
        if instr.operation == 'sum' or instr.operation == 'prod':
          in_0= reg[pe][instr.reg_in_0]
          in_1= reg[pe][instr.reg_in_1]

          out= custom_precision_operation(in_0, in_1, hw_details, instr.operation)
          #if instr.operation == 'sum': 
          #  out= in_0 + in_1
          #elif instr.operation == 'prod': 
          #  out= in_0 * in_1
          
#          print pe, instr.operation, in_0, in_1, out, hex(convert_data(in_0, hw_details, mode='to')), hex(convert_data(in_1, hw_details, mode='to')), hex(convert_data(out, hw_details, mode='to'))
#          print pe, instr.operation, hex(in_0), hex(in_1), hex(out), convert_data(in_0, hw_details, mode='from'), convert_data(in_1, hw_details, mode='from'), convert_data(out, hw_details, mode='from')
#          print instr.node, instr.in_0_node, instr.in_1_node, out, instr.operation, instr.reg_o, instr.reg_in_0, instr.reg_in_1
          assert debug_map_node_to_val[instr.in_0_node] == in_0, [instr.in_0_node, debug_map_node_to_val[instr.in_0_node], in_0]
          assert debug_map_node_to_val[instr.in_1_node] == in_1, [instr.in_1_node, debug_map_node_to_val[instr.in_1_node], in_1]

          debug_map_node_to_val[instr.node] = out

          reg[pe][instr.reg_o] = out

        if instr.to_load_0:
          load(instr.load_0_addr, reg, local_mem, global_mem, pe, instr.load_0_reg)

        if instr.to_load_1:
          load(instr.load_1_addr, reg, local_mem, global_mem, pe, instr.load_1_reg)

        if instr.to_store:
          assert out != None
          store(instr.store_addr, local_mem, global_mem, out)
#    print "Global barrier: ", global_barrier_idx

  # get the output out
  final_val_dict= {n: read(status_dict[n], local_mem, global_mem) for n in final_output_nodes}
  
  if config_obj.write_files:
    lines= []
    for n in final_output_nodes:
      addr= construct_init_data_mem_addr(status_dict[n], hw_details)
      final_val_replicated= replicate_val(final_val_dict[n], hw_details)
      line= format_hex(addr, 32) + ' ' + format_hex(final_val_replicated, 32) + '\n'
      lines.append(line)
    
    fname= path_prefix(global_var, config_obj) + '_golden.txt'
    with open(fname, 'w+') as f:
      f.writelines(lines)
  
  final_val_normal = {n: float(convert_data(final_val_dict[n] , hw_details, mode='from')) for n in final_output_nodes}
  return final_val_normal

def read(mem_obj, local_mem, global_mem):
  bank= mem_obj.bank
  pos= mem_obj.pos
  
  if mem_obj.store_in_local_mem:
    val = local_mem[bank][pos]
  elif mem_obj.store_in_global_mem:
    val = global_mem[bank][pos]
  else:
    assert 0
  
  return val

def load(mem_obj, reg, local_mem, global_mem, pe, reg_pos):  
  """
    mem_obj is of type mem_alloc_detail_class
  """
  
  reg[pe][reg_pos] = read(mem_obj, local_mem, global_mem)

def store(mem_obj, local_mem, global_mem, val):
  """
    mem_obj is of type mem_alloc_detail_class
  """
  bank= mem_obj.bank
  pos= mem_obj.pos
  
  if mem_obj.store_in_local_mem:
    local_mem[bank][pos] = val
  elif mem_obj.store_in_global_mem:
    global_mem[bank][pos] = val
  else:
    assert 0

class const():
  def __init__(self, hw_details):
    # data mem addr space
    self.GLOBAL_MEM_ADDR_L = clog2(hw_details.GLOBAL_MEM_DEPTH) + clog2(hw_details.N_PE)

    self.LOCAL_MEM_INDICATOR_S= self.GLOBAL_MEM_ADDR_L
    self.LOCAL_MEM_INDICATOR_L= 1
    self.LOCAL_MEM_INDICATOR = 0 # An address should go to local mem if the MSB bit of addr is set to it (in a data addr), 
                                 # not to be used during init

    # stream addr space
    self.STREAM_MAX_ADDR_L_PER_BANK= hw_details.STREAM_MAX_ADDR_L_PER_BANK
    self.STREAM_ID_START= self.STREAM_MAX_ADDR_L_PER_BANK
    self.STREAM_ID_L= clog2(hw_details.N_PE)
    self.STREAM_TYPE_START = self.STREAM_ID_START + self.STREAM_ID_L
    self.STREAM_TYPE_L= 2
    
    self.STREAM_TYPE_INSTR= 0
    self.STREAM_TYPE_LD= 1
    self.STREAM_TYPE_ST= 2

    # init addr space
    self.INIT_ADDR_TYPE_L= 2
    self.INIT_ADDR_TYPE_START= 32 - self.INIT_ADDR_TYPE_L
    self.INIT_ADDR_TYPE_STREAM= 0
    self.INIT_ADDR_TYPE_GLOBAL= 1
    self.INIT_ADDR_TYPE_LOCAL = 2

    # instruction encoding
    self.REG_ADDR_L          = clog2(hw_details.REGBANK_L)
 
    self.OPCODE_L            = 4
    self.NOP_OPCODE               = 0
    self.SUM_OPCODE               = 1
    self.PROD_OPCODE              = 2
    self.LOCAL_BARRIER_OPCODE     = 3
    self.GLOBAL_BARRIER_OPCODE    = 4
    self.SET_LD_STREAM_LEN_OPCODE = 5
    self.PASS_OPCODE              = 6

    self.INSTR_L                  = self.OPCODE_L + 3*self.REG_ADDR_L + 3

    # Config
    self.PRECISION_CONFIG_32B= 0
    self.PRECISION_CONFIG_16B= 1
    self.PRECISION_CONFIG_8B = 2
      
class init_data_c():
  def self(self):
    # config data
    self.config_instr_stream_start_addr_io = None
    self.config_instr_stream_end_addr_io = None

    self.config_ld_stream_start_addr_io = None
    self.config_ld_stream_end_addr_io = None

    self.config_st_stream_start_addr_io = None
    self.config_st_stream_end_addr_io = None
    
    self.config_precision_config= None
    
    # init 
    self.init_leaf_values= None

    self.init_stream_instr= None
    self.init_stream_ld= None
    self.init_stream_st= None
  
def replicate_val(val, hw_details):
  """
    for multi-precision mode
  """
  assert (val >> hw_details.TOT_L) == 0, "0x{:0x} { }".format(val, hw_details.TOT_L)
  concat_val = 0
  for i in range(32//hw_details.TOT_L):
    concat_val= (concat_val << hw_details.TOT_L) | val
  
  return concat_val

# Output for SV verification
def output_data(global_var, graph, status_dict, init_data_class_obj, config_obj):
  """
    Format:  memory addr <space> value
  """
  wr_lines= []
  init_leaf_values= []

  for node, obj in list(graph.items()):
    if obj.is_leaf():
      val= convert_data(obj.curr_val, config_obj.hw_details, mode= 'to')
      val= replicate_val(val, config_obj.hw_details)
      for idx, bank in enumerate(status_dict[node].bank):
        mem_obj= memory_allocation.create_mem_obj(node, status_dict, bank)
        #mem_obj= status_dict[node]
        addr= construct_init_data_mem_addr(mem_obj, config_obj.hw_details)

        #print hex(addr), mem_obj.bank, mem_obj.pos, mem_obj.store_in_local_mem, mem_obj.store_in_global_mem
        
        init_leaf_values.append((addr, val))
        

        line= format_hex(addr, 32) + ' ' + format_hex(val, 32) + '\n'
        
        wr_lines.append(line)

  if config_obj.write_files:
    fname= path_prefix(global_var, config_obj) + '_data.txt'
    with open(fname, 'w+') as f:
      f.writelines(wr_lines)
  
  init_data_class_obj.init_leaf_values = init_leaf_values

def construct_init_data_mem_addr(mem_obj, hw_details):
  """
    mem_obj can be of type status_node or mem_alloc_detail_class
  """
  const_o= const(hw_details)

  N_PE = hw_details.N_PE
  LOCAL_MEM_DEPTH= hw_details.LOCAL_MEM_DEPTH
  GLOBAL_MEM_DEPTH= hw_details.GLOBAL_MEM_DEPTH
  
  LOCAL_POS_BITS= clog2(LOCAL_MEM_DEPTH)
  GLOBAL_POS_BITS= clog2(GLOBAL_MEM_DEPTH)
  TYPE_SHIFT= 32 - const_o.INIT_ADDR_TYPE_L

  bank= mem_obj.bank
  pos= mem_obj.pos
  
  if mem_obj.store_in_local_mem:
    addr= (const_o.INIT_ADDR_TYPE_LOCAL << TYPE_SHIFT) | (bank << LOCAL_POS_BITS) | pos 
  elif mem_obj.store_in_global_mem:
    addr= (const_o.INIT_ADDR_TYPE_GLOBAL << TYPE_SHIFT) | (bank << GLOBAL_POS_BITS) | pos 
  else:
    assert 0
  
  return addr

def output_program(global_var, list_of_schedules, status_dict, init_data_class_obj, config_obj):
  init_stream_ld, file_lines_ld= ld_streams(list_of_schedules, config_obj.hw_details)
  init_stream_st, file_lines_st= st_streams(list_of_schedules, config_obj.hw_details)
  init_stream_instr, file_lines_instr= instr_streams(list_of_schedules, config_obj.hw_details)

  init_data_class_obj.init_stream_instr = init_stream_instr
  init_data_class_obj.init_stream_ld    = init_stream_ld   
  init_data_class_obj.init_stream_st    = init_stream_st   

  if config_obj.write_files:
    fname= path_prefix(global_var, config_obj) + '_ld_stream.txt'
    with open(fname, 'w+') as f:
      f.writelines(file_lines_ld)

    fname= path_prefix(global_var, config_obj) + '_st_stream.txt'
    with open(fname, 'w+') as f:
      f.writelines(file_lines_st)

    fname= path_prefix(global_var, config_obj) + '_instr_stream.txt'
    with open(fname, 'w+') as f:
      f.writelines(file_lines_instr)
    
    op_count, _, _, _ = count_statistics(init_stream_instr, config_obj.hw_details)
    fname= path_prefix(global_var, config_obj) + '_op_count.txt'
    with open(fname, 'w+') as f:
      f.write(str(op_count))


def construct_init_stream_addr(pe_id, pos, hw_details, stream_type):
  assert stream_type in ['instr', 'ld', 'st']
  const_o= const(hw_details)
  
  addr= (pe_id << const_o.STREAM_ID_START) | pos
  
  if stream_type == 'instr':
    addr |= (const_o.STREAM_TYPE_INSTR << const_o.STREAM_TYPE_START)
  elif stream_type == 'ld':
    addr |= (const_o.STREAM_TYPE_LD << const_o.STREAM_TYPE_START)
  elif stream_type == 'st':
    addr |= (const_o.STREAM_TYPE_ST << const_o.STREAM_TYPE_START)
  else:
    assert 0
  
  addr |= (const_o.INIT_ADDR_TYPE_STREAM << const_o.INIT_ADDR_TYPE_START)
  
  return addr

def construct_runtime_data_addr(mem_obj, hw_details):
  N_PE = hw_details.N_PE
  LOCAL_MEM_DEPTH= hw_details.LOCAL_MEM_DEPTH
  GLOBAL_MEM_DEPTH= hw_details.GLOBAL_MEM_DEPTH
  
  LOCAL_POS_BITS= clog2(LOCAL_MEM_DEPTH)
  GLOBAL_POS_BITS= clog2(GLOBAL_MEM_DEPTH)
  const_o= const(hw_details)

  assert const_o.GLOBAL_MEM_ADDR_L == GLOBAL_POS_BITS + clog2(N_PE)

  bank= mem_obj.bank
  pos= mem_obj.pos
  
  if mem_obj.store_in_local_mem:
    addr= (const_o.LOCAL_MEM_INDICATOR << const_o.LOCAL_MEM_INDICATOR_S) | pos 
  elif mem_obj.store_in_global_mem:
    addr= ((const_o.LOCAL_MEM_INDICATOR ^ 1) << const_o.LOCAL_MEM_INDICATOR_S) | (bank << GLOBAL_POS_BITS) | pos 
  else:
    assert 0
  
  return addr

def ld_streams(list_of_schedules, hw_details):
  
  const_o= const(hw_details)

  # List of lists, one per pe
  # list element is the stream value of load stream (consists of register address and global mem addr)
  init_stream_ld= []

  for pe_id, pe_schedule in enumerate(list_of_schedules):
    init_stream_ld.append([])
    for schedule in pe_schedule:
      for instr in schedule:
        if instr.to_load_0:
          mem_obj= instr.load_0_addr
          data_addr= construct_runtime_data_addr(mem_obj, hw_details)
          stream_val= (instr.load_0_reg << const_o.GLOBAL_MEM_ADDR_L + 1) | data_addr
          init_stream_ld[pe_id].append(stream_val)

        if instr.to_load_1:
          mem_obj= instr.load_1_addr
          data_addr= construct_runtime_data_addr(mem_obj, hw_details)
          stream_val= (instr.load_1_reg << const_o.GLOBAL_MEM_ADDR_L + 1) | data_addr
          init_stream_ld[pe_id].append(stream_val)
          

  # A single merged-list with data in format "init_addr <space> stream value". To be directly written to file
  file_lines= []
  for pe_id, stream_ld in enumerate(init_stream_ld):
    #assert len(stream_ld) < hw_details.STREAM_LD_BANK_DEPTH
    for pos, stream_val in enumerate(stream_ld):
#      if pos >= len(stream_ld):
      if pos >= hw_details.STREAM_LD_BANK_DEPTH:
        break
      init_addr= construct_init_stream_addr(pe_id, pos, hw_details, stream_type= 'ld')
      line= format_hex(init_addr, 32) + ' ' + format_hex(stream_val, 32) + '\n'
      file_lines.append(line)
  
  return init_stream_ld, file_lines

def st_streams(list_of_schedules, hw_details):
  
  # List of lists, one per pe
  # list element is the stream value of store stream (global mem addr)
  init_stream_st= []

  for pe_id, pe_schedule in enumerate(list_of_schedules):
    init_stream_st.append([])
    for schedule in pe_schedule:
      for instr in schedule:
        if instr.to_store:
          mem_obj= instr.store_addr
          data_addr= construct_runtime_data_addr(mem_obj, hw_details)
          stream_val= data_addr
          init_stream_st[pe_id].append(stream_val)

  # A single merged-list with data in format "init_addr <space> stream value". To be directly written to file
  file_lines= []
  for pe_id, stream_st in enumerate(init_stream_st):
    #assert len(stream_st) < hw_details.STREAM_ST_BANK_DEPTH
    for pos, stream_val in enumerate(stream_st):
#      if pos >= len(stream_st):
      if pos >= hw_details.STREAM_ST_BANK_DEPTH:
        break
      init_addr= construct_init_stream_addr(pe_id, pos, hw_details, stream_type= 'st')
      line= format_hex(init_addr, 32) + ' ' + format_hex(stream_val, 32) + '\n'
      file_lines.append(line)
  
  return init_stream_st, file_lines

def instr_streams(list_of_schedules, hw_details):
  
  const_o = const(hw_details)

  init_stream_instr= []
  for pe_id, pe_schedule in enumerate(list_of_schedules):
    init_stream_instr.append([])
    for schedule in pe_schedule:
      for instr in schedule:
        if instr.is_sum() or instr.is_prod() or instr.is_pass() or instr.is_nop():
          if instr.is_sum():
            instr_bits = const_o.SUM_OPCODE
          elif instr.is_prod():
            instr_bits = const_o.PROD_OPCODE
          elif instr.is_pass():
            instr_bits = const_o.PASS_OPCODE
          elif instr.is_nop():
            instr_bits = const_o.NOP_OPCODE
          else:
            assert 0
          
          if instr.reg_in_0 != None:
            instr_bits |= (instr.reg_in_0 << const_o.OPCODE_L)

          if instr.reg_in_1 != None:
            instr_bits |= (instr.reg_in_1 << (const_o.OPCODE_L + const_o.REG_ADDR_L))

          if instr.reg_o != None:
            instr_bits |= (instr.reg_o << (const_o.OPCODE_L + 2 * const_o.REG_ADDR_L))

          if instr.to_load_0:
            instr_bits |= (1 << (const_o.OPCODE_L + 3 * const_o.REG_ADDR_L))
          
          if instr.to_load_1:
            assert instr.to_load_0 == True, instr.to_load_0
            instr_bits |= (1 << (const_o.OPCODE_L + 3 * const_o.REG_ADDR_L + 1))

          if instr.to_store:
            instr_bits |= (1 << (const_o.OPCODE_L + 3 * const_o.REG_ADDR_L + 2))

          init_stream_instr[pe_id].append(instr_bits)

        if instr.is_local_barrier():
          instr_bits = const_o.LOCAL_BARRIER_OPCODE
          init_stream_instr[pe_id].append(instr_bits)

        if instr.is_set_ld_stream_len():
          instr_bits = const_o.SET_LD_STREAM_LEN_OPCODE
          instr_bits |= (instr.ld_stream_len << const_o.OPCODE_L)

          assert instr.ld_stream_len < 2**(3*const_o.REG_ADDR_L), "Need to add local barriers: {instr.ld_stream_len}"

          init_stream_instr[pe_id].append(instr_bits)
          
        
      # append a global barrier
      instr_bits = const_o.GLOBAL_BARRIER_OPCODE
      init_stream_instr[pe_id].append(instr_bits)

  # A single merged-list with data in format "init_addr <space> stream value". To be directly written to file
  file_lines= []
  for pe_id, stream_instr in enumerate(init_stream_instr):
    #assert len(stream_instr) < hw_details.STREAM_INSTR_BANK_DEPTH
    for pos, stream_val in enumerate(stream_instr):
      assert (stream_val & (0xffffffff << const_o.INSTR_L)) == 0
#      if pos >= len(stream_instr):
      if pos >= hw_details.STREAM_INSTR_BANK_DEPTH:
        break
      init_addr= construct_init_stream_addr(pe_id, pos, hw_details, stream_type= 'instr')
      line= format_hex(init_addr, 32) + ' ' + format_hex(stream_val, 32) + '\n'
      file_lines.append(line)

  return init_stream_instr, file_lines      

def config_data(global_var, init_data_class_obj, config_obj):
  hw_details= config_obj.hw_details
  const_o = const(hw_details)
  
  N_PE= hw_details.N_PE

  # generate config data
  init_data_class_obj.config_instr_stream_start_addr_io = [0] * N_PE 
  init_data_class_obj.config_instr_stream_end_addr_io   = [0] * N_PE

  init_data_class_obj.config_ld_stream_start_addr_io    = [0] * N_PE
  init_data_class_obj.config_ld_stream_end_addr_io      = [0] * N_PE

  init_data_class_obj.config_st_stream_start_addr_io    = [0] * N_PE
  init_data_class_obj.config_st_stream_end_addr_io      = [0] * N_PE
  
  init_stream_instr= init_data_class_obj.init_stream_instr
  init_stream_ld   = init_data_class_obj.init_stream_ld
  init_stream_st   = init_data_class_obj.init_stream_st

  op_count, instr_count, ld_count, st_count = count_statistics(init_stream_instr, hw_details)

  for pe in range(N_PE):
    init_data_class_obj.config_instr_stream_start_addr_io[pe] = 0
#    stream_len= len(init_stream_instr[pe])
#    if stream_len > hw_details.STREAM_INSTR_BANK_DEPTH - 4:
#      stream_len = hw_details.STREAM_INSTR_BANK_DEPTH - 4
#    init_data_class_obj.config_instr_stream_end_addr_io  [pe] = stream_len 
    init_data_class_obj.config_instr_stream_end_addr_io  [pe] = instr_count[pe]

    init_data_class_obj.config_ld_stream_start_addr_io   [pe] = 0
#    stream_len= len(init_stream_ld[pe])
#    if stream_len > hw_details.STREAM_LD_BANK_DEPTH - 4:
#      stream_len = hw_details.STREAM_LD_BANK_DEPTH - 4
#    init_data_class_obj.config_ld_stream_end_addr_io     [pe] = stream_len
    init_data_class_obj.config_ld_stream_end_addr_io     [pe] = ld_count[pe]

    init_data_class_obj.config_st_stream_start_addr_io   [pe] = 0
#    stream_len= len(init_stream_st[pe])
#    if stream_len > hw_details.STREAM_ST_BANK_DEPTH - 4:
#      stream_len = hw_details.STREAM_ST_BANK_DEPTH - 4
#    init_data_class_obj.config_st_stream_end_addr_io     [pe] = stream_len
    init_data_class_obj.config_st_stream_end_addr_io     [pe] = st_count[pe]


  if hw_details.TOT_L   == 32:
    init_data_class_obj.config_precision_config= const_o.PRECISION_CONFIG_32B
  elif hw_details.TOT_L == 16:
    init_data_class_obj.config_precision_config= const_o.PRECISION_CONFIG_16B
  elif hw_details.TOT_L == 8:
    init_data_class_obj.config_precision_config= const_o.PRECISION_CONFIG_8B
  else:
    assert 0
  
  # write to file
  file_lines= []
  for pe in range(N_PE):
    file_lines.append(format_hex(init_data_class_obj.config_instr_stream_start_addr_io[pe], 32) + '\n')
    file_lines.append(format_hex(init_data_class_obj.config_instr_stream_end_addr_io  [pe], 32) + '\n')

    file_lines.append(format_hex(init_data_class_obj.config_ld_stream_start_addr_io   [pe], 32) + '\n')
    file_lines.append(format_hex(init_data_class_obj.config_ld_stream_end_addr_io     [pe], 32) + '\n')

    file_lines.append(format_hex(init_data_class_obj.config_st_stream_start_addr_io   [pe], 32) + '\n')
    file_lines.append(format_hex(init_data_class_obj.config_st_stream_end_addr_io     [pe], 32) + '\n')
  
  file_lines.append( format_hex( init_data_class_obj.config_precision_config , 32) + '\n')
  
  # memory slp and sd
  mem_cnt= ((N_PE + 31)//32) * 10
  for i in range(mem_cnt):
    file_lines.append(format_hex(0,32) + '\n')    

  file_lines.reverse()

  if config_obj.write_files:
    fname= path_prefix(global_var, config_obj) + '_config.txt'
    print(fname)
    with open(fname, 'w+') as f:
      f.writelines(file_lines)

def instr_decode(instr, hw_details):
  const_o = const(hw_details)
  opcode= instr & (2**(const_o.OPCODE_L) -1)
  
  reg_in_0 = 0
  reg_in_1 = 0
  reg_o    = 0
  ld_0_en  = 0
  ld_1_en  = 0
  st_en    = 0

  if opcode == const_o.SUM_OPCODE or opcode == const_o.PROD_OPCODE or opcode == const_o.NOP_OPCODE or opcode == const_o.PASS_OPCODE:
    instr >>= const_o.OPCODE_L
    reg_in_0= instr & (2**(const_o.REG_ADDR_L) -1)

    instr >>= const_o.REG_ADDR_L
    reg_in_1= instr & (2**(const_o.REG_ADDR_L) -1)

    instr >>= const_o.REG_ADDR_L
    reg_o= instr & (2**(const_o.REG_ADDR_L) -1)

    instr >>= const_o.REG_ADDR_L
    ld_0_en= instr & 1
    
    instr >>= 1
    ld_1_en= instr & 1

    instr >>= 1
    st_en= instr & 1
  
  return opcode, reg_in_0, reg_in_1, reg_o, ld_0_en, ld_1_en, st_en
  
def count_statistics(init_stream_instr, hw_details):
  N_PE= hw_details.N_PE

  op_count= 0
  instr_count = [0 for _ in range(N_PE)]
  ld_count    = [0 for _ in range(N_PE)]
  st_count    = [0 for _ in range(N_PE)]
  const_o = const(hw_details)

  last_global_barrier= 10000000
  for pe, instr_ls in enumerate(init_stream_instr):
    last_barrier= 0
    for idx, instr in enumerate(instr_ls):
      if idx > hw_details.STREAM_INSTR_BANK_DEPTH - 4:
        break
      if st_count[pe] > hw_details.STREAM_ST_BANK_DEPTH - 4:
        break
      if ld_count[pe] > hw_details.STREAM_LD_BANK_DEPTH - 4:
        break
      
      opcode, _ , _ , _ , ld_0_en, ld_1_en, st_en = instr_decode(instr, hw_details)
      
      instr_count[pe] += 1

      if opcode == const_o.SUM_OPCODE or opcode == const_o.PROD_OPCODE:
        op_count += 1

      ld_count[pe] += ld_0_en + ld_1_en
      st_count[pe] += st_en

      if opcode == const_o.GLOBAL_BARRIER_OPCODE:
        last_barrier += 1

    last_global_barrier= min(last_global_barrier, last_barrier)

#  last_global_barrier = 2

  if last_global_barrier < last_barrier:
    log.warning('Full graph is not being computed')
  log.info(f'last_global_barrier: {last_global_barrier} out of {last_barrier} barriers')
  if last_global_barrier == 0:
    assert 0, "First global barrier is beyond mem sizes"

  op_count= 0
  instr_count = [0 for _ in range(N_PE)]
  ld_count    = [0 for _ in range(N_PE)]
  st_count    = [0 for _ in range(N_PE)]
  for pe, instr_ls in enumerate(init_stream_instr):
    last_barrier= 0
    for idx, instr in enumerate(instr_ls):
      if idx > hw_details.STREAM_INSTR_BANK_DEPTH - 4:
        assert 0

      if st_count[pe] > hw_details.STREAM_ST_BANK_DEPTH - 4:
        assert 0

      if ld_count[pe] > hw_details.STREAM_LD_BANK_DEPTH - 4:
        assert 0

      opcode, _ , _ , _ , ld_0_en, ld_1_en, st_en = instr_decode(instr, hw_details)
      
      instr_count[pe] += 1

      if opcode == const_o.SUM_OPCODE or opcode == const_o.PROD_OPCODE:
        op_count += 1

      ld_count[pe] += ld_0_en + ld_1_en
      st_count[pe] += st_en

      if opcode == const_o.GLOBAL_BARRIER_OPCODE:
        last_barrier += 1

      if last_barrier ==  last_global_barrier:
        break

  log.info(f'last_global_barrier: {last_global_barrier} out of {last_barrier} barriers')
  log.info(f'Statistics: op_count: {op_count}, instr_count: {instr_count}, ld_count: {ld_count}, st_count: {st_count}')
  return op_count, instr_count, ld_count, st_count

def path_prefix(global_var, config_obj):
  hw_details = config_obj.hw_details
  prefix  = str(global_var.PRU_ASYNC_VERIF_PATH)
  prefix += '_' + config_obj.partition_mode.name
  if config_obj.sub_partition_mode != None:
    prefix += '_' + config_obj.sub_partition_mode.name
  prefix += '/' + config_obj.name.replace("/", "_")
  # prefix += '_' + str('hybrid')
  prefix += '_' + str(hw_details.N_PE)
  prefix += '_' + str(hw_details.LOCAL_MEM_DEPTH)
  prefix += '_' + str(hw_details.GLOBAL_MEM_DEPTH)
  prefix += '_' + str(hw_details.REGBANK_L)
  prefix += '_' + str(hw_details.DTYPE)
  prefix += '_' + str(hw_details.TOT_L) + 'b'
  prefix += '_' + str(hw_details.STREAM_MAX_ADDR_L_PER_BANK)

  return prefix
