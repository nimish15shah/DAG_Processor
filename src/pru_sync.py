
from collections import defaultdict

from . import common_classes
from . import ac_eval

from .useful_methods import clog2, format_hex, printcol

from .util import convert_data, custom_precision_operation

import logging
logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def gen_mem_addr(hw_details, addr, bank):
  n_banks = hw_details.n_banks
  mem_bank_depth = hw_details.mem_bank_depth
  mem_addr_bits = hw_details.mem_addr_bits
  
  bits_per_bank = clog2(mem_bank_depth)

  assert bits_per_bank == mem_addr_bits

  init_addr_bits= clog2(n_banks) + bits_per_bank
  
  assert addr <= mem_bank_depth
  assert bank <= n_banks

  final_addr= (bank << bits_per_bank) + addr

  assert (final_addr >> init_addr_bits) == 0

  return final_addr

def write_param_files(graph, prefix, hw_details, map_param_to_addr, write_files):
  
  lines= []

  for node, addr_bank_tup in map_param_to_addr.items():
    val= graph[node].curr_val
    val= convert_data(val, hw_details, mode= 'to')

    addr, bank = addr_bank_tup
    final_addr= gen_mem_addr(hw_details, addr, bank)
    line= format_hex(final_addr, hw_details.mem_addr_bits) + ' ' + format_hex(val, hw_details.n_bits) + '\n'
    lines.append(line)

  fname= prefix + '_data.txt'
  if write_files:
    logger.info(f"Writing file: {fname}")
    with open(fname, 'w+') as f:
      f.writelines(lines)

def main(graph, prefix, hw_details, instr_ls, map_param_to_addr, map_output_to_addr, write_files):
  
  write_param_files(graph, prefix, hw_details, map_param_to_addr, write_files)

  # map (addr,bank) to value
  map_addr_to_val= {}
  map_addr_to_val_normal= {}

  final_node= [node for node, obj in list(graph.items()) if len(obj.parent_key_list) == 0]
  final_node= final_node.pop()

  for node, obj in list(graph.items()):
    if obj.is_leaf():
      assert node in map_param_to_addr

  # init_leaf_val(graph, mode='random')

  for node, addr_bank_tup in list(map_param_to_addr.items()):
    val = graph[node].curr_val
    # graph[node].curr_val= val
#    map_addr_to_val[tuple(addr_bank_tup)]= val
    map_addr_to_val[tuple(addr_bank_tup)]= convert_data(val , hw_details, mode= 'to') 
    map_addr_to_val_normal[tuple(addr_bank_tup)]= val

  print('Golden:', ac_eval.ac_eval(graph, final_node))

  # simulated should run after golden because there are assertions depending on the golden run
  simulated_val= simulate_instr_sync(graph, instr_ls,map_addr_to_val_normal, hw_details, prefix, write_files) 
  print('Simulated:', simulated_val[final_node])

  result= ac_eval.ac_eval(graph, final_node, precision='CUSTOM' ,arith_type= 'FLOAT', exp= hw_details.EXP_L, mant= hw_details.MANT_L)
  print('Golden w/ limited precision:', result)
  
  # Write the golden value
  lines= []
  assert len(map_output_to_addr) == 1, "Modify the following code to handle multiple outputs"
  for obj in map_output_to_addr.values():
    bank, pos = obj
    assert pos== 0, "register banks should be empty for the final node"

  val= convert_data(result, hw_details, mode= 'to')
  line= format_hex(bank, 6) + ' ' + format_hex(val, hw_details.n_bits) + '\n'
  lines.append(line)

  if write_files:
    fname= prefix+ '_golden.txt'
    logger.info(f"Writing file: {fname}")
    with open(fname, 'w+') as f:
      f.writelines(lines)

class reg_bank_c():
  def __init__(self, DEPTH):
    self.free_pos= set(list(range(DEPTH)))
    self.data= [None for _ in range(DEPTH)]
  
  def read(self, pos):
    assert pos not in self.free_pos
    return self.data[pos]

  def inv(self, pos):
    assert pos not in self.free_pos
    self.data[pos]= None
    self.free_pos.add(pos)

  def write(self, data):
    pos= min(list(self.free_pos))
    self.data[pos]= data
    
    self.free_pos.remove(pos)

    return pos

def reg_rd_stage(graph, instr, reg_bank, memory, curr_val, hw_details, prefix, write_files):
  reg_re= defaultdict(lambda : 0)
  reg_inv= defaultdict(lambda : 0)
  reg_rd_data= defaultdict(lambda : 0)
  reg_rd_addr= defaultdict(lambda : 0)

  n_banks = hw_details.n_banks

  this_pipestage_out_data= {}
  if instr.is_type('nop'):
    pass
  elif instr.is_type('ld') or instr.is_type('ld_sc'):
    pass
  elif instr.is_type('st') or instr.is_type('st_8') or instr.is_type('st_4'):
    mem_addr= instr.mem_addr
    for node, obj in list(instr.node_details_dict.items()):
      bank= obj.bank
      pos= obj.pos
      data= reg_bank[bank].read(pos)
      memory[bank][mem_addr]= data
      reg_bank[bank].inv(pos)

      reg_re[bank] = 1
      reg_rd_data[bank]= convert_data(data, hw_details, mode='to')
      reg_inv[bank] = 1
      reg_rd_addr[bank] = pos

  elif instr.is_type('sh') or instr.is_type('sh_4') or instr.is_type('sh_8') or instr.is_type('sh_2'):
    map_node_to_val= {}
    for node, src_dst_tup in list(instr.sh_dict_bank.items()):
      src_bank= src_dst_tup[0]
      src_pos= instr.sh_dict_pos[node][0]
      data= reg_bank[src_bank].read(src_pos)
      map_node_to_val[node]= data

      reg_re[src_bank] = 1
      reg_rd_data[src_bank]= convert_data(data, hw_details, mode='to')
      reg_rd_addr[src_bank] = src_pos

      if node in instr.invalidate_node_set:
        reg_bank[src_bank].inv(src_pos)
        reg_inv[src_bank] = 1
        
    this_pipestage_out_data= map_node_to_val

  elif instr.is_type('bb'):
    # map pe to val
    # key: pe, val: val
    map_pe_to_val= {}

    # first-level PEs
    for pe, pe_details in list(instr.pe_details.items()):
      if pe[1]==1:
        if pe_details.input_0_reg != None:
          bank, pos= pe_details.input_0_reg
          in_0 = reg_bank[bank].read(pos)

          reg_re[bank] = 1
          reg_rd_data[bank]= convert_data(in_0, hw_details, mode='to')
          reg_rd_addr[bank]= pos
        else:
          in_0= None
        if pe_details.input_1_reg != None:
          bank, pos= pe_details.input_1_reg
          in_1 = reg_bank[bank].read(pos)

          reg_re[bank] = 1
          reg_rd_data[bank]= convert_data(in_1, hw_details, mode='to')
          reg_rd_addr[bank]= pos

        if pe_details.is_sum():
          map_pe_to_val[pe] = in_0 + in_1
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          assert graph[pe_details.node].is_prod()
          map_pe_to_val[pe]= in_0 * in_1
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val, [map_pe_to_val[pe], graph[pe_details.node].curr_val, pe_details.node, graph[pe_details.node].child_key_list]
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= in_0
        elif pe_details.is_pass_1():
          # map_pe_to_val[pe]= None
          map_pe_to_val[pe]= in_0
        else:
          assert 0

    # rest of PEs
    sorted_pes = sorted(list(instr.pe_details.keys()), key= lambda x: x[1])
    for pe in sorted_pes:
      pe_details = instr.pe_details[pe]
      if pe[1] != 1:
        in_0= (pe[0], pe[1]-1, 2*pe[2])
        in_1= (pe[0], pe[1]-1, 2*pe[2] + 1)
        if pe_details.is_sum():
          assert in_0 in map_pe_to_val, [in_0, map_pe_to_val, pe]
          assert in_1 in map_pe_to_val, [in_1, map_pe_to_val, pe]
          map_pe_to_val[pe] = map_pe_to_val[in_0] + map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_prod():
          assert in_0 in map_pe_to_val, [in_0, map_pe_to_val, pe]
          assert in_1 in map_pe_to_val, [in_1, map_pe_to_val, pe]
          map_pe_to_val[pe]= map_pe_to_val[in_0] * map_pe_to_val[in_1]
          assert map_pe_to_val[pe] == graph[pe_details.node].curr_val
          curr_val[pe_details.node]= map_pe_to_val[pe]
        elif pe_details.is_pass_0():
          map_pe_to_val[pe]= map_pe_to_val[in_0]
        elif pe_details.is_pass_1():
          map_pe_to_val[pe]= map_pe_to_val[in_0]
          # map_pe_to_val[pe]= None
        else:
          assert 0

    # Invalidate
    for  node in instr.invalidate_node_set:
      bank= instr.in_node_details_dict[node].bank
      pos= instr.in_node_details_dict[node].pos
      reg_bank[bank].inv(pos)

      reg_inv[bank] = 1

    this_pipestage_out_data= map_pe_to_val

  else:
    assert False

  fname_re= prefix + '_reg_re.txt'
  fname_inv= prefix + '_reg_inv.txt'
  fname_rd_data= prefix + '_reg_rd_data.txt'
  fname_rd_addr= prefix + '_reg_rd_addr.txt'

  if write_files:
    with open(fname_re, 'a') as f:
      re_ls= [reg_re[b] for b in range(n_banks)]
      re_str= ""
      for re in re_ls:
        re_str += str(re)
      print(re_str, file= f)

    with open(fname_inv, 'a') as f:
      inv_ls= [reg_inv[b] for b in range(n_banks)]
      inv_str= ""
      for bank, inv in enumerate(inv_ls):
        if inv:
          assert reg_re[bank], f"bank: {bank}, reg_re[bank], reg_inv[bank]"
        inv_str += str(inv)
      print(inv_str, file= f)

    with open(fname_rd_data, 'a') as f:
      rd_data_str= ""
      for b in range(n_banks):
        rd_data_str += format_hex(reg_rd_data[b], hw_details.n_bits) + ' '
      print(rd_data_str, file= f)

    with open(fname_rd_addr, 'a') as f:
      rd_addr_str= ""
      for b in range(n_banks):
        rd_addr_str += format_hex(reg_rd_addr[b], clog2(hw_details.reg_bank_depth)) + ' '
      print(rd_addr_str, file= f)

  return this_pipestage_out_data


def reg_wr_stage(instr, reg_bank, memory, write_data, hw_details, prefix, write_files):

  reg_we= defaultdict(lambda : 0)
  reg_wr_data= defaultdict(lambda : 0)
  reg_wr_addr= defaultdict(lambda : 0)

  n_banks = hw_details.n_banks

  if instr.is_type('nop'):
    pass

  elif instr.is_type('ld') or instr.is_type('ld_sc'):
    mem_addr= instr.mem_addr
    for node, obj in list(instr.node_details_dict.items()):
      bank= obj.bank
      data= memory[bank][mem_addr]
      assert data != None, [mem_addr, bank, data, node]
      pos= reg_bank[bank].write(data)
      assert pos== obj.pos, f"{node}, {pos}, {obj.pos}, {bank}"
      reg_we[bank] = 1
      reg_wr_data[bank]= convert_data(data, hw_details, mode='to')
      reg_wr_addr[bank] = pos

  elif instr.is_type('st') or instr.is_type('st_8') or instr.is_type('st_4'):
    pass

  elif instr.is_type('sh') or instr.is_type('sh_4') or instr.is_type('sh_8') or instr.is_type('sh_2'):
    for node, src_dst_tup in list(instr.sh_dict_bank.items()):
      dst_bank= src_dst_tup[1]
      data= write_data[node]
      dst_pos= reg_bank[dst_bank].write(data)
      assert dst_pos == instr.sh_dict_pos[node][1]
      reg_we[dst_bank] = 1
      reg_wr_data[dst_bank]= convert_data(data, hw_details, mode='to')
      reg_wr_addr[dst_bank] = dst_pos

  elif instr.is_type('bb'):

    # Outputs
    for node, pe in list(instr.output_to_pe.items()):
      pe_details= instr.pe_details[pe]
      assert node == pe_details.node
      bank, pos= pe_details.output_reg
      data= write_data[pe]
      written_pos= reg_bank[bank].write(data)
      assert pos == written_pos

      reg_we[bank] = 1
      reg_wr_data[bank]= convert_data(data, hw_details, mode='to')
      reg_wr_addr[bank] = pos
      
  else:
    assert 0

  fname_we= prefix + '_reg_we.txt'
  fname_wr_data= prefix + '_reg_wr_data.txt'
  fname_wr_addr= prefix + '_reg_wr_addr.txt'
  if write_files:
    with open(fname_we, 'a') as f:
      we_ls= [reg_we[b] for b in range(n_banks)]
      we_str= ""
      for we in we_ls:
        we_str += str(we)
      print(we_str, file= f)

    with open(fname_wr_data, 'a') as f:
      wr_data_str= ""
      for b in range(n_banks):
        wr_data_str += format_hex(reg_wr_data[b], hw_details.n_bits) + ' '
        if reg_wr_data[b] != 0:
          assert reg_we[b]

      print(wr_data_str, file= f)

    with open(fname_wr_addr, 'a') as f:
      wr_addr_str= ""
      for b in range(n_banks):
        wr_addr_str += format_hex(reg_wr_addr[b], clog2(hw_details.reg_bank_depth)) + ' '
        if reg_wr_addr[b] != 0:
          assert reg_we[b]
      print(wr_addr_str, file= f)

def simulate_instr_sync(graph, instr_ls, map_addr_to_val, hw_details, prefix, write_files):

  
  MEM_DEPTH= hw_details.mem_bank_depth
  BANK_DEPTH= hw_details.reg_bank_depth
  N_BANKS= hw_details.n_banks

  reg_bank= [reg_bank_c(BANK_DEPTH) for _i in range(N_BANKS)]
  # memory=  [[None for _j in range(MEM_DEPTH)] for _i in range(N_BANKS)]
  memory=  [defaultdict(None) for _i in range(N_BANKS)]
  
  # to store local compute value
  # key: node, val: curr_val
  curr_val= {node: obj.curr_val for node, obj in list(graph.items()) if obj.is_leaf()}
  

  # memory init
  for addr_bank_tup, data in list(map_addr_to_val.items()):
    addr= addr_bank_tup[0]
    bank= addr_bank_tup[1]
    memory[bank][addr]= data
    assert data != None
    
  # pipestages
  instr_in_pipe= [common_classes.nop_instr()] * (hw_details.n_pipe_stages - 1)
  data_in_pipe= [{}] * (hw_details.n_pipe_stages - 1)

  # clear files
  fname_we= prefix + '_reg_we.txt'
  fname_wr_data= prefix + '_reg_wr_data.txt'
  fname_wr_addr= prefix + '_reg_wr_addr.txt'
  fname_re= prefix + '_reg_re.txt'
  fname_inv= prefix + '_reg_inv.txt'
  fname_rd_data= prefix + '_reg_rd_data.txt'
  fname_rd_addr= prefix + '_reg_rd_addr.txt'
  if write_files:
    open(fname_we, 'w+').close()
    open(fname_wr_data, 'w+').close()
    open(fname_wr_addr , 'w+').close()
    open(fname_re, 'w+').close()
    open(fname_inv, 'w+').close()
    open(fname_rd_data, 'w+').close()
    open(fname_rd_addr, 'w+').close()

  for idx, instr in enumerate(instr_ls):
    # First perform the register writes and then reads, because invalidation should reflect in the next cycle
    commit_instr= instr_in_pipe.pop(0)
    # print("WR stage instr")
    # commit_instr.print_details()
    write_data= data_in_pipe.pop(0)
    reg_wr_stage(commit_instr, reg_bank, memory, write_data, hw_details, prefix, write_files)

    write_data= reg_rd_stage(graph, instr, reg_bank, memory, curr_val, hw_details, prefix, write_files)
    instr_in_pipe.append(instr)
    data_in_pipe.append(write_data)
    # print("RD stage instr")
    # instr.print_details()

    # if (idx == 18):
    #   exit(1)


  return curr_val


