#=======================================================================
# Created by         : KU Leuven
# Filename           : ./src/reporting_tools/write_binary.py
# Author             : Nimish Shah
# Created On         : 2019-10-22 10:03
# Last Modified      : 
# Update Count       : 2019-10-22 10:03
# Description        : 
#                      
#=======================================================================


from .useful_methods import printlog, printcol, clog2

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def write_binary(file_name, global_var, instr_ls, hw_details, verb= False):
  """
    Writes the instruction binary, based on instr_ls
  """
  
  #-------------------------------------------
  # Variable declaration
  #-------------------------------------------
  BIT_L          = hw_details.n_bits  # from hw_details                    
  TREE_DEPTH     = hw_details.tree_depth                        
  BANK_DEPTH     = hw_details.reg_bank_depth                      
  MEM_ADDR_L     = hw_details.mem_addr_bits                      
  N_TREE         = hw_details.n_tree

  N_IN_PER_TREE  = (2**TREE_DEPTH)         
  N_IN           = N_TREE * N_IN_PER_TREE  
  N_ALU_PER_TREE = ((2**TREE_DEPTH)-1)     
  N_ALU          = N_TREE * N_ALU_PER_TREE 
  N_BANKS        = N_IN                    
  PIPE_STAGES    = TREE_DEPTH + 2

  
  OPCODE_L  = 4 # from opcodes package
  NOP_STALL = 0
  NOP       = 1
  BB        = 2
  CP_8      = 3
  ST        = 4
  LD        = 5
  ST_4      = 6
  ST_8      = 7
  CP_2      = 8
  CP_4      = 9
  CP        = 10

  SUM   = 0  # typedefs
  PROD  = 1
  PASS_0= 2 
  PASS_1= 3 
  alu_mode_enum_t = 4

  ARITH_L= clog2(alu_mode_enum_t)
  BANK_ADDR_L    = clog2(BANK_DEPTH)  # from instr_decd_pkg
  CROSSBAR_SEL_L = clog2(N_BANKS)                                                                    
  BANK_WR_SEL_L= clog2(TREE_DEPTH + 1)

  NOP_OPCODE = NOP                                                                                    
  NOP_L      = OPCODE_L                                                                               

  NOP_STALL_OPCODE = NOP_STALL
  NOP_STALL_L      = OPCODE_L

  LD_OPCODE     = LD                                                                                  
  LD_EN_S       = OPCODE_L                                                                            
  LD_EN_L       = N_BANKS                                                                             
  LD_MEM_ADDR_S = LD_EN_S + LD_EN_L                                                                   
  LD_MEM_ADDR_L = MEM_ADDR_L                                                                          
  LD_L          = OPCODE_L + LD_MEM_ADDR_L + LD_EN_L                                                  

  ST_OPCODE         = ST                                                                              
  ST_EN_S           = OPCODE_L                                                                        
  ST_EN_L           = N_BANKS                                                                         
  ST_BANK_RD_ADDR_S = ST_EN_S + ST_EN_L                                                               
  ST_BANK_RD_ADDR_L = BANK_ADDR_L * N_BANKS                                                           
  ST_MEM_ADDR_S     = ST_BANK_RD_ADDR_S + ST_BANK_RD_ADDR_L                                           
  ST_MEM_ADDR_L     = MEM_ADDR_L                                                                      
  ST_L              = OPCODE_L + ST_BANK_RD_ADDR_L + ST_MEM_ADDR_L + ST_EN_L                          

  ST_4_OPCODE         = ST_4                                                                          
  ST_4_EN_S           = OPCODE_L                                                                      
  ST_4_EN_L           = N_BANKS                                                                       
  ST_4_BANK_RD_ADDR_S = ST_4_EN_S + ST_4_EN_L                                                         
  ST_4_BANK_RD_ADDR_L = BANK_ADDR_L * 4                                                               
  ST_4_MEM_ADDR_S     = ST_4_BANK_RD_ADDR_S + ST_4_BANK_RD_ADDR_L                                     
  ST_4_MEM_ADDR_L     = MEM_ADDR_L                                                                    
  ST_4_L              = OPCODE_L + ST_4_BANK_RD_ADDR_L + ST_4_MEM_ADDR_L + ST_4_EN_L                  

  ST_8_OPCODE         = ST_8                                                                          
  ST_8_EN_S           = OPCODE_L                                                                      
  ST_8_EN_L           = N_BANKS                                                                       
  ST_8_BANK_RD_ADDR_S = ST_8_EN_S + ST_8_EN_L                                                         
  ST_8_BANK_RD_ADDR_L = BANK_ADDR_L * 8                                                               
  ST_8_MEM_ADDR_S     = ST_8_BANK_RD_ADDR_S + ST_8_BANK_RD_ADDR_L                                     
  ST_8_MEM_ADDR_L     = MEM_ADDR_L                                                                    
  ST_8_L              = OPCODE_L + ST_8_BANK_RD_ADDR_L + ST_8_MEM_ADDR_L + ST_8_EN_L                  

  CP_OPCODE         = CP;
  CP_WR_EN_S        = OPCODE_L;
  CP_WR_EN_L        = N_BANKS;
  CP_RD_EN_S        = CP_WR_EN_S + CP_WR_EN_L;
  CP_RD_EN_L        = N_BANKS;
  CP_BANK_RD_ADDR_S = CP_RD_EN_S + CP_RD_EN_L;
  CP_BANK_RD_ADDR_L = BANK_ADDR_L * N_BANKS;
  CP_CROSS_SEL_S    = CP_BANK_RD_ADDR_S + CP_BANK_RD_ADDR_L;
  CP_CROSS_SEL_L    = CROSSBAR_SEL_L * N_BANKS;
  CP_INVLD_S        = CP_CROSS_SEL_S + CP_CROSS_SEL_L;
  CP_INVLD_L        = N_BANKS;
  CP_L              = OPCODE_L + CP_WR_EN_L + CP_RD_EN_L + CP_BANK_RD_ADDR_L + CP_CROSS_SEL_L + CP_INVLD_L;

  CP_8_OPCODE         = CP_8                                                                          
  CP_8_EN_S           = OPCODE_L                                                                      
  CP_8_EN_L           = N_BANKS                                                                       
  CP_8_BANK_RD_ADDR_S = CP_8_EN_S + CP_8_EN_L                                                         
  CP_8_BANK_RD_ADDR_L = BANK_ADDR_L * 8                                                               
  CP_8_CROSS_SEL_S    = CP_8_BANK_RD_ADDR_S + CP_8_BANK_RD_ADDR_L                                     
  CP_8_CROSS_SEL_L    = CROSSBAR_SEL_L * 8                                                            
  CP_8_INVLD_S        = CP_8_CROSS_SEL_S + CP_8_CROSS_SEL_L                                           
  CP_8_INVLD_L        = 8                                                                             
  CP_8_L              = OPCODE_L + CP_8_EN_L + CP_8_BANK_RD_ADDR_L + CP_8_CROSS_SEL_L + CP_8_INVLD_L  

  CP_4_OPCODE         = CP_4                                                                          
  CP_4_EN_S           = OPCODE_L                                                                      
  CP_4_EN_L           = N_BANKS                                                                       
  CP_4_BANK_RD_ADDR_S = CP_4_EN_S + CP_4_EN_L                                                         
  CP_4_BANK_RD_ADDR_L = BANK_ADDR_L * 4                                                               
  CP_4_CROSS_SEL_S    = CP_4_BANK_RD_ADDR_S + CP_4_BANK_RD_ADDR_L                                     
  CP_4_CROSS_SEL_L    = CROSSBAR_SEL_L * 4                                                            
  CP_4_INVLD_S        = CP_4_CROSS_SEL_S + CP_4_CROSS_SEL_L                                           
  CP_4_INVLD_L        = 4                                                                             
  CP_4_L              = OPCODE_L + CP_4_EN_L + CP_4_BANK_RD_ADDR_L + CP_4_CROSS_SEL_L + CP_4_INVLD_L  

  CP_2_OPCODE         = CP_2                                                                          
  CP_2_EN_S           = OPCODE_L                                                                      
  CP_2_EN_L           = N_BANKS                                                                       
  CP_2_BANK_RD_ADDR_S = CP_2_EN_S + CP_2_EN_L                                                         
  CP_2_BANK_RD_ADDR_L = BANK_ADDR_L * 2                                                               
  CP_2_CROSS_SEL_S    = CP_2_BANK_RD_ADDR_S + CP_2_BANK_RD_ADDR_L                                     
  CP_2_CROSS_SEL_L    = CROSSBAR_SEL_L * 2                                                            
  CP_2_INVLD_S        = CP_2_CROSS_SEL_S + CP_2_CROSS_SEL_L                                           
  CP_2_INVLD_L        = 2                                                                             
  CP_2_L              = OPCODE_L + CP_2_EN_L + CP_2_BANK_RD_ADDR_L + CP_2_CROSS_SEL_L + CP_2_INVLD_L  

  BB_OPCODE              = BB                                                                         
  BB_INVLD_S             = OPCODE_L                                                                   
  BB_INVLD_L             = N_BANKS                                                                    
  BB_BANK_RD_ADDR_S      = BB_INVLD_S + BB_INVLD_L                                                    
  BB_BANK_RD_ADDR_L      = BANK_ADDR_L * N_BANKS                                                      
  BB_CROSS_SEL_S         = BB_BANK_RD_ADDR_S + BB_BANK_RD_ADDR_L                                      
  BB_CROSS_SEL_L         = CROSSBAR_SEL_L * N_BANKS                                                   
  BB_ARITH_OP_S          = BB_CROSS_SEL_S + BB_CROSS_SEL_L                                            
  BB_ARITH_OP_L          = ARITH_L * N_ALU                                             
  BB_BANK_WR_SEL_S       = BB_ARITH_OP_S + BB_ARITH_OP_L                                              
  BB_BANK_WR_SEL_L       = BANK_WR_SEL_L * N_BANKS                                          
  BB_L                   = OPCODE_L + BB_INVLD_L + BB_BANK_RD_ADDR_L + BB_CROSS_SEL_L + BB_ARITH_OP_L + BB_BANK_WR_SEL_L
                                                                                                                          
  INSTR_L                = max(BB_L, NOP_L, NOP_STALL_L, LD_L, ST_L, ST_4_L, ST_8_L, CP_L, CP_8_L)
  MIN_COMPRESSION_FACTOR = 4; 
  MAX_COMPRESSED_BANKS   = N_BANKS/MIN_COMPRESSION_FACTOR

  #-------------------------------------------
  #       Actual instr_ls processing
  #-------------------------------------------
  
  # instr binary for the whole instr_ls
  i_bin= "" 

  # replace "sh" instructions to compatible shorter sh instructions types
  use_shorter_st_instr(hw_details, MAX_COMPRESSED_BANKS, instr_ls)
  use_shorter_sh_instr(hw_details, MAX_COMPRESSED_BANKS, instr_ls)

  for instr in instr_ls:
    temp_instr_idx = len(i_bin)/INSTR_L
    start_temp_idx= 24
    last_temp_idx= 33

    curr_bin= ""
    # instr is an object of type common_classes.instr
    if instr.is_type('nop'):
      curr_bin= bin_update(curr_bin, NOP_OPCODE, OPCODE_L)
    elif instr.is_type('nop_stall'):
      curr_bin= bin_update(curr_bin, NOP_STALL_OPCODE, OPCODE_L)
    elif instr.is_type('ld') or instr.is_type('ld_sc'):
      curr_bin= bin_update(curr_bin, LD_OPCODE, OPCODE_L)

      # ld_en
      ld_en= 0
      for io_node_obj in list(instr.node_details_dict.values()):
        ld_en |= (1<<(io_node_obj.bank)) 
      curr_bin= bin_update(curr_bin, ld_en, LD_EN_L)
      
      # mem addr
      assert in_range(instr.mem_addr, LD_MEM_ADDR_L)
      curr_bin= bin_update(curr_bin, instr.mem_addr, LD_MEM_ADDR_L)

      # if temp_instr_idx > start_temp_idx and temp_instr_idx < last_temp_idx:
        # printcol(temp_instr_idx, 'blue')
        # printcol('ld', 'red')
        # print(bin(ld_en))
        # for io_node_obj in list(instr.node_details_dict.values()):
        #   print(io_node_obj.bank, io_node_obj.pos)

    elif instr.is_type('st') or instr.is_type('st_8') or instr.is_type('st_4'):
      if instr.is_type('st_8'):
        OPCODE         = ST_8_OPCODE
        EN_L           = ST_8_EN_L
        BANK_RD_ADDR_L = ST_8_BANK_RD_ADDR_L
        CURR_MEM_ADDR_L= ST_8_MEM_ADDR_L
        TOTAL_L        = ST_8_L
      elif instr.is_type('st_4'):
        OPCODE         = ST_4_OPCODE
        EN_L           = ST_4_EN_L
        BANK_RD_ADDR_L = ST_4_BANK_RD_ADDR_L
        CURR_MEM_ADDR_L= ST_4_MEM_ADDR_L
        TOTAL_L        = ST_4_L
      elif instr.is_type('st'):
        OPCODE         = ST_OPCODE
        EN_L           = ST_EN_L
        BANK_RD_ADDR_L = ST_BANK_RD_ADDR_L
        CURR_MEM_ADDR_L= ST_MEM_ADDR_L
        TOTAL_L        = ST_L
      else:
        assert 0

      curr_bin= bin_update(curr_bin, OPCODE, OPCODE_L)

      # st_en
      # st bank addr
      st_en= 0
      all_bank_addr= 0
      assert len(instr.node_set) == len(instr.node_details_dict)
      sorted_nodes= list(sorted(list(instr.node_set), key= lambda x : instr.node_details_dict[x].bank))
      # for io_node_obj in list(instr.node_details_dict.values()):
      for idx, node in enumerate(sorted_nodes):
        io_node_obj= instr.node_details_dict[node]
        bank= io_node_obj.bank
        pos= io_node_obj.pos
        st_en |= (1 << bank) 
        if instr.is_type('st'):
          all_bank_addr |= pos << (bank * BANK_ADDR_L)
        else:
          all_bank_addr |= pos << (idx * BANK_ADDR_L)

      curr_bin= bin_update(curr_bin, st_en, EN_L)
      curr_bin= bin_update(curr_bin, all_bank_addr, BANK_RD_ADDR_L)
      
      # mem addr
      assert in_range(instr.mem_addr, CURR_MEM_ADDR_L)
      curr_bin= bin_update(curr_bin, instr.mem_addr, CURR_MEM_ADDR_L)

      assert len(curr_bin) <= TOTAL_L, f"{len(curr_bin), TOTAL_L}"
        
    # TODO: Following instruction types
#    elif instr.is_type('st_2'):
#    elif instr.is_type('st_4'):
#    elif instr.is_type('sh_2'):
#    elif instr.is_type('sh_4'):
    
    elif instr.is_type('sh'):
      curr_bin= bin_update(curr_bin, CP_OPCODE, OPCODE_L)

      # wr_en and rd_en
      # bank addr, crossbar sel, invalids
      wr_en= 0
      rd_en= 0
      all_bank_addr= 0
      all_crossbar_sel= 0
      all_invalids= 0
      for node, src_dst_tup in list(instr.sh_dict_bank.items()):
        src_bank= src_dst_tup[0]
        dst_bank= src_dst_tup[1]
        rd_en |= (1<< src_bank) 
        wr_en |= (1<< dst_bank) 

        src_pos = instr.sh_dict_pos[node][0]
        all_bank_addr |= src_pos << (src_bank * BANK_ADDR_L)

        if node in instr.invalidate_node_set:
          all_invalids |= (1 << src_bank)

        all_crossbar_sel |= src_bank << (dst_bank * CROSSBAR_SEL_L)

      curr_bin= bin_update(curr_bin, wr_en, CP_WR_EN_L)
      curr_bin= bin_update(curr_bin, rd_en, CP_RD_EN_L)
      curr_bin= bin_update(curr_bin, all_bank_addr, CP_BANK_RD_ADDR_L)
      curr_bin= bin_update(curr_bin, all_crossbar_sel, CP_CROSS_SEL_L)
      curr_bin= bin_update(curr_bin, all_invalids, CP_INVLD_L)

    # elif instr.is_type('sh') or instr.is_type('sh_8'):
    elif instr.is_type('sh_8') or instr.is_type('sh_4') or instr.is_type('sh_2'):
      # instr.print_details()

      # if temp_instr_idx > start_temp_idx and temp_instr_idx < last_temp_idx:
      #   print(temp_instr_idx, 'cp')
      if instr.is_type('sh_8'):
        OPCODE         = CP_8_OPCODE
        EN_L           = CP_8_EN_L
        BANK_RD_ADDR_L = CP_8_BANK_RD_ADDR_L
        CROSS_SEL_L    = CP_8_CROSS_SEL_L
        INVLD_L        = CP_8_INVLD_L
      elif instr.is_type('sh_4'):
        OPCODE         = CP_4_OPCODE
        EN_L           = CP_4_EN_L
        BANK_RD_ADDR_L = CP_4_BANK_RD_ADDR_L
        CROSS_SEL_L    = CP_4_CROSS_SEL_L
        INVLD_L        = CP_4_INVLD_L
      elif instr.is_type('sh_2'):
        OPCODE         = CP_2_OPCODE
        EN_L           = CP_2_EN_L
        BANK_RD_ADDR_L = CP_2_BANK_RD_ADDR_L
        CROSS_SEL_L    = CP_2_CROSS_SEL_L
        INVLD_L        = CP_2_INVLD_L
      else:
        assert 0

      curr_bin= bin_update(curr_bin, OPCODE, OPCODE_L)

      assert len(instr.sh_dict_bank) <= 8

      # cp_en
      cp_en= 0
      for src_dst_tup in list(instr.sh_dict_bank.values()):
        dst_bank= src_dst_tup[1]
        cp_en |= (1<< dst_bank) 
      curr_bin= bin_update(curr_bin, cp_en, EN_L)
      
      
      # -- sort according to dst banks
      dst_to_node_dict={}
      dst_to_invl_dict={}
      for node, src_dst_tup in list(instr.sh_dict_bank.items()):
        dst_bank= src_dst_tup[1]
        dst_to_node_dict[dst_bank]= node
        if node in instr.invalidate_node_set:
          dst_to_invl_dict[dst_bank] = True
        else: #NOTE: newly added, not sure if this correct. This is to remove the error occuring in "if dst_to_invl_dict[dst_bank]:" statement below
          dst_to_invl_dict[dst_bank] = False
      assert len(dst_to_node_dict) == len(instr.sh_dict_bank)," Repeating dst_bank"
      
      # bank addr, crossbar sel, invalids
      all_bank_addr= 0
      all_crossbar_sel= 0
      all_invalids= 0
      for idx, dst_bank in enumerate(sorted(dst_to_node_dict.keys())):
        node= dst_to_node_dict[dst_bank]

        src_bank= instr.sh_dict_bank[node][0]
        all_crossbar_sel |= src_bank << (idx * CROSSBAR_SEL_L)

        src_pos= instr.sh_dict_pos[node][0]
        all_bank_addr |= src_pos << (idx * BANK_ADDR_L)
      
        if dst_to_invl_dict[dst_bank]:
          all_invalids |= (1 << idx)

      curr_bin= bin_update(curr_bin, all_bank_addr, BANK_RD_ADDR_L)
      curr_bin= bin_update(curr_bin, all_crossbar_sel, CROSS_SEL_L)
      curr_bin= bin_update(curr_bin, all_invalids, INVLD_L)
      
    elif instr.is_type('bb'):
      curr_bin= bin_update(curr_bin, BB_OPCODE, OPCODE_L)
      
      # invalids
      all_invalids= 0
      for node in instr.invalidate_node_set:
        all_invalids |= (1 << instr.in_node_details_dict[node].bank)
      assert in_range(all_invalids, BB_INVLD_L)
      curr_bin= bin_update(curr_bin, all_invalids, BB_INVLD_L)

      # bank addr, crossbar_sel
      debug_reg_re= 0
      all_bank_addr= 0
      all_crossbar_sel= 0
      for pe_tup, pe_details in list(instr.pe_details.items()):
        if pe_details.is_leaf():
          tree = pe_tup[0]
          pe =  pe_tup[2]

          # input 0
          if pe_details.input_0_reg != None:
            input_idx= tree * (2**TREE_DEPTH) + pe * 2
            bank= pe_details.input_0_reg[0]
            pos= pe_details.input_0_reg[1]

            all_crossbar_sel |= bank << (input_idx * CROSSBAR_SEL_L)
            all_bank_addr |= pos << (bank * BANK_ADDR_L)
            debug_reg_re |= (1 << bank)
            
            assert not pe_details.is_pass_1()
          else:
            assert not pe_details.is_sum()
            assert not pe_details.is_prod()
          
          # input 1
          if pe_details.input_1_reg != None:
            input_idx= tree * (2**TREE_DEPTH) + pe * 2 + 1
            bank= pe_details.input_1_reg[0]
            pos= pe_details.input_1_reg[1]

            all_crossbar_sel |= bank << (input_idx * CROSSBAR_SEL_L)
            all_bank_addr |= pos << (bank * BANK_ADDR_L)
            debug_reg_re |= (1 << bank)

            assert not pe_details.is_pass_0()
          else:
            assert not pe_details.is_sum()
            assert not pe_details.is_prod()
      
      curr_bin= bin_update(curr_bin, all_bank_addr, BB_BANK_RD_ADDR_L)
      curr_bin= bin_update(curr_bin, all_crossbar_sel, BB_CROSS_SEL_L)

      # arith_op, 
      all_arith_op = 0
      for pe_tup, pe_details in list(instr.pe_details.items()):
        tree = pe_tup[0]
        lvl  = pe_tup[1] 
        pe   = pe_tup[2]
        if pe_details.is_sum():
          arith_bin= SUM
        elif pe_details.is_prod():
          arith_bin= PROD 
        elif pe_details.is_pass_0():
          arith_bin= PASS_0
        elif pe_details.is_pass_1():
          arith_bin= PASS_1
        else:
          assert 0

        arith_idx= tree * N_ALU_PER_TREE + (2**(TREE_DEPTH-lvl)) - 1 + pe
        arith_bin <<= (arith_idx * ARITH_L)
        all_arith_op |= arith_bin
        
      curr_bin= bin_update(curr_bin, all_arith_op, BB_ARITH_OP_L)

      # bank_wr_sel
      all_bank_wr_sel = 0
      for pe_tup, pe_details in list(instr.pe_details.items()):
        if pe_details.output_reg != None:
          assert (pe_details.is_sum() or pe_details.is_prod())
          lvl  = pe_tup[1] 
          bank= pe_details.output_reg[0]
          all_bank_wr_sel |= lvl << (bank * BANK_WR_SEL_L)
      curr_bin= bin_update(curr_bin, all_bank_wr_sel, BB_BANK_WR_SEL_L)

      # if temp_instr_idx > start_temp_idx and temp_instr_idx < last_temp_idx:
      #   if temp_instr_idx == 29:
          # printcol(temp_instr_idx, 'blue')
          # printcol('bb', 'red')
          # print(bin(all_bank_addr))
          # print(hex(debug_reg_re))
      
      if verb: 
        printlog('bb')
        printlog('invalid ' + bin(all_invalids), 'red')
        printlog('bank_addr ' + bin(all_bank_addr), 'red')
        printlog('crossbar_sel ' + bin(all_crossbar_sel), 'red')
        printlog('arith ' + bin(all_arith_op), 'red')
        printlog('bank_wr_sel ' + bin(all_bank_wr_sel), 'red')

    else:
      assert False
  
    i_bin= str_update(i_bin, curr_bin)
    
    if verb: printlog(instr.name + ' ' + curr_bin)

  # make i_bin multiples of INSTR_L
  ciel_multiple= (len(i_bin) + INSTR_L - 1) // INSTR_L
  i_bin= i_bin.zfill(ciel_multiple * INSTR_L)
  
  # split i_bin in multiples of INSTR_L
  i_bin_chunks= [i_bin[i : i+INSTR_L] for i in range(0, len(i_bin), INSTR_L)]
  
  # reverse to bring the first instruction first
  i_bin_chunks.reverse()

  with open(file_name, 'w+') as f:
    printcol(f"Writing to file {file_name}", 'red')
    for bin_str in i_bin_chunks:
      f.write(bin_str)
      f.write('\n')
  

def use_shorter_st_instr(hw_details, MAX_COMPRESSED_BANKS, instr_ls):
  instr_type = {4 : 'st_4', 8 : 'st_8'}
  possible_shorter_versions= list(sorted(list(instr_type.keys())))

  compatible_versions= [x for x in possible_shorter_versions if x <= MAX_COMPRESSED_BANKS]
  
  count= 0
  total_count= 0
  for instr in instr_ls:
    if instr.is_type('st'):
      total_count += 1
      for compressed_len in compatible_versions: # important that compatible_versions is sorted
        if len(instr.node_set) <= compressed_len:
          instr.name= instr_type[compressed_len]
          assert instr.is_type(instr_type[compressed_len])
          count += 1
          break

  logger.info(f"Number of shorter store instructions: {count} out of total_count: {total_count}")

def use_shorter_sh_instr(hw_details, MAX_COMPRESSED_BANKS, instr_ls):
  instr_type = {2: 'sh_2', 4 : 'sh_4', 8 : 'sh_8'}
  possible_shorter_versions= sorted(list(instr_type.keys()))

  compatible_versions= [x for x in possible_shorter_versions if x <= MAX_COMPRESSED_BANKS]
  
  count= 0
  total_count= 0
  for instr in instr_ls:
    if instr.is_type('sh'):
      total_count += 1
      for compressed_len in compatible_versions: # important that compatible_versions is sorted
        if len(instr.sh_dict_bank) <= compressed_len:
          instr.name= instr_type[compressed_len]
          assert instr.is_type(instr_type[compressed_len])
          count += 1
          break

  logger.info(f"Number of shorter shift instructions: {count} out of total_count: {total_count}")

def str_update(i_bin, new_bin):
  return new_bin + i_bin

def bin_update(curr_bin, input_num, input_len):
  """
    Appends input_num to curr_bin
  """
  assert isinstance(input_num, int) or isinstance(input_num, int)
  assert isinstance(input_len, int)
  assert in_range(input_num, input_len) ,"num exceeds len"

  input_num_bin = in_bin(input_num, input_len)
  return str_update(curr_bin, input_num_bin)

def in_bin(input_num, bin_len):
  """
    in_bin stands for "in binary"
  """
  assert len(bin(input_num)) - 2 <= bin_len
  format_str= "{0:0"+ str(bin_len) + "b}"
  return format_str.format(input_num)


def in_range(num, len):
  if num < 2**len:
    return True
  else:
    return False
