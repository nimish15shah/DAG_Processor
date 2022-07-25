//=======================================================================
// Created by         : KU Leuven
// Filename           :instr_decd_pkg.sv
// Author             : Nimish Shah
// Created On         :2019-09-25 16:26
// Last Modified      : 
// Update Count       :2019-09-25 16:26
// Description        :
//                     
//=======================================================================
`ifdef RUN_FROM_SCRIPT
  `include "common_pkg_SCRIPT.sv"
`else
  `include "common_pkg.sv"
`endif

`ifndef CONTROLLER_TYPEDEFS
  `define CONTROLLER_TYPEDEFS
  
  package controller_type_defs;
    import hw_config_pkg::*;
    typedef enum logic [1:0] {SUM=0, PROD, PASS_0, PASS_1} alu_mode_enum_t;
    typedef alu_mode_enum_t [N_TREE-1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_mode_t;
    typedef enum logic [1:0] {ALU=0, RAM, CROSSBAR} reg_wr_mode_t;
//    typedef enum logic [$clog2(`TREE_DEPTH + 1) - 1: 0] {LVL_[0:`TREE_DEPTH]=0} reg_wr_sel_enum_t;
//    typedef reg_wr_sel_t [N_BANKS-1 : 0] reg_wr_sel_t;
    typedef logic [N_BANKS-1 : 0] [$clog2(`TREE_DEPTH+1) - 1: 0] reg_wr_sel_t;

    typedef logic [N_BANKS - 1 : 0] [$clog2(N_BANKS) - 1 : 0] crossbar_sel_t;
    typedef logic [DATA_MEM_ADDR_L-1 : 0] ram_addr_t;
    typedef logic [N_BANKS -1: 0] ram_we_t;
    typedef logic [N_BANKS -1: 0] ram_re_t;
    typedef logic [N_BANKS-1 : 0] [$clog2(BANK_DEPTH)-1 : 0] reg_rd_addr_t;
    typedef logic [N_BANKS -1: 0] reg_inv_t;
    typedef logic [N_BANKS -1: 0] reg_we_t;
    typedef logic [N_BANKS -1: 0] reg_re_t;
    
    function automatic alu_mode_enum_t op2alu_mode(logic[$bits(alu_mode_enum_t)-1 : 0] op);
      case(op)
        2'b00 : return SUM;
        2'b01 : return PROD;
        2'b10 : return PASS_0;
        2'b11 : return PASS_1;
        default : $fatal(1);
      endcase // case (op)
    endfunction : op2alu_mode

  endpackage

  import controller_type_defs::*;
  

/*  
module temporay_test(); 
//  initial $display($bits(N_BANKS), $bits(DATA_MEM_ADDR_L), $clog2(N_BANKS), $clog2(DATA_MEM_ADDR_L));
crossbar_sel_t crossbar_sel;
logic [$clog2(N_BANKS) - 1: 0][7:0] A;
logic [N_BANKS][7:0] B;
initial begin
  $display($bits(crossbar_sel), $size(crossbar_sel), $bits(A), $size(A), $bits(B), $size(B));
  $display(4%6);
  $display($clog2(BANK_DEPTH));
end
endmodule
*/

`endif


`ifndef OPCODES
  `define OPCODES
  package opcodes;
    parameter OPCODE_L  = 4;
    parameter NOP_STALL = 0;
    parameter NOP       = 1;
    parameter BB        = 2;
    parameter CP_8      = 3;
    parameter ST        = 4;
    parameter LD        = 5;
    parameter ST_4      = 6;
    parameter ST_8      = 7;
    parameter CP_2      = 8;
    parameter CP_4      = 9;
    parameter CP      = 10;

//    typedef enum logic [OPCODE_L - 1 : 0] {NOP=0, NOP_STALL=1, BB=2, CP_8=3, ST=4, LD=5, ST_4=6, ST_8=7, CP_2=8, CP_4=9} opcode_t;
  endpackage: opcodes

  import opcodes::*;
`endif  

`ifndef INSTR_DECD_PKG
  `define INSTR_DECD_PKG
  package instr_decd_pkg;
    import hw_config_pkg::*;
    import controller_type_defs::*;
    import opcodes::*;
    int MAX_INSTR_L= 0;
    parameter BANK_ADDR_L    = $clog2(BANK_DEPTH);
    parameter CROSSBAR_SEL_L = $clog2(N_BANKS);

    // Instruction formats
    //===========================================
    // *_S : indicates begining position of a field (inclusive)
    // *_L  : indicates the length
    // Usage: instr[_S + _L -1 : _S]
    //===========================================

    //===========================================
    //  nop
    //  | opcode |
    //===========================================
    parameter NOP_OPCODE = NOP;
    parameter NOP_L      = OPCODE_L;

    //===========================================
    //  nop_stall
    //  | opcode |
    //===========================================
    parameter NOP_STALL_OPCODE = NOP_STALL;
    parameter NOP_STALL_L      = OPCODE_L;

    //===========================================
    //  ld
    //  | opcode | ld_en | mem addr |
    //===========================================
    parameter LD_OPCODE     = LD;
    parameter LD_EN_S       = OPCODE_L; // Start bit for ld_en (this is the exact bit to slice from);
    parameter LD_EN_L       = N_BANKS;
    parameter LD_MEM_ADDR_S = LD_EN_S + LD_EN_L ;
    parameter LD_MEM_ADDR_L = DATA_MEM_ADDR_L;
    parameter LD_L          = OPCODE_L + LD_MEM_ADDR_L + LD_EN_L;

    //===========================================
    //  st
    //  | opcode | st_en | bank rd addr x N_BANKS |  mem addr |
    //===========================================
    parameter ST_OPCODE         = ST;
    parameter ST_EN_S           = OPCODE_L;
    parameter ST_EN_L           = N_BANKS;
    parameter ST_BANK_RD_ADDR_S = ST_EN_S + ST_EN_L;
    parameter ST_BANK_RD_ADDR_L = BANK_ADDR_L * N_BANKS;
    parameter ST_MEM_ADDR_S     = ST_BANK_RD_ADDR_S + ST_BANK_RD_ADDR_L;
    parameter ST_MEM_ADDR_L     = DATA_MEM_ADDR_L;
    parameter ST_L              = OPCODE_L + ST_BANK_RD_ADDR_L + ST_MEM_ADDR_L + ST_EN_L;


    //===========================================
    //  st_4
    //  | opcode | st_en | bank rd addr x 4 | mem addr |
    //===========================================
    parameter ST_4_OPCODE         = ST_4;
    parameter ST_4_EN_S           = OPCODE_L;
    parameter ST_4_EN_L           = N_BANKS;
    parameter ST_4_BANK_RD_ADDR_S = ST_4_EN_S + ST_4_EN_L;
    parameter ST_4_BANK_RD_ADDR_L = BANK_ADDR_L * 4;
    parameter ST_4_MEM_ADDR_S     = ST_4_BANK_RD_ADDR_S + ST_4_BANK_RD_ADDR_L;
    parameter ST_4_MEM_ADDR_L     = DATA_MEM_ADDR_L;
    parameter ST_4_L              = OPCODE_L + ST_4_BANK_RD_ADDR_L + ST_4_MEM_ADDR_L + ST_4_EN_L;

    

    //===========================================
    //  st_8
    //  | opcode | st_en | bank rd addr x 8 | mem addr |
    //===========================================
    parameter ST_8_OPCODE         = ST_8;
    parameter ST_8_EN_S           = OPCODE_L;
    parameter ST_8_EN_L           = N_BANKS;
    parameter ST_8_BANK_RD_ADDR_S = ST_8_EN_S + ST_8_EN_L;
    parameter ST_8_BANK_RD_ADDR_L = BANK_ADDR_L * 8;
    parameter ST_8_MEM_ADDR_S     = ST_8_BANK_RD_ADDR_S + ST_8_BANK_RD_ADDR_L;
    parameter ST_8_MEM_ADDR_L     = DATA_MEM_ADDR_L;
    parameter ST_8_L              = OPCODE_L + ST_8_BANK_RD_ADDR_L + ST_8_MEM_ADDR_L + ST_8_EN_L;

    //===========================================
    //      cp
    //      | opcode | cp_en | bank_rd_addr * 8 | crossbar_sel * 8 | invalidates | 
    //===========================================
    parameter CP_OPCODE         = CP;
    parameter CP_WR_EN_S           = OPCODE_L;
    parameter CP_WR_EN_L           = N_BANKS;
    parameter CP_RD_EN_S           = CP_WR_EN_S + CP_WR_EN_L;
    parameter CP_RD_EN_L           = N_BANKS;
    parameter CP_BANK_RD_ADDR_S = CP_RD_EN_S + CP_RD_EN_L;
    parameter CP_BANK_RD_ADDR_L = BANK_ADDR_L * N_BANKS;
    parameter CP_CROSS_SEL_S    = CP_BANK_RD_ADDR_S + CP_BANK_RD_ADDR_L;
    parameter CP_CROSS_SEL_L    = CROSSBAR_SEL_L * N_BANKS;
    parameter CP_INVLD_S        = CP_CROSS_SEL_S + CP_CROSS_SEL_L;
    parameter CP_INVLD_L        = N_BANKS;
    parameter CP_L              = OPCODE_L + CP_WR_EN_L + CP_RD_EN_L + CP_BANK_RD_ADDR_L + CP_CROSS_SEL_L + CP_INVLD_L;

    //===========================================
    //      cp_8
    //      | opcode | cp_en | bank_rd_addr * 8 | crossbar_sel * 8 | invalidates | 
    //===========================================
    parameter CP_8_OPCODE         = CP_8;
    parameter CP_8_EN_S           = OPCODE_L;
    parameter CP_8_EN_L           = N_BANKS;
    parameter CP_8_BANK_RD_ADDR_S = CP_8_EN_S + CP_8_EN_L;
    parameter CP_8_BANK_RD_ADDR_L = BANK_ADDR_L * 8;
    parameter CP_8_CROSS_SEL_S    = CP_8_BANK_RD_ADDR_S + CP_8_BANK_RD_ADDR_L;
    parameter CP_8_CROSS_SEL_L    = CROSSBAR_SEL_L * 8;
    parameter CP_8_INVLD_S        = CP_8_CROSS_SEL_S + CP_8_CROSS_SEL_L;
    parameter CP_8_INVLD_L        = 8;
    parameter CP_8_L              = OPCODE_L + CP_8_EN_L + CP_8_BANK_RD_ADDR_L + CP_8_CROSS_SEL_L + CP_8_INVLD_L;

    //===========================================
    //       cp_4
    //  | opcode | cp_en | bank_rd_addr * 4 | crossbar_sel * 4 | invalidates | 
    //===========================================
    parameter CP_4_OPCODE         = CP_4;
    parameter CP_4_EN_S           = OPCODE_L;
    parameter CP_4_EN_L           = N_BANKS;
    parameter CP_4_BANK_RD_ADDR_S = CP_4_EN_S + CP_4_EN_L;
    parameter CP_4_BANK_RD_ADDR_L = BANK_ADDR_L * 4;
    parameter CP_4_CROSS_SEL_S    = CP_4_BANK_RD_ADDR_S + CP_4_BANK_RD_ADDR_L;
    parameter CP_4_CROSS_SEL_L    = CROSSBAR_SEL_L * 4;
    parameter CP_4_INVLD_S        = CP_4_CROSS_SEL_S + CP_4_CROSS_SEL_L;
    parameter CP_4_INVLD_L        = 4;
    parameter CP_4_L              = OPCODE_L + CP_4_EN_L + CP_4_BANK_RD_ADDR_L + CP_4_CROSS_SEL_L + CP_4_INVLD_L;

    //===========================================
    //       cp_2
    //      | opcode | cp_en | bank_rd_addr * 2 | crossbar_sel * 2 | invalidates | 
    //===========================================
    parameter CP_2_OPCODE         = CP_2;
    parameter CP_2_EN_S           = OPCODE_L;
    parameter CP_2_EN_L           = N_BANKS;
    parameter CP_2_BANK_RD_ADDR_S = CP_2_EN_S + CP_2_EN_L;
    parameter CP_2_BANK_RD_ADDR_L = BANK_ADDR_L * 2;
    parameter CP_2_CROSS_SEL_S    = CP_2_BANK_RD_ADDR_S + CP_2_BANK_RD_ADDR_L;
    parameter CP_2_CROSS_SEL_L    = CROSSBAR_SEL_L * 2;
    parameter CP_2_INVLD_S        = CP_2_CROSS_SEL_S + CP_2_CROSS_SEL_L;
    parameter CP_2_INVLD_L        = 2;
    parameter CP_2_L              = OPCODE_L + CP_2_EN_L + CP_2_BANK_RD_ADDR_L + CP_2_CROSS_SEL_L + CP_2_INVLD_L;

    //===========================================
    //       bb
    //      | opcode | invalidates | bank_rd_addr * N_BANKS | crossbar_sel * N_BANKS | arith_op * N_ALU | bank_wr_sel * N_BANKS |
    //===========================================
    parameter BB_OPCODE              = BB;
    parameter BB_INVLD_S             = OPCODE_L;
    parameter BB_INVLD_L             = N_BANKS;
    parameter BB_BANK_RD_ADDR_S      = BB_INVLD_S + BB_INVLD_L;
    parameter BB_BANK_RD_ADDR_L      = BANK_ADDR_L * N_BANKS;
    parameter BB_CROSS_SEL_S         = BB_BANK_RD_ADDR_S + BB_BANK_RD_ADDR_L;
    parameter BB_CROSS_SEL_L         = CROSSBAR_SEL_L * N_BANKS;
    parameter BB_ARITH_OP_S          = BB_CROSS_SEL_S + BB_CROSS_SEL_L;
    parameter BB_ARITH_OP_L          = $bits(alu_mode_enum_t) * N_ALU;
    parameter BB_BANK_WR_SEL_S       = BB_ARITH_OP_S + BB_ARITH_OP_L;
    parameter BB_BANK_WR_SEL_L       = $clog2(`TREE_DEPTH + 1) * N_BANKS;
    parameter BB_L                   = OPCODE_L + BB_INVLD_L + BB_BANK_RD_ADDR_L + BB_CROSS_SEL_L + BB_ARITH_OP_L + BB_BANK_WR_SEL_L;
    
    /* parameter INSTR_L                = BB_L; */
    parameter INSTR_L                = get_max_l(NOP_L, ST_L, LD_L, CP_L, BB_L, 0);
    parameter MIN_COMPRESSION_FACTOR = 4; // For compressed store and copy instructions
    parameter MAX_COMPRESSED_BANKS   = (MIN_COMPRESSION_FACTOR > N_BANKS) ? 1 : 
            (N_BANKS / MIN_COMPRESSION_FACTOR <= 8 ? N_BANKS / MIN_COMPRESSION_FACTOR : 8);

    typedef logic [INSTR_L - 1 : 0] instr_t;
    
    parameter int INSTR_L_MACRO_ADJUSTED= (INSTR_L + `SRAM_MACRO_WIDTH) / `SRAM_MACRO_WIDTH ;
    parameter INSTR_MEM_DEPTH  = 2 ** $clog2 ((`INSTR_MEM_SIZE * 1024/4)/INSTR_L_MACRO_ADJUSTED);
    /* parameter INSTR_MEM_DEPTH  = 4096; */
    parameter INSTR_MEM_ADDR_L = $clog2(INSTR_MEM_DEPTH);
    
    typedef logic [INSTR_MEM_ADDR_L - 1 : 0] instr_addr_t; 

    function automatic instr_decd_pkg_asserts(); // Call this in a modules' initial block
      assert (LD_L == LD_MEM_ADDR_S + DATA_MEM_ADDR_L);      
      assert (ST_L == ST_MEM_ADDR_S + ST_MEM_ADDR_L);
      assert (ST_4_L == ST_4_MEM_ADDR_S + ST_4_MEM_ADDR_L);
      assert (ST_8_L == ST_8_MEM_ADDR_S + ST_8_MEM_ADDR_L);
      assert (BB_L == BB_BANK_WR_SEL_S + BB_BANK_WR_SEL_L);
      assert (CP_2_L == CP_2_INVLD_S + CP_2_INVLD_L);
      assert (CP_8_L == CP_8_INVLD_S + CP_8_INVLD_L);
      assert (CP_4_L == CP_4_INVLD_S + CP_4_INVLD_L);
      
    endfunction

    function int get_max_l (int a1,a2,a3,a4,a5, a6);
      int max_l=0;
      max_l = a1 > max_l ? a1 : max_l;
      max_l = a2 > max_l ? a2 : max_l;
      max_l = a3 > max_l ? a3 : max_l;
      max_l = a4 > max_l ? a4 : max_l;
      max_l = a5 > max_l ? a5 : max_l;
      max_l = a6 > max_l ? a6 : max_l;

      return max_l;
    endfunction
  endpackage: instr_decd_pkg

  import instr_decd_pkg::*;

`endif
