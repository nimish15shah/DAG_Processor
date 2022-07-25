//=======================================================================
// Created by         : KU Leuven
// Filename           : /users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/src/common_pkg.sv
// Author             : Nimish Shah
// Created On         : 2019-10-21 16:53
// Last Modified      : 
// Update Count       : 2019-10-21 16:53
// Description        : 
//                      
//=======================================================================


`ifndef GENERAL_PKG
  `define GENERAL_PKG

  `ifdef GATE_NETLIST
    `define USE_SRAM_MACRO
  `endif

  // comment the following for synthesis
  /* `define VERIFICATION */
  
  /* `define TSMC_INITIALIZE_MEM // initize memory macros from a file */

  `define USE_SRAM_MACRO // uncomment to use sram macros for memory

  `ifdef VERIFICATION
    `define USE_SRAM_VERILOG_MODELS
  `endif

// Do not enable following for synthesis, only use for simulation
  `ifdef USE_SRAM_VERILOG_MODELS
    `define UNIT_DELAY // For faster SRAM sims */
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb8192x32m8swaso_180a/VERILOG/ts1n28hpcphvtb8192x32m8swaso_180a_tt0p9v25c.v"
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb1024x32m4swaso_180a/VERILOG/ts1n28hpcphvtb1024x32m4swaso_180a_tt0p9v25c.v"
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb512x32m4swaso_180a/VERILOG/ts1n28hpcphvtb512x32m4swaso_180a_tt0p9v25c.v"
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb256x32m4swaso_180a/VERILOG/ts1n28hpcphvtb256x32m4swaso_180a_tt0p9v25c.v"
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb1024x24m4swaso_180a/VERILOG/ts1n28hpcphvtb1024x24m4swaso_180a_tt0p9v25c.v"
    `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb512x24m4swaso_180a/VERILOG/ts1n28hpcphvtb512x24m4swaso_180a_tt0p9v25c.v"
  //  `include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts1n28hpcphvtb256x24m4swaso_180a/VERILOG/ts1n28hpcphvtb256x24m4swaso_180a_tt0p9v25c.v"
  `endif

  package general_pkg;
    parameter RESET_STATE= 0;
  endpackage
  
  import general_pkg::*;
`endif

`ifndef HW_CONFIG_PKG
  `define HW_CONFIG_PKG
  /* `define REG_BANK_DEPTH 32 */
  `define REG_BANK_DEPTH 256
  /* `define TREE_DEPTH 2 */
  /* `define N_TREE 8 */
  `define TREE_DEPTH 3
  `define MIN_DEPTH 2
  `define N_TREE 8
  /* `define DATA_MEM_SIZE 512 // KB */
  /* `define INSTR_MEM_SIZE 512 // KB */
  `define DATA_MEM_SIZE 2048 // KB
  `define INSTR_MEM_SIZE 512 // KB
  `define SRAM_MACRO_WIDTH 32 // b

  /* `define SRAM_RF */
  
  `define SIMPLE_INVALID // invalidation will reflect in the next cycle. i.e. invalidate register will be available for writing in the next cycle

  `ifdef SRAM_RF
    `ifndef SIMPLE_INVALID
      `define SIMPLE_INVALID
    `endif
  `endif

  package hw_config_pkg;
    
    parameter BIT_L            = 32;
    parameter EXP_L            = 8;
    parameter MNT_L            = BIT_L - EXP_L - 1;
    parameter TREE_DEPTH       = `TREE_DEPTH;
    parameter N_TREE           = `N_TREE;
    parameter N_IN_PER_TREE    = (2**TREE_DEPTH);
    parameter N_IN             = N_TREE * N_IN_PER_TREE;
    parameter N_ALU_PER_TREE   = ((2**TREE_DEPTH)-1);
    parameter N_ALU            = N_TREE * N_ALU_PER_TREE;
    parameter N_BANKS          = N_IN;
    parameter int BANK_DEPTH   = `REG_BANK_DEPTH;
    parameter DATA_MEM_RD_LATENCY= 1;
    parameter INSTR_MEM_RD_LATENCY= 1;

    /* parameter DATA_MEM_ADDR_L       = 12; */
    parameter DATA_MEM_ADDR_L       = $clog2((`DATA_MEM_SIZE*1024/4) / N_BANKS);

    //===========================================
    //       Pipestages
    //  
    //    RegBank -> Crossbar | tree lvl_1 | tree lvl_2 | ... | tree_lvl_top | -> to regbank
    //===========================================
    parameter PIPE_STAGES= TREE_DEPTH + 2;

    typedef logic [BIT_L - 1 : 0] word_t;
    typedef reg [BIT_L - 1 : 0] word_t_reg; // Needed for non-resettable Flipflop
    typedef word_t [N_BANKS - 1 : 0] mem_word_t;

  endpackage
  
  import hw_config_pkg::*;
`endif 

