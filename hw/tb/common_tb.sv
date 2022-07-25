//=======================================================================
// Created by         : KU Leuven
// Filename           : common.sv
// Author             : Nimish Shah
// Created On         : 2019-10-21 16:54
// Last Modified      : 
// Update Count       : 2019-10-21 16:54
// Description        : 
//                      
//=======================================================================
`ifndef COMMON_TB_DEF
  `define COMMON_TB_DEF
  
/* `define GATE_NETLIST // Uncomment this for simulating gate-level netlist */

//`include "uvm_macros.svh"

`define VERIFICATION
/* `define DISPLAY_ERRORS */

`define RUN_FROM_SCRIPT

`ifdef RUN_FROM_SCRIPT
  `include "common_pkg_SCRIPT.sv"
`else
  `include "common_pkg.sv"
`endif

/* `define DEBUG */

`include "instr_decd_pkg.sv"

  // Include DUT files
`ifdef GATE_NETLIST
  `define UNIT_DELAY // For faster SRAM sims */
  `include "/users/micas/micas/design/tsmc28hpcplus/libs/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp12t30p140hvt_170a/tcbn28hpcplusbwp12t30p140hvt.v"
  `include "/users/micas/micas/design/tsmc28hpcplus/libs/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp12t30p140lvt_170a/tcbn28hpcplusbwp12t30p140lvt.v"
  `include "/users/micas/micas/design/tsmc28hpcplus/libs/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp12t30p140ulvt_170a/tcbn28hpcplusbwp12t30p140ulvt.v"
  `include "/users/micas/micas/design/tsmc28hpcplus/libs/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp12t30p140_170a/tcbn28hpcplusbwp12t30p140.v"
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_sync_SVT_512_1024_32_16_compiled.v" */
  `include "/esat/puck1/users/nshah/Backend/pru_sync/netlists_2/pru_sync_TREE_DEPTH_3_N_TREE_8_REG_BANK_DEPTH_256_MIN_DEPTH_2.v"
  //  `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_async_top_post_cts.v"
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_async_top.sta.v" */
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_async_top_SVT_512_1024_32_16_compiled.v" */
  /* `include "/esat/jupiter1/users/nshah/bayesian_network/hardware/backend//floorplan/netlist_with_data_gating_in_regs.v" */
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/SignOff/00_DESIGN/no_backup/pru_async_top.sta.v" */
  /* `include "/esat/centauri1/users/nshah/bayesian_network/hardware/backend/placeroute/optRoute.v" */
  /* `include "/volume1/users/nshah/bayesian_network/hardware/backend/placeroute/optRoute.v" */
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_async_top_SVT_512_1024_32_16_Wed_Mar_25_04-05-33_CET_2020_compiled.v" */
  // The following file runs without sdf, and it is the post-syn netlist that is tapedout, 405 config registers, 32b posit format seems to be <32,7>
  /* `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/pru_async_top_SVT_512_1024_32_16_Wed_Mar_25_19-39-34_CET_2020_compiled.v" */
//  `include "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/BackEnd/DesignDataIn/netlist/optRoute.v"

`else
  `ifdef INSTR_PING_PONG
    `include "pru_sync.sv"
    /* `include "pru_sync_ping_pong.sv" */
  `else
    `include "pru_sync.sv"
  `endif
`endif

  // Common parameters, typedefs, enums
package common_types_params;
endpackage

`endif //COMMON_TB_DEF

