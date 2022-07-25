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

/* `define RUN_FROM_SCRIPT */

`ifdef RUN_FROM_SCRIPT
  `include "common_pkg_SCRIPT.sv"
`else
  `include "common_pkg.sv"
`endif

/* `define DEBUG */

`include "instr_decd_pkg.sv"

  // Include DUT files
`ifdef GATE_NETLIST
  // Add netlist path and tech libraries
`else
  `include "pru_sync.sv"
`endif

  // Common parameters, typedefs, enums
package common_types_params;
endpackage

`endif //COMMON_TB_DEF

