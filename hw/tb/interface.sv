//=======================================================================
// Created by         : KU Leuven
// Filename           : interface.sv
// Author             : Nimish Shah
// Created On         : 2019-10-21 16:53
// Last Modified      : 
// Update Count       : 2019-10-21 16:53
// Description        : 
//                      
//=======================================================================
`ifndef INTERFACE_DEF
  `define INTERFACE_DEF

  `include "common_tb.sv"

interface intf(input logic clk);
  logic rst;

  logic enable_execution;
  logic [INSTR_L-1 : 0] init_instr;
  instr_addr_t init_instr_addr;
  logic init_instr_we;
  logic io_ping_wr;
  instr_addr_t current_instr_rd_addr;

  word_t init_data_in;
  word_t init_data_out;
  logic [$clog2(N_BANKS) + DATA_MEM_ADDR_L - 1 : 0] init_data_addr;
  logic init_data_we;
  logic init_data_re;

  // clocking block
  clocking cb @(posedge clk);
    default input#0.2ns output#0.2ns;

    output rst;

    output enable_execution;
    output init_instr;
    output init_instr_addr;
    output init_instr_we;
    output io_ping_wr;
    input current_instr_rd_addr;

    output init_data_in;
    input init_data_out;
    output init_data_addr;
    output init_data_we;
    output init_data_re;
  endclocking

  modport tb_port (clocking cb); // synchronous TB

endinterface

`endif //INTERFACE_DEF
