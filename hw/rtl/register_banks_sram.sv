//-----------------------------------------------------------------------
// Created by         : KU Leuven
// Filename           : register_banks_sram.sv
// Author             : Nimish Shah
// Created On         : 2019-11-04 11:22
// Last Modified      : 
// Update Count       : 2019-11-04 11:22
// Description        : 
//                      
//-----------------------------------------------------------------------


`ifndef REGBANKS_SRAM_DEF
  `define REGBANKS_SRAM_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

`include "invalid_states.sv"

//`include "/users/micas/micas/design/tsmc28hpcplus/memories/Front_end/ts6n28hpcphvta64x16m2swsod_130a/VERILOG/ts6n28hpcphvta64x16m2swsod_130a_ssg0p63v0p81v0c.v"

module register_banks (
  input clk, rst,
  input pipe_en,
  input word_t [N_BANKS - 1 : 0] inputs,
  input controller_type_defs::reg_re_t reg_re,
  input controller_type_defs::reg_we_t reg_we,
  input controller_type_defs::reg_inv_t reg_inv,
  input controller_type_defs::reg_rd_addr_t reg_rd_addr,
  output word_t [N_BANKS - 1 : 0] outputs
  ); 
  import instr_decd_pkg::*;

  reg_rd_addr_t reg_wr_addr;

  /* Instances */
  invalid_states invalid_states_ins(.clk(clk), .rst(rst), 
                                    .pipe_en     (pipe_en    ), 
                                    .reg_we      (reg_we     ), 
                                    .reg_inv     (reg_inv    ), 
                                    .reg_rd_addr (reg_rd_addr), 
                                    .reg_wr_addr (reg_wr_addr));
  
  generate 
    localparam SRAM_DEPTH= 64;
    genvar bank_gen_i, depth_gen_i;
    for (bank_gen_i=0; bank_gen_i< N_BANKS ; bank_gen_i= bank_gen_i+1) begin : bank_loop
      for (depth_gen_i=0; depth_gen_i< BANK_DEPTH/SRAM_DEPTH ; depth_gen_i=depth_gen_i+1) begin : depth_loop
        TS6N28HPCPHVTA64X16M2SWSOD register_sram (
          .AA5   (reg_wr_addr[bank_gen_i][5]),
          .AA4   (reg_wr_addr[bank_gen_i][4]),
          .AA3   (reg_wr_addr[bank_gen_i][3]),
          .AA2   (reg_wr_addr[bank_gen_i][2]),
          .AA1   (reg_wr_addr[bank_gen_i][1]),
          .AA0   (reg_wr_addr[bank_gen_i][0]),
          .D15   (inputs[bank_gen_i][15]),
          .D14   (inputs[bank_gen_i][14]),
          .D13   (inputs[bank_gen_i][13]),
          .D12   (inputs[bank_gen_i][12]),
          .D11   (inputs[bank_gen_i][11]),
          .D10   (inputs[bank_gen_i][10]),
          .D9    (inputs[bank_gen_i][9]),
          .D8    (inputs[bank_gen_i][8]),
          .D7    (inputs[bank_gen_i][7]),
          .D6    (inputs[bank_gen_i][6]),
          .D5    (inputs[bank_gen_i][5]),
          .D4    (inputs[bank_gen_i][4]),
          .D3    (inputs[bank_gen_i][3]),
          .D2    (inputs[bank_gen_i][2]),
          .D1    (inputs[bank_gen_i][1]),
          .D0    (inputs[bank_gen_i][0]),
          .BWEB15(1'b0),
          .BWEB14(1'b0),
          .BWEB13(1'b0),
          .BWEB12(1'b0),
          .BWEB11(1'b0),
          .BWEB10(1'b0),
          .BWEB9 (1'b0),
          .BWEB8 (1'b0),
          .BWEB7 (1'b0),
          .BWEB6 (1'b0),
          .BWEB5 (1'b0),
          .BWEB4 (1'b0),
          .BWEB3 (1'b0),
          .BWEB2 (1'b0),
          .BWEB1 (1'b0),
          .BWEB0 (1'b0),
          .WEB   (~(reg_we[bank_gen_i] & pipe_en)),
          .CLKW  (clk),
          .AB5   (reg_rd_addr[bank_gen_i][5]),
          .AB4   (reg_rd_addr[bank_gen_i][4]),
          .AB3   (reg_rd_addr[bank_gen_i][3]),
          .AB2   (reg_rd_addr[bank_gen_i][2]),
          .AB1   (reg_rd_addr[bank_gen_i][1]),
          .AB0   (reg_rd_addr[bank_gen_i][0]),
          .REB   (~(reg_re[bank_gen_i] & pipe_en)),
          .CLKR  (clk),
          .SLP   (1'b0),
          .SD    (1'b0),
          .Q15   (outputs[bank_gen_i][15]),
          .Q14   (outputs[bank_gen_i][14]),
          .Q13   (outputs[bank_gen_i][13]),
          .Q12   (outputs[bank_gen_i][12]),
          .Q11   (outputs[bank_gen_i][11]),
          .Q10   (outputs[bank_gen_i][10]),
          .Q9    (outputs[bank_gen_i][9 ]),
          .Q8    (outputs[bank_gen_i][8 ]),
          .Q7    (outputs[bank_gen_i][7 ]),
          .Q6    (outputs[bank_gen_i][6 ]),
          .Q5    (outputs[bank_gen_i][5 ]),
          .Q4    (outputs[bank_gen_i][4 ]),
          .Q3    (outputs[bank_gen_i][3 ]),
          .Q2    (outputs[bank_gen_i][2 ]),
          .Q1    (outputs[bank_gen_i][1 ]),
          .Q0    (outputs[bank_gen_i][0 ])
        );
      end
    end
  endgenerate
  
  initial begin
    assert (BANK_DEPTH == 64);
    assert (BIT_L == 16);
  end

endmodule


`endif // REGBANKS_SRAM_DEF
