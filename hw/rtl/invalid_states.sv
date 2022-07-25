//=======================================================================
// Created by         : KU Leuven
// Filename           : invalid_states.sv
// Author             : Nimish Shah
// Created On         : 2019-11-04 11:20
// Last Modified      : 
// Update Count       : 2019-11-04 11:20
// Description        : 
//                      
//=======================================================================
`ifndef INVALID_STATES_DEF
  `define INVALID_STATES_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

module invalid_states (
  input clk, rst, 
  input pipe_en,
  input controller_type_defs::reg_we_t reg_we,
  input controller_type_defs::reg_inv_t reg_inv,
  input controller_type_defs::reg_rd_addr_t reg_rd_addr,
  output controller_type_defs::reg_rd_addr_t reg_wr_addr
  ); 

  logic [N_BANKS -1: 0] [BANK_DEPTH - 1 : 0] valid;
  reg_rd_addr_t reg_wr_addr_pre;

  /* Valid bit */
  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      foreach ( valid[i]) begin
        for (integer j= 0; j < $size(valid[i]); j= j+1) begin
          valid[i][j] <= 0;
        end
      end
    end else begin
      if (pipe_en) begin
        for (integer i=0; i < N_BANKS; i=i+1) begin          
`ifdef SIMPLE_INVALID
          if (reg_inv[i] == 1) begin
            valid[i][reg_rd_addr[i]] <= 1'b0;
          end 
          if (reg_we[i] == 1) begin
            valid[i][reg_wr_addr_pre[i]] <= 1'b1;
          end
          if (reg_we[i] == 1 && reg_inv[i]==1) begin
            assert (reg_rd_addr[i] != reg_wr_addr_pre[i]);
          end

`else
          if (reg_inv[i] == 1 && reg_we[i]== 1) begin
            if (reg_rd_addr[i] == reg_wr_addr_pre[i]) begin
              valid[i][reg_wr_addr_pre[i]] <= 1'b1;
            end else begin
              valid[i][reg_rd_addr[i]] <= 1'b0;
              valid[i][reg_wr_addr_pre[i]] <= 1'b1;
            end

          end else if (reg_inv[i] == 0 && reg_we[i]== 1) begin
            valid[i][reg_wr_addr_pre[i]] <= 1'b1;

          end else if (reg_inv[i] == 1 && reg_we[i]== 0) begin
            valid[i][reg_rd_addr[i]] <= 1'b0;
          end
`endif
        end
      end
    end
  end
  
  /* Priority encoding */
  always_comb begin
    foreach (reg_wr_addr_pre[i]) begin
      reg_wr_addr_pre[i] = '0;
    end
    for (integer i=0; i< N_BANKS; i=i+1) begin
      for (integer j=BANK_DEPTH - 1; j>= 0; j=j-1) begin // Need to go in descending order to create a priority encoder
`ifdef SIMPLE_INVALID
        if (valid[i][j] == 0) begin 
`else
        if ( (valid[i][j] == 0) || (reg_inv[i] == 1 && reg_rd_addr[i] == j)) begin//Either this reg is empty,
                                                                                  //or going to be free in this cycle
`endif
          reg_wr_addr_pre[i]= j;
        end
      end
    end
  end 

  // outputs
  assign reg_wr_addr= reg_wr_addr_pre;

endmodule
`endif //INVALID_STATES_DEF
