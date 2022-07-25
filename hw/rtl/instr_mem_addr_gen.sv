//=======================================================================
// Created by         : KU Leuven
// Filename           : instr_mem_addr_gen.sv
// Author             : Nimish Shah
// Created On         : 2019-11-04 10:04
// Last Modified      : 
// Update Count       : 2019-11-04 10:04
// Description        : 
//                      
//=======================================================================
`ifndef INSTR_MEM_ADDR_GEN
  `define INSTR_MEM_ADDR_GEN

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

//===========================================
//                      |                                                
//                      |                                                
//                      |     +--------------------+                     
//                      |     |                    |                     
// instr_mem_rd_addr <------- | instr_mem_addr_gen |                     
//                      |     |                    |                     
//                      |     +--------------------+                     
//                      |            ^     ^                             
//                      |            |     |            +---------------+
//         instr_vld  ---------------------|----------> |               |
//         fetcher_rdy<-------------------------------- | instr_fetcher |
//                      |                               +---------------+
//                      |                                                
//                      |                                                       
//
//===========================================
module instr_mem_addr_gen (
  input clk,
  input rst,
  input fetcher_rdy,
  input instr_vld,
  input instr_addr_t base_instr_addr,
  output instr_addr_t instr_mem_rd_addr,
  output instr_mem_rd_rdy,
  output instr_vld_to_fetcher
  ); 
  
  instr_addr_t addr_reg;
  logic addr_increament;
  assign addr_increament = instr_vld & fetcher_rdy;
  logic [INSTR_MEM_RD_LATENCY - 1 : 0] instr_vld_to_fetcher_q;

  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      addr_reg <= '0; 
      instr_vld_to_fetcher_q <= '0;
    end else begin
      if (addr_increament) begin
        addr_reg <= addr_reg + 1 + base_instr_addr;
        if (addr_reg % (INSTR_MEM_DEPTH-1) == 0) $display("Overflowing addr_reg at %d\n", addr_reg);
      end 
      if (fetcher_rdy) begin
        instr_vld_to_fetcher_q[0] <= instr_vld;
        for (integer i=1; i< INSTR_MEM_RD_LATENCY; i=i+1) begin
          instr_vld_to_fetcher_q[i] <= instr_vld_to_fetcher_q[i-1];
        end
      end
    end
  end
  
  assign instr_mem_rd_addr = addr_reg;
  assign instr_mem_rd_rdy = addr_increament;
  assign instr_vld_to_fetcher = instr_vld_to_fetcher_q[INSTR_MEM_RD_LATENCY - 1];

  initial begin
    assert (INSTR_MEM_RD_LATENCY >= 1) else $fatal("instr_vld_to_fetcher_q assumes that INSTR_MEM_RD_LATENCY == 1");
  end
endmodule



`endif //INSTR_MEM_ADDR_GEN
