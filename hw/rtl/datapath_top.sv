
`ifndef DATAPATH_TOP
  `define DATAPATH_TOP

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"
`include "module_library.sv"

`include "alu_trees.sv"
`include "pipelined_data.sv"
`include "write_back_logic.sv"

`ifdef SRAM_RF
  `include "register_banks_sram.sv"
`else
  `include "register_banks.sv"
`endif


module datapath_top (
  input clk, rst,

  input alu_mode_t      alu_mode     , 
  input crossbar_sel_t  crossbar_sel , 
  input                 pipe_en      , 
  input [N_BANKS - 1 : 0] crossbar_flop_en,
  input [N_BANKS - 1 : 0] crossbar_pipe_en,
  input reg_wr_mode_t   reg_wr_mode  , 
  input reg_wr_sel_t    reg_wr_sel   , 
  input reg_rd_addr_t   reg_rd_addr  , 
  input reg_we_t        reg_we       , 
  input reg_re_t        reg_re       , 
  input reg_inv_t       reg_inv      , 

  output word_t [N_BANKS - 1 : 0] mem_inputs,
  input word_t [N_BANKS - 1 : 0] mem_outputs
  ); 
  
  // Signals
  word_t [N_BANKS - 1: 0] reg_outputs;
  word_t [N_BANKS - 1: 0] reg_inputs;
  word_t [N_BANKS - 1: 0] crossbar_outputs_d;
  word_t [N_BANKS - 1: 0] crossbar_outputs_q;
  word_t [N_BANKS - 1: 0] crossbar_outputs_piped;
  logic [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_outputs_vld;
  word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_outputs;
  word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_outputs_piped;
  word_t [N_BANKS - 1 : 0] mem_outputs_piped;

  // Instances
  register_banks register_banks_ins(
    .clk        (clk         ), 
    .rst        (rst         ), 
    .pipe_en    (pipe_en     ), 
    .inputs     (reg_inputs  ), 
    .reg_we     (reg_we      ), 
    .reg_re     (reg_re      ), 
    .reg_inv    (reg_inv     ), 
    .reg_rd_addr(reg_rd_addr ), 
    .outputs    (reg_outputs )
  ); 

  crossbar #(.WORD_LEN(BIT_L), .IN_PORTS(N_BANKS), .OUT_PORTS(N_BANKS)) crossbar_ins (.input_words(reg_outputs), .sel(crossbar_sel), .output_words(crossbar_outputs_d));
  
  processing_block processing_block_ins(
    .clk     (clk               ), 
    .rst     (rst               ), 
    .en      (pipe_en   ), 
    .inputs  (crossbar_outputs_q), 
    .alu_mode(alu_mode  ), 
    .outputs_vld (alu_outputs_vld),
    .outputs (alu_outputs       )
  );

  pipelined_data pipelined_data_ins (
    .clk                  (clk                   ), 
    .rst                  (rst                   ), 
    .pipe_en              (pipe_en               ), 
    .crossbar_pipe_en (crossbar_pipe_en),
    .in_alu_trees_results (alu_outputs           ), 
    .in_alu_trees_results_vld(alu_outputs_vld    ), 
    .in_crossbar_output   (crossbar_outputs_q    ), 
    .out_crossbar_output  (crossbar_outputs_piped), 
    .mem_inputs   (mem_inputs), 
    .out_alu_trees_results(alu_outputs_piped     )
  );
    
  write_back_logic write_back_logic_ins(
    .reg_we      (reg_we), 
    .reg_wr_sel  (reg_wr_sel             ), 
    .reg_wr_mode (reg_wr_mode            ), 
    .alu_results (alu_outputs_piped      ), 
    .memory_out  (mem_outputs_piped      ), 
    .crossbar_out(crossbar_outputs_piped ), 
    .reg_inputs  (reg_inputs             ) 
  );

  // Logic if needed
  assign mem_outputs_piped = mem_outputs;
  /* assign mem_inputs        = reg_outputs; */

  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      foreach ( crossbar_outputs_q[i]) begin
        crossbar_outputs_q[i] <= 0;
      end
    end else begin
      foreach ( crossbar_outputs_q[i]) begin
        if (crossbar_flop_en[i] == 1) begin
          crossbar_outputs_q[i] <= crossbar_outputs_d[i];
        end
      end
    end
  end
endmodule

`endif //DATAPATH_TOP
