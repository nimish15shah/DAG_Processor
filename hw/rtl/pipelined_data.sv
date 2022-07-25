
`ifndef PIPELINED_DATA_DEF
  `define PIPELINED_DATA_DEF

`include "common_pkg.sv"

module  pipelined_data (
  input clk, rst,
  input pipe_en,
  input [N_BANKS - 1 : 0] crossbar_pipe_en,
  input word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] in_alu_trees_results,
  input [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] in_alu_trees_results_vld,
  input word_t [N_BANKS - 1 : 0] in_crossbar_output,

  output word_t [N_BANKS - 1 : 0] out_crossbar_output,
  output word_t [N_BANKS - 1 : 0] mem_inputs,
  output word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] out_alu_trees_results
  ); 

  //===========================================
  //       Pipelining of ALU outputs
  //===========================================

  // Pipedepth of TREE_DEPTH -1 , as the top node need not be piped again
  word_t [TREE_DEPTH -2 : 0][N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_trees_results_pipe_q, alu_trees_results_pipe_d; 
  logic [TREE_DEPTH -2 : 0][N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_trees_results_pipe_vld_q, alu_trees_results_pipe_vld_d; 
  word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] out_alu_trees_results_pre;

  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      foreach ( alu_trees_results_pipe_q[i,j]) begin
        alu_trees_results_pipe_q[i][j] <= '0;
        alu_trees_results_pipe_vld_q[i][j] <= '0;
      end
    end else begin
      if (pipe_en==1) begin
        alu_trees_results_pipe_q <= alu_trees_results_pipe_d;
        alu_trees_results_pipe_vld_q <= alu_trees_results_pipe_vld_d;
      end
    end
  end
  
  // Pipeline vld
  always_comb begin
    alu_trees_results_pipe_vld_d[0] = in_alu_trees_results_vld;
    for (integer i=1; i< TREE_DEPTH-1; i=i+1) begin
      alu_trees_results_pipe_vld_d[i] = alu_trees_results_pipe_vld_q[i-1];
    end
  end
  // Pipeline data, with gating based on vld
  always_comb begin
    alu_trees_results_pipe_d = alu_trees_results_pipe_q;
    for (integer j=0; j < N_TREE ; j= j + 1) begin
      for (integer k=0; k < N_ALU_PER_TREE ; k= k + 1) begin
        if (alu_trees_results_pipe_vld_d[0][j][k]) begin
          alu_trees_results_pipe_d[0][j][k] = in_alu_trees_results[j][k];
        end
      end
    end
    for (integer i=1; i< TREE_DEPTH-1; i=i+1) begin
      for (integer j=0; j < N_TREE ; j= j + 1) begin
        for (integer k=0; k < N_ALU_PER_TREE ; k= k + 1) begin
          if (alu_trees_results_pipe_vld_d[i][j][k]) begin
            alu_trees_results_pipe_d[i][j][k] = alu_trees_results_pipe_q[i-1][j][k];
          end
        end
      end
    end
  end

  // Outputs accoring to pipestages
  always_comb begin
    for (integer tree_i=0; tree_i < N_TREE ; tree_i= tree_i + 1) begin
      // Top ALU's result not to be piped
      out_alu_trees_results_pre[tree_i][0] = in_alu_trees_results[tree_i][0]; 
      // All other ALUs except top, pipelined according to lvl
      for (integer lvl_i=1; lvl_i < TREE_DEPTH ; lvl_i= lvl_i+1) begin
        integer rvr_lvl, start_i;
        rvr_lvl= TREE_DEPTH - lvl_i;
        start_i= (2**rvr_lvl) - 1;
        for (integer alu_i=0; alu_i< 2**(TREE_DEPTH-lvl_i); alu_i=alu_i+1) begin
//          out_alu_trees_results_pre[tree_i][2**(TREE_DEPTH-lvl_i) - 1 + alu_i] = alu_trees_results_pipe_q[TREE_DEPTH - lvl_i -1][tree_i][2**(TREE_DEPTH-lvl_i) - 1 + alu_i];
          out_alu_trees_results_pre[tree_i][start_i + alu_i] = alu_trees_results_pipe_q[rvr_lvl -1][tree_i][start_i + alu_i];

        end
      end
    end
  end
  
  //===========================================
  //       Pipelining of crossbar data
  //===========================================
  word_t_reg [TREE_DEPTH-1 : 0][N_BANKS - 1 : 0] crossbar_output_pipe;
  logic [TREE_DEPTH-1 : 0][N_BANKS - 1 : 0] crossbar_output_pipe_vld;

  // TODO, FIXME: Implement clock gating based on a valid signal. Currently, the pipestages are always on irrespective of data.
  // TODO: Combine alu pipe stages with crossbar pipe stages? Why have two separate paths?
  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      foreach ( crossbar_output_pipe[i,j]) begin
        crossbar_output_pipe[i][j] <= '0;
      end
      crossbar_output_pipe_vld <= '0;
    end else begin
      crossbar_output_pipe_vld[0] <= crossbar_pipe_en;
      for (integer i=1; i< TREE_DEPTH; i=i+1) begin
        crossbar_output_pipe_vld[i] <= crossbar_output_pipe_vld[i-1];
      end

      for (integer j=0; j< N_BANKS; j=j+1) begin
        if (crossbar_pipe_en[j]) begin
          crossbar_output_pipe[0][j] <= in_crossbar_output[j];
        end
        for (integer i=1; i< TREE_DEPTH; i=i+1) begin
          if (crossbar_output_pipe_vld[i-1][j]) begin
            crossbar_output_pipe[i][j] <= crossbar_output_pipe[i-1][j];
          end
        end
      end
    end
  end

  // Assign final outputs
  assign out_alu_trees_results = out_alu_trees_results_pre;
  assign out_crossbar_output   = crossbar_output_pipe[TREE_DEPTH-1];
  assign mem_inputs   = (TREE_DEPTH > 1) ? crossbar_output_pipe[TREE_DEPTH - 1 - DATA_MEM_RD_LATENCY] : in_crossbar_output;

endmodule


`endif //PIPELINED_DATA_DEF
