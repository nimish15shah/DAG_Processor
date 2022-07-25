
`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"
`include "module_library.sv"

`ifndef ALU_DEF
  `define ALU_DEF

module alu (
  input clk, rst, en,
  input word_t in_0,
  input word_t in_1,
  input alu_mode_enum_t mode,
  output out_vld,
  output word_t out
  ); 

  word_t out_d;
  word_t_reg out_q;
  logic out_vld_q;
  
  logic [EXP_L + MNT_L - 1 : 0] in1_mul, in2_mul, in1_add, in2_add;
  logic [EXP_L + MNT_L - 1 : 0] out_mul, out_add;
  logic alu_en;
  logic reg_en;
  
  assign reg_en = (mode == PASS_1) ? 0 : en;

  flt_mul #(
    .EXP_L (EXP_L),
    .MNT_L (MNT_L)
  ) MUL (
    .in1 (in1_mul),
    .in2 (in2_mul),
    .out (out_mul)
  );

  flt_add #(
    .EXP_W (EXP_L),
    .MAN_W (MNT_L)
  ) ADD (
    .in1 (in1_add),
    .in2 (in2_add),
    .out (out_add)
  );
  
  
  always_comb begin
    in1_mul = '0;
    in2_mul = '0;
    in1_add = '0;
    in2_add = '0;
    out_d = 'x;
    alu_en= 1'b0;

    case (mode) 
      /* SUM       : out_d= in_0 + in_1; */
      SUM       : begin 
        in1_add = in_0;
        in2_add = in_1;
        out_d= {1'b0, out_add};
        alu_en= 1'b1;
      end
//      PROD      : out_d= in_0 * in_1;
      PROD      : begin
        in1_mul = in_0;
        in2_mul = in_1;
        out_d= {1'b0, out_mul};
        alu_en= 1'b1;
      end
      PASS_0    : out_d= in_0;
      PASS_1    : out_d= 'x;
      default : begin 
        out_d = 'x;
        $warning(1);
      end
    endcase
  end

  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
        out_q <= 0;
        out_vld_q <= 0;
    end else begin
      if (reg_en==1) begin
        out_q <= out_d;
      end
      out_vld_q <= alu_en;
    end
  end

  assign out = out_q;
  assign out_vld = out_vld_q;

endmodule
`endif // ALU_DEF

`ifndef ALU_TREE_DEF
  `define ALU_TREE_DEF
module alu_tree (
  input clk,rst, en,
  input word_t [N_IN_PER_TREE - 1 : 0] inputs,
  input alu_mode_enum_t [N_ALU_PER_TREE - 1 : 0] alu_mode_per_tree, // 0th mode is for top node, 1st mode is for 1st ALU on the next lvl, etc..
  output [N_ALU_PER_TREE -1 : 0] outputs_vld,
  output word_t [N_ALU_PER_TREE -1 : 0] outputs
  ); 

  //===========================================
  //       alu_mode_per_tree
  //     
  //    | 0 | 1 | 2 | 3 |    
  //     top  /\
  //         0th ALU on the second highest level
  //
  //      outputs
  //
  //    |   0   |   1   |   2   |   3   |    
  //    top out    /\
  //            out of 0th ALU on second highest level
  //===========================================

  word_t [TREE_DEPTH : 0] [N_IN_PER_TREE - 1 :0 ] interm_words; // A 2D array, as handling a 1D vect was too complex
  logic [TREE_DEPTH : 0] [N_IN_PER_TREE - 1 :0 ] interm_outputs_vld; // A 2D array, as handling a 1D vect was too complex
  logic [N_ALU_PER_TREE -1 : 0] outputs_vld_pre;
  word_t [N_ALU_PER_TREE -1 : 0] outputs_pre;

  // initialize with inputs
  assign interm_words[0] = inputs;
  
  generate
    genvar lvl, alu_i;
    for (lvl= 1; lvl<= TREE_DEPTH; lvl= lvl+1) begin: lvl_loop
      for (alu_i=0; alu_i < 2**(TREE_DEPTH-lvl); alu_i= alu_i + 1 ) begin: alu_loop
        alu alu_ins(.clk(clk), .rst(rst), .en(en), 
                    .in_0(interm_words[lvl-1][2*alu_i]), 
                    .in_1(interm_words[lvl-1][2*alu_i + 1]),
                    .mode(alu_mode_per_tree[(2**(TREE_DEPTH-lvl))-1 + alu_i]),
                    .out_vld(interm_outputs_vld[lvl][alu_i]),
                    .out(interm_words[lvl][alu_i]));
      end
    end
  endgenerate

  // Convert from 2D interm_words to 1D outputs_pre
  always_comb begin
    integer rvr_lvl, start_i;
    for (integer lvl_i=1; lvl_i<= TREE_DEPTH; lvl_i= lvl_i+1) begin
      rvr_lvl= TREE_DEPTH - lvl_i;
      start_i= 2**rvr_lvl - 1;
      for (integer alu_i=0; alu_i< 2**rvr_lvl; alu_i=alu_i+1) begin
        outputs_pre[start_i + alu_i] = interm_words[lvl_i][alu_i];
        outputs_vld_pre[start_i + alu_i] = interm_outputs_vld[lvl_i][alu_i];
      end
    end
  end

  assign outputs = outputs_pre;
  assign outputs_vld = outputs_vld_pre;

endmodule
`endif // ALU_TREE_DEF

`ifndef PROCESS_BLOCK_DEF
  `define PROCESS_BLOCK_DEF

module processing_block (
  input clk, rst, en,
  input word_t [N_BANKS -1 : 0] inputs,
  input alu_mode_t alu_mode,
  output [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] outputs_vld,
  output word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] outputs
  ); 

  genvar tree_i;
  generate
    for (tree_i=0; tree_i < N_TREE ; tree_i= tree_i + 1) begin: tree_loop
      alu_tree alu_tree_ins(.clk(clk), .rst(rst), .en(en), 
                           .inputs(inputs[(tree_i + 1)*N_IN_PER_TREE - 1 -: N_IN_PER_TREE]), 
                           .alu_mode_per_tree(alu_mode[tree_i]),
                           .outputs_vld(outputs_vld[tree_i]),
                           .outputs(outputs[tree_i]));
    end 
  endgenerate
endmodule
`endif // PROCESS_BLOCK_DEF
