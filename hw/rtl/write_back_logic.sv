//=======================================================================
// Created by         : KU Leuven
// Filename           : /users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/src/write_back_logic.sv
// Author             : Nimish Shah
// Created On         : 2019-10-21 16:52
// Last Modified      : 
// Update Count       : 2019-10-21 16:52
// Description        : 
//                      
//=======================================================================

`ifndef WRITE_BACK_LOGIC_DEF
  `define WRITE_BACK_LOGIC_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

module  write_back_logic(
  input reg_we_t reg_we,
  input reg_wr_sel_t reg_wr_sel,
  input reg_wr_mode_t reg_wr_mode,
  input word_t [N_TREE - 1 : 0] [N_ALU_PER_TREE - 1 : 0] alu_results,
  input word_t [N_BANKS - 1 : 0] memory_out,
  input word_t [N_BANKS - 1 : 0] crossbar_out,
  output word_t [N_BANKS - 1 : 0] reg_inputs
  ); 

  //===========================================
  //       reg_wr_sel
  // 0 : Nothing to write
  // 1 : Write output of 1st lvl ALU
  // 2 : Write output of 2nd lvl ALU, and so on..
  //===========================================
  word_t [N_BANKS - 1 : 0] reg_inputs_pre;

  // Convert complex dimensions of alu_results into simple 2D array
//  word_t [N_BANKS-1: 0] [$clog2(TREE_DEPTH) - 1 : 0] alu_results_rearranged;
  word_t [N_BANKS-1: 0] [TREE_DEPTH - 1 : 0] alu_results_rearranged;
  always_comb begin
    for (integer i=0; i< N_BANKS ; i=i+1) begin
      for (integer j=0; j< TREE_DEPTH ; j=j+1) begin
        alu_results_rearranged[i][j] = alu_results[i/N_IN_PER_TREE][(2**(TREE_DEPTH-j-1)) - 1 + ((i%N_IN_PER_TREE)/(2**(j+1)))];
      end
    end
  end

  always_comb begin
    reg_inputs_pre= '0; 
    foreach (reg_we[i]) begin
      if (reg_we[i] == 1) begin
        case (reg_wr_mode) 
          RAM      : reg_inputs_pre[i] = memory_out[i];
          CROSSBAR : reg_inputs_pre[i] = crossbar_out[i];
          ALU      : begin
            assert (reg_wr_sel[i] != 0) else $finish;
            reg_inputs_pre[i] = alu_results_rearranged[i][reg_wr_sel[i]-1];
            /* if (reg_wr_sel[i]==0) begin // No output */
            /*   reg_inputs_pre[i] = 0; */
            /* end else begin */
            /* end */
          end

          default  : begin
            reg_inputs_pre[i]= 'x;
            $warning(1);
          end
        endcase
      end
    end
  end

  assign reg_inputs = reg_inputs_pre;

endmodule


`endif //WRITE_BACK_LOGIC_DEF
