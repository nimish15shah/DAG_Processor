//-----------------------------------------------------------------------
// Created by         : KU Leuven
// Filename           : utils_pkg.sv
// Author             : Nimish Shah
// Created On         : 2019-11-06 10:16
// Last Modified      : 
// Update Count       : 2019-11-06 10:16
// Description        : 
//                      
//-----------------------------------------------------------------------

`ifndef UTILS_PKG
  `define UTILS_PKG

  `ifdef RUN_FROM_SCRIPT
    `include "common_pkg_SCRIPT.sv"
  `else
    `include "common_pkg.sv"
  `endif
  `include "instr_decd_pkg.sv"

  package utils_pkg;

    import hw_config_pkg::*;
    
    // Example
    function automatic [31:0] multiplier (input [31:0] a, b);
      return a*b;
    endfunction
  
  `ifdef VERIFICATION
    function string get_fname_prefix();
      string net= "asia";
      string prefix="./no_backup/data/";
      string suffix= "";
      int w_conflict= 35;
      /* localparam W_CONFLICT= PIPE_STAGES * 7; */
      $value$plusargs("NET_NAME=%s", net);
      $value$plusargs("PREFIX=%s", prefix);
      $value$plusargs("W_CONFLICT=%d", w_conflict);
      prefix = {prefix, "/"};
      $sformat(suffix, "D%0d_N%0d_%0d_%0d_%0d_%0d_%0d_%0d%0d%0d%0d.00.00.0300False", TREE_DEPTH, N_TREE, N_BANKS, BANK_DEPTH, (2**DATA_MEM_ADDR_L), DATA_MEM_ADDR_L, BIT_L, PIPE_STAGES, TREE_DEPTH, `MIN_DEPTH, w_conflict);

      $display(suffix);

      return {prefix, net, suffix};
    endfunction

    function bit verif_data(input word_t golden, simulated);
      /* if ((golden <= (simulated + 2**10)) && (golden >= (simulated - 2**10))) begin */
      /* localparam SH_FACTOR= 12; */
      /* if ((golden >> SH_FACTOR) == (simulated >> SH_FACTOR)) begin */
      if ((golden <= (simulated + 2**10)) && (simulated <= (golden + 2**10))) begin
        return 1;
      end else begin
        return 0;
      end
    endfunction

    function string log_prefix();
      string net= "asia";
      string prefix;
      $value$plusargs("NET_NAME=%s", net);
      $sformat(prefix,"NET, %s, N_TREE, %0d, TREE_DEPTH, %0d, N_BANKS, %0d, REG_BANK_DEPTH, %0d, PIPE_STAGES, %0d, DATA_MEM_ADDR_L, %0d, BIT_L, %0d", net, N_TREE, TREE_DEPTH, N_BANKS, BANK_DEPTH, PIPE_STAGES, DATA_MEM_ADDR_L, BIT_L);

      return prefix;
    endfunction
  `endif 

  endpackage

  import utils_pkg::*;

  `define print(str) $display(str);

`endif

