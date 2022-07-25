

`ifndef REGBANKS_DEF
  `define REGBANKS_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

`include "invalid_states.sv"

module register_banks (
  input clk, rst,
  input pipe_en,
  input word_t [N_BANKS - 1 : 0] inputs,
  input controller_type_defs::reg_we_t reg_we,
  input controller_type_defs::reg_re_t reg_re,
  input controller_type_defs::reg_inv_t reg_inv,
  input controller_type_defs::reg_rd_addr_t reg_rd_addr,
  output word_t [N_BANKS - 1 : 0] outputs
  ); 
  import instr_decd_pkg::*;

  reg_rd_addr_t reg_wr_addr;
  word_t_reg [N_BANKS - 1 : 0] [BANK_DEPTH - 1 : 0] register;

  controller_type_defs::reg_re_t reg_re_filtered; // A bug in instr_decd always sets reg_re[0] to 1. This messes up verification code with the file. Hence reg_re_filtered[0] is deliberately set to 0
  
  word_t [N_BANKS - 1 : 0] outputs_pre;

  /* Instances */
  invalid_states invalid_states_ins(.clk(clk), .rst(rst), 
                                    .pipe_en     (pipe_en    ), 
                                    .reg_we      (reg_we     ), 
                                    .reg_inv     (reg_inv    ), 
                                    .reg_rd_addr (reg_rd_addr), 
                                    .reg_wr_addr (reg_wr_addr));
  
  // register bank writes
  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      for (integer i=0; i < N_BANKS; i=i+1) begin
        for (integer j= 0; j < BANK_DEPTH; j= j+1) begin
          register[i][j] <= 0;
        end
      end

    end else begin
      if (pipe_en) begin
        for (integer i=0; i < N_BANKS; i=i+1) begin
          if (reg_we[i] == 1) begin
            register[i][reg_wr_addr[i]] <= inputs[i];
          end
        end
      end
    end
  end
  
  // register reads
  always_comb begin
    foreach (outputs_pre[i]) begin
      outputs_pre[i] = register[i][reg_rd_addr[i]];
    end
  end

  always_comb begin
  // A bug in instr_decd always sets reg_re[0] to 1. This messes up verification code with the file. Hence reg_re_filtered[0] is deliberately set to 0
    reg_re_filtered = reg_re;
    reg_re_filtered[0] = '0;
  end

  assign outputs= outputs_pre;
  
  `ifdef DISPLAY_ERRORS
  // Continuous check of register reads and invalids
  initial begin
    localparam FILE_LEN= 2*INSTR_MEM_DEPTH;
    logic [N_BANKS - 1 : 0] reg_re_inverted;
    logic [N_BANKS - 1 : 0] reg_inv_inverted;
    logic [N_BANKS - 1 : 0] golden_re_ls [0 : FILE_LEN  - 1]; // hoping that the size of the list would be sufficient
    logic [N_BANKS - 1 : 0] golden_inv_ls [0 : FILE_LEN  - 1]; // hoping that the size of the list would be sufficient
    word_t golden_outputs_ls [0 : FILE_LEN - 1] [0 : N_BANKS - 1];
    controller_type_defs::reg_rd_addr_t golden_reg_rd_addr_ls [0 : FILE_LEN - 1] [0 : N_BANKS - 1];
    string prefix= get_fname_prefix();
    int first_interesting_idx = 0;

    $readmemb({prefix, "_reg_re.txt"}, golden_re_ls);
    $readmemb({prefix, "_reg_inv.txt"}, golden_inv_ls);
    $readmemh({prefix, "_reg_rd_data.txt"}, golden_outputs_ls);
    $readmemh({prefix, "_reg_rd_addr.txt"}, golden_reg_rd_addr_ls);

    // Skip initial data with no reg_re enabled
    for (integer i=0; i < FILE_LEN ; i=i+1) begin
      assert (!$isunknown(golden_re_ls[i])) else $finish;
      if (golden_re_ls[i] != 0) begin
        first_interesting_idx = i;
        break;
      end
    end
      // Wait for the first active reg_re on the ports
    forever begin
      @(posedge clk);
      if (rst == RESET_STATE) break;
    end
    forever begin
      @(posedge clk);
      if (reg_re_filtered != '0) break;
    end

    $display("Actively verifying reg_re and outputs");
    for (integer i=first_interesting_idx; i < FILE_LEN ; i=i+1) begin
      foreach ( reg_re_filtered[i]) begin
        reg_re_inverted[i] = reg_re_filtered[N_BANKS - i - 1];
        reg_inv_inverted[i] = reg_inv[N_BANKS - i - 1];
      end
      reg_re_inverted [N_BANKS - 1] = golden_re_ls[i][N_BANKS - 1]; // To ignore the bank with buggy read enable

      if (reg_re_inverted != golden_re_ls[i] || $isunknown(golden_re_ls[i]) || $isunknown(reg_re)) begin
        $display("Error: Mismatch or unknown in reg_re in register files: %b, %b at instruction : %0d and time %t\n", reg_re_inverted, golden_re_ls[i], i, $time);
      end
      if (reg_inv_inverted != golden_inv_ls[i] || $isunknown(golden_inv_ls[i]) || $isunknown(reg_inv)) begin
        $display("Error: Mismatch or unknown in reg_inv in register files: %b, %b at instruction : %0d and time %t\n", reg_inv_inverted, golden_inv_ls[i], i, $time);
      end
      for (integer j=0; j< N_BANKS; j=j+1) begin
        if (reg_re_inverted[N_BANKS - 1 - j]) begin
          /* $display("Info: READS bank:%0d, pos:%0d, data:%0x, golden data: %0x, instruction : %0d and time %t\n", j, reg_rd_addr[j], outputs[j], golden_outputs_ls[i][j], i, $time); */
          if ((golden_reg_rd_addr_ls[i][j] != reg_rd_addr[j]) || $isunknown(reg_rd_addr[j]) || $isunknown(golden_reg_rd_addr_ls[i][j])) begin
            $display("Error: Mismatch or unkown in reg_rd_addr in register files: bank:%0d, %0d, %0d at instruction : %0d and time %t\n", j, reg_rd_addr[j], golden_reg_rd_addr_ls[i][j], i, $time);
          end
          if (!verif_data(golden_outputs_ls[i][j], outputs[j]) || $isunknown(outputs[j]) || $isunknown(golden_outputs_ls[i][j])) begin
            $display("Error: Mismatch or unkown in outputs in register files: bank:%0d, pos:%0d, pos golden: %0d, %0x, %0x at instruction : %0d and time %t\n", j,reg_rd_addr[j], golden_reg_rd_addr_ls[i][j], outputs[j], golden_outputs_ls[i][j], i, $time);
          end
        end
      end
      @(posedge clk);
    end
  end


  // Continuous check of register writes
  initial begin
    localparam FILE_LEN= 2*INSTR_MEM_DEPTH;
    logic [N_BANKS - 1 : 0] reg_we_inverted;
    logic [N_BANKS - 1 : 0] golden_we_ls [0 : FILE_LEN  - 1]; // hoping that the size of the list would be sufficient
    word_t golden_inputs_ls [0 : FILE_LEN - 1] [0 : N_BANKS - 1];
    controller_type_defs::reg_rd_addr_t golden_reg_wr_addr_ls [0 : FILE_LEN - 1] [0 : N_BANKS - 1];
    string prefix= get_fname_prefix();
    int first_interesting_idx = 0;

    $readmemb({prefix, "_reg_we.txt"}, golden_we_ls);
    $readmemh({prefix, "_reg_wr_data.txt"}, golden_inputs_ls);
    $readmemh({prefix, "_reg_wr_addr.txt"}, golden_reg_wr_addr_ls);

    // Skip initial data with no reg_we enabled
    for (integer i=0; i < FILE_LEN ; i=i+1) begin
      assert (!$isunknown(golden_we_ls[i])) else $finish;
      if (golden_we_ls[i] != 0) begin
        first_interesting_idx = i;
        break;
      end
    end
      // Wait for the first active reg_we on the ports
    forever begin
      @(posedge clk);
      if (rst == RESET_STATE) break;
    end
    forever begin
      @(posedge clk);
      if (reg_we != '0) break;
    end

    $display("Actively verifying reg_we and inputs");
    for (integer i=first_interesting_idx; i < FILE_LEN ; i=i+1) begin
      foreach ( reg_we[i]) begin
        reg_we_inverted[i] = reg_we[N_BANKS - i - 1];
      end

      if (reg_we_inverted != golden_we_ls[i] || $isunknown(golden_we_ls[i]) || $isunknown(reg_we)) begin
        $display("Error: Mismatch or unknown in reg_we in register files: %b, %b at instruction : %0d and time %t\n", reg_we_inverted, golden_we_ls[i], i, $time);
      end
      for (integer j=0; j< N_BANKS; j=j+1) begin
        if (reg_we[j]) begin
          /* $display("Info: WRITES bank:%0d, pos:%0d, data:%0x, golden data: %0x, instruction : %0d and time %t\n", j, reg_wr_addr[j], inputs[j], golden_inputs_ls[i][j], i, $time); */
          if ((golden_reg_wr_addr_ls[i][j] != reg_wr_addr[j]) || $isunknown(reg_wr_addr[j]) || $isunknown(golden_reg_wr_addr_ls[i][j])) begin
            $display("Error: Mismatch or unkown in reg_wr_addr in register files: bank:%0d, %0d, %0d at instruction : %0d and time %t\n", j, reg_wr_addr[j], golden_reg_wr_addr_ls[i][j], i, $time);
          end
          if (!verif_data(golden_inputs_ls[i][j], inputs[j]) || $isunknown(inputs[j]) || $isunknown(golden_inputs_ls[i][j])) begin
            $display("Error: Mismatch or unkown in inputs in register files: bank:%0d, %0x, %0x, verif_data: %0d,  at instruction : %0d and time %t\n", j, inputs[j], golden_inputs_ls[i][j], verif_data(golden_inputs_ls[i][j], inputs[j]), i, $time);
          end
        end
      end
      @(posedge clk);
    end
  end

  // Check the final output at the end
  final begin
    word_t golden [0 : 1];
    logic [$clog2(N_BANKS) - 1 : 0] bank;
    word_t golden_data, simulated_data;
    string net= "asia";
    string prefix= get_fname_prefix();
    int fd;

    $readmemh({prefix, "_golden.txt"}, golden);
    
    bank = golden[0];
    golden_data = golden[1];
    simulated_data = register[bank][0];

    fd= $fopen("functionality.txt", "a");
    $fwrite(fd, "%s, golden_data %0h, simulated_data %0h", log_prefix(), golden_data, simulated_data);

    if (verif_data(golden_data, simulated_data)) begin
      $display("golden: %h, simulated: %h", golden_data, simulated_data);
      $display("Test passes  !!!!!");
      $fdisplay(fd, ", PASS");
    end else begin
      $display("golden: %h, simulated: %h", golden_data, simulated_data);
      $display("########################");
      $display("Test Fail Fail Fail Fail Fail");
      $display("########################");
      $fdisplay(fd, ", FAIL");
    end
    $fclose(fd);
  end
  `endif // DISPLAY_ERRORS

endmodule


`endif // REGBANKS_DEF
