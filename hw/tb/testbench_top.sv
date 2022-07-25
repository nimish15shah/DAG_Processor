
`ifndef TESTBENCH_TOP_DEF
  `define TESTBENCH_TOP_DEF

  `include "common_tb.sv"
 // `include "testprogram.sv"
  `include "interface.sv"
  `include "init.sv"


//`delay_mode_zero

module tbench_top;
  //`delay_mode_zero
  //`delay_mode_distributed
  //`vcs_mipdexpand

  //clock and reset signal declaration
  bit clk;

  localparam real CLK_HALF_PERIOD= 1.7ns;

  //clock generation
  always #(CLK_HALF_PERIOD) clk = ~clk;

  intf intf_ins(clk);

  //DUT instance, interface signals are connected to the DUT ports
  pru_sync DUT (
    .clk                 (clk                      ),
    .rst                 (intf_ins.rst                 ),
    .enable_execution (intf_ins.enable_execution),
    .init_instr      (intf_ins.init_instr     ),
    .init_instr_addr (intf_ins.init_instr_addr),
    .init_instr_we   (intf_ins.init_instr_we  ),
`ifdef INSTR_PING_PONG
    .io_ping_wr      (intf_ins.io_ping_wr),
`endif
    .current_instr_rd_addr (intf_ins.current_instr_rd_addr),
    .init_data_in    (intf_ins.init_data_in   ),
    .init_data_out   (intf_ins.init_data_out  ),
    .init_data_addr  (intf_ins.init_data_addr ),
    .init_data_we    (intf_ins.init_data_we   ),
    .init_data_re    (intf_ins.init_data_re)

 );

  Init init_object;

  //---------------------------------------
  //passing the interface handle to lower heirarchy using set method 
  //and enabling the wave dump
  //---------------------------------------
  initial begin 
    int DUMP=0;
    int SDF=0;
    int SAIF= 0;

    init_object= new(intf_ins);

    $disable_warnings; // Disable warnings as soon as simulation started
    $assertoff; // Disable assertions as soon as simulation started
    
    init_object.reset();
    $display("Start initialization");
    init_object.init_data();
`ifndef INSTR_PING_PONG
    init_object.init_program();
`endif
    $display("Done initialization");
    
    if (SAIF) begin
      $set_toggle_region("DUT");
      $toggle_reset();
      $toggle_start();
    end
    if (DUMP) $dumpon;
`ifdef INSTR_PING_PONG
    init_object.execute_ping_pong(2*CLK_HALF_PERIOD);
`else
    init_object.execute(2*CLK_HALF_PERIOD);
`endif
    if (DUMP) $dumpoff;
    if (SAIF) begin
      $toggle_stop();
    end
    /* init_object.check_final_output(); */
    $finish;
  end

endmodule : tbench_top

`endif //TESTBENCH_TOP_DEF
