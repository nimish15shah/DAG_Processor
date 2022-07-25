
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
    int SAIF= 1;
    //uvm_config_db#(virtual intf)::set(uvm_root::get(),"*","vif",intf_ins);
    //enable wave dump

    //if (DUMP) begin
    //  $dumpfile("/volume1/users/nshah/synopsys_vcs/dump.vcd"); 
    //  $dumpvars;
    //  $dumpoff;
    //end
    
    if (SDF) begin
      /* $sdf_annotate("/volume1/users/nshah/bayesian_network/hardware/backend//floorplan/netlist_with_data_gating_in_regs.verilog.sdf", DUT, , , "maximum"); */
      /* $sdf_annotate("/esat/betelgeuse1/users/nshah/bayesian_network/hardware/backend//floorplan/netlist_without_data_gating_in_regs.verilog.sdf", DUT, , , "maximum"); */
      /* $sdf_annotate("/volume1/users/nshah/bayesian_network/hardware/backend/placeroute/optRoute.verilog.sdf", DUT, , , "maximum"); */
      //$sdf_annotate("/volume1/users/nshah/bayesian_network/hardware/backend/gds/out/pru_async_top.verilog.sdf");
      /* $sdf_annotate("/esat/centauri1/users/nshah/bayesian_network/hardware/backend/placeroute/pru_async_top_1.verilog.sdf", DUT, , , "maximum"); */
    end

    init_object= new(intf_ins);

    $disable_warnings; // Disable warnings as soon as simulation started
    $assertoff; // Disable assertions as soon as simulation started
    
    init_object.reset();
    /* $enable_warnings; */
    /* $asserton; */
    $display("Start initialization");
    init_object.init_data();
`ifndef INSTR_PING_PONG
    init_object.init_program();
`endif
    $display("Done initialization");
    
    //Instructs VCS MX to start monitoring switching activity.
    //$set_gate_level_monitoring("rtl_on");
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
      /* $toggle_report("/volume1/users/nshah/synopsys_vcs/activity.saif", 1.0e-9,"DUT"); */
      $toggle_report("/esat/puck1/users/nshah/vcs_simulation_data/activity.saif", 1.0e-9,"DUT");
    end
    init_object.check_final_output();
    //$toggle_report("/volume1/users/nshah/synopsys_vcs/activity.saif", 1.0e-9,"pru_async_top");
    $finish;
  end

  //initial $sdf_annotate("./no_backup/activity.sdf", pru_async_top);
 
 //---------------------------------------
 //calling test
 //---------------------------------------
 //initial begin 
 //  run_test("testprogram");
 //end

endmodule : tbench_top

`endif //TESTBENCH_TOP_DEF
