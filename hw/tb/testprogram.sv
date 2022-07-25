`ifndef TESTPROGRAM_DEF
  `define TESTPROGRAM_DEF

  `include "interface.sv"
  `include "config.sv"
  `include "environment.sv"

class testprogram extends uvm_test;
  `uvm_component_utils(testprogram)
  
  test_config config_obj;
  
  environment env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction : new
  
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);

    config_obj = test_config::type_id::create();
`ifdef GATE_NETLIST
    config_obj.intf_type = test_config_pkg::GATE_NETLIST;
`else
    config_obj.intf_type = test_config_pkg::RTL;
`endif
    uvm_config_db #(test_config)::set(this,"*", "test_config", config_obj);

    env= environment::type_id::create("env", this);
  endfunction : build_phase

  //---------------------------------------
  // end_of_elobaration phase
  //---------------------------------------  
  virtual function void end_of_elaboration();
    //print's the topology
    print();
  endfunction

endclass: testprogram

`endif //TESTPROGRAM_DEF

