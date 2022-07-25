`ifndef INIT_DEF
  `define INIT_DEF


  `include "utils_pkg.sv"
  `include "interface.sv"

class Init;
  
  virtual intf ifc;
  string net= "asia";
  localparam INIT_DEBUG= 0;
  localparam bit_len =32;
  string prefix="./no_backup/data/";
  string suffix= "";
  int target_last_instr_addr= 0;

  function new(virtual intf i);
    this.ifc= i;

    this.prefix= get_fname_prefix();

    $value$plusargs("NET_NAME=%s", net);
    $display("INSTR_L : %0d, INSTR_L_MACRO_ADJUSTED: %0d, INSTR_MEM_DEPTH: %d\n", INSTR_L, INSTR_L_MACRO_ADJUSTED, INSTR_MEM_DEPTH);
  endfunction : new

  task reset();
    ifc.cb.enable_execution <= '0;
    ifc.cb.init_instr      <= '0;
    ifc.cb.init_instr_addr <= '0;
    ifc.cb.init_instr_we   <= '0;
                          
    ifc.cb.init_data_in    <= '0;
    ifc.cb.init_data_addr  <= '0;
    ifc.cb.init_data_we    <= '0;
    ifc.cb.init_data_re    <= '0;

    $display(" ----- Reset Started -----");
    this.ifc.cb.rst <= 0; // Synchronous reset
    repeat (2) @(this.ifc.cb);

    this.ifc.cb.rst <= 1; //release Synchronous reset
    repeat (2) @(this.ifc.cb);
    $display(" ----- Reset Ended   -----"); 
  
  endtask

  task init_data();
    localparam N_INIT_DATA= N_BANKS * (2**DATA_MEM_ADDR_L);
    logic [$clog2(N_BANKS) + DATA_MEM_ADDR_L - 1 : 0] init_addr;
    logic [BIT_L - 1 : 0] init_data;
    logic [BIT_L - 1 : 0] out;
    logic  [BIT_L - 1 : 0] file_lines[ 0 : N_INIT_DATA - 1 ] [0 : 1];
    
    $readmemh({prefix, "_data.txt"}, file_lines);

    // write data
    for (integer i=0; i< N_INIT_DATA; i=i+1) begin
      init_addr= file_lines[i][0];
      init_data= file_lines[i][1];
      if ($isunknown(init_addr)) break;
      ifc.cb.init_data_addr <= init_addr;
      ifc.cb.init_data_in <= init_data;
      ifc.cb.init_data_we <= 1;
      repeat(1) @(ifc.cb);
      ifc.cb.init_data_we <= 0;
      ifc.cb.init_data_addr <= 'x;
      ifc.cb.init_data_in <= 'x;
    end
    
    
    if (INIT_DEBUG) begin
      // Verify by reading data
      for (integer i=0; i< N_INIT_DATA; i=i+1) begin
        init_addr= file_lines[i][0];
        init_data= file_lines[i][1];
        if ($isunknown(init_addr)) break;
        ifc.cb.init_data_addr <= init_addr;
        ifc.cb.init_data_re <= 1;
        repeat(1) @(ifc.cb);
        repeat(1) @(ifc.cb);

        if(ifc.cb.init_data_out != init_data || $isunknown(ifc.cb.init_data_out)) begin
          $display("Fail! %h %h %h\n", ifc.cb.init_data_out, init_data, init_addr);
          /* $fatal(1); */
          /* $finish; */
        end else begin
          assert (!$isunknown(ifc.cb.init_data_out));
          if (INIT_DEBUG) $display("Pass! %h %h %h\n", ifc.cb.init_data_out, init_data, init_addr);
        end
        
        ifc.cb.init_data_re <= 0;
        ifc.cb.init_data_addr <= 'x;
      end
    end
  endtask : init_data
  
  task init_program();
    logic [INSTR_MEM_ADDR_L -1 : 0] init_addr;
    logic [INSTR_L - 1 : 0] init_instr;
    logic  [INSTR_L - 1 : 0] file_lines_instr[ 0 : INSTR_MEM_DEPTH - 1 ];
    
    $readmemb({prefix, "_instr.txt"}, file_lines_instr);
    for (integer i=0; i< INSTR_MEM_DEPTH; i=i+1) begin
      init_instr = file_lines_instr[i];
      this.target_last_instr_addr = i;
      if ($isunknown(init_instr)) break;
      ifc.cb.init_instr_addr <= i;
      ifc.cb.init_instr <= init_instr;
      ifc.cb.init_instr_we <= 1;
      repeat(1) @(ifc.cb);
      ifc.cb.init_instr_we <= 0;
      ifc.cb.init_instr_addr <= 'x;
      ifc.cb.init_instr <= 'x;
    end

    $display("target_last_instr_addr : %0d\n", this.target_last_instr_addr);

    if (target_last_instr_addr > INSTR_MEM_DEPTH - 3) begin
      $fatal("INSTR_MEM_SIZE not enough: %0d, %0d\n", target_last_instr_addr, INSTR_MEM_DEPTH);
    end

  endtask : init_program

  task execute_ping_pong (input real CLK_PERIOD);
    time start_time;
    time end_time;
    time run_time;
    int fd;
    logic [INSTR_MEM_ADDR_L -1 : 0] init_addr;
    logic [INSTR_L - 1 : 0] init_instr;
    logic  [INSTR_L - 1 : 0] file_lines_instr[ 0 : /* INSTR_MEM_DEPTH */ 250000- 1 ];
    logic io_ping_wr= 1;

    repeat(1) @(this.ifc.cb);
    ifc.cb.enable_execution <= 0;
    ifc.cb.io_ping_wr <= io_ping_wr;
    repeat(1) @(this.ifc.cb);

    $readmemb({prefix, "_instr.txt"}, file_lines_instr);
    // Initialize ping buffer
    for (integer i=0; i< INSTR_MEM_DEPTH; i=i+1) begin
      init_instr = file_lines_instr[i];
      this.target_last_instr_addr = i;
      if ($isunknown(init_instr)) begin
        $display("Unknown instruction excountered at %d line\n", i);
        break;
      end
      ifc.cb.init_instr_addr <= i;
      ifc.cb.init_instr <= init_instr;
      ifc.cb.init_instr_we <= 1;
      repeat(1) @(ifc.cb);
      ifc.cb.init_instr_we <= 0;
    end
    $display("Initialized ping buffer, target_last_instr_addr = %0d\n", target_last_instr_addr);

    fork
      begin
        forever begin
          repeat(1000) @(this.ifc.cb);
          $display("cycles: %.2f and current_instr_rd_addr: %d", (($time - start_time)/CLK_PERIOD), ifc.cb.current_instr_rd_addr);
        end
      end
      begin
        repeat(250000) @(this.ifc.cb); // avoid infinite loops
      end
      begin
        // Enable excution
        io_ping_wr = 0;
        repeat(1) @(this.ifc.cb);
        ifc.cb.io_ping_wr <= io_ping_wr;
        repeat(1) @(this.ifc.cb);
        ifc.cb.enable_execution <= 1;
        start_time= $time;
        repeat(10) @(this.ifc.cb);

        for (integer i= target_last_instr_addr + 1; i< 300000; i=i+1) begin
        /* for (integer i= INSTR_MEM_DEPTH; i< 300000; i=i+1) begin */
          init_instr = file_lines_instr[i];
          this.target_last_instr_addr = i;
          if ($isunknown(init_instr)) begin
            $display("Unknown instruction excountered at %d line\n", i);
            break;
          end
          if (i % 1000 == 0) $display("At %0d\n", i);
          ifc.cb.init_instr_addr <= (i % INSTR_MEM_DEPTH);
          ifc.cb.init_instr <= init_instr;
          ifc.cb.init_instr_we <= 1;
          repeat(1) @(ifc.cb);
          ifc.cb.init_instr_we <= 0;
          /* ifc.cb.init_instr_addr <= 'x; */
          /* ifc.cb.init_instr <= 'x; */
          if (i % (INSTR_MEM_DEPTH) == (INSTR_MEM_DEPTH - 1)) begin // Switch ping-pong
            // Wait for the processor to consume the other buffer
            $display("Waiting to switch buffer at : at i= %d and %0d time\n",i, $time);
            /* @(ifc.cb iff (ifc.cb.current_instr_rd_addr % (INSTR_MEM_DEPTH) == 0)); */
            @(ifc.cb iff (ifc.cb.current_instr_rd_addr % (INSTR_MEM_DEPTH) == (INSTR_MEM_DEPTH - 1)));
            ifc.cb.enable_execution <= 0;
            repeat(INSTR_MEM_RD_LATENCY + 10) @(this.ifc.cb); // IMP!! Wait for 1 cycle INSTR_MEM_RD_LATENCY

            $display("Switching buffer at : at i= %d and current_instr_rd_addr=%d at %0d time\n",i, ifc.cb.current_instr_rd_addr, $time);
            ifc.cb.io_ping_wr <= ~io_ping_wr;
            /* ifc.cb.io_ping_wr <= 1; */
            repeat(10) @(this.ifc.cb);
            io_ping_wr = ~io_ping_wr;
            ifc.cb.enable_execution <= 1;
          end
        end

        $display("Waiting for the processor to consume the buffer");
        $display("target_last_instr_addr : %0d\n", this.target_last_instr_addr);
        // Let the processor consume the last buffer
        $display("Waiting to switch buffer at : %.2f cycle and at %0d time\n", (($time - start_time)/CLK_PERIOD), $time);
        @(ifc.cb iff (ifc.cb.current_instr_rd_addr % (INSTR_MEM_DEPTH) == (INSTR_MEM_DEPTH - 1)));
        ifc.cb.enable_execution <= 0;
        repeat(INSTR_MEM_RD_LATENCY + 10) @(this.ifc.cb); // IMP!! Wait for 1 cycle INSTR_MEM_RD_LATENCY
        $display("Switching buffer at : %.2f cycle and current_instr_rd_addr at %0d time\n", (($time - start_time)/CLK_PERIOD), ifc.cb.current_instr_rd_addr, $time);
        ifc.cb.io_ping_wr <= ~io_ping_wr;
        repeat(5) @(this.ifc.cb);
        io_ping_wr = ~io_ping_wr;
        ifc.cb.enable_execution <= 1;
        $display("Waiting to reach target_last_instr_addr: %0d, after mod:%0d, at  %.2f cycle and at %0d time\n", target_last_instr_addr, (target_last_instr_addr % INSTR_MEM_DEPTH), (($time - start_time)/CLK_PERIOD), $time);
        @(ifc.cb iff (ifc.cb.current_instr_rd_addr > (target_last_instr_addr % INSTR_MEM_DEPTH)));

        repeat(PIPE_STAGES + 5) @(this.ifc.cb); //flush
        ifc.cb.enable_execution <= 0;
      end
    join_any

    end_time= $time;
    run_time = end_time - start_time;

    $display("########################");
    $display("Done execution!");
    $display("########################");
    $display("\n");
    $display("Time statistics: ");

    $display("start time, end time, run_time: %t, %t, %t\n", start_time, end_time, run_time);
    $display("clk_cycles,%.2f\n", (run_time/CLK_PERIOD));

    `ifdef GATE_NETLIST
      fd= $fopen("latency_netlist_2.txt", "a");
    `else
      fd= $fopen("../../out/reports/rtl_latency.txt", "a+");
    `endif
    $fdisplay(fd, "NET, %s, N_TREE, %0d, TREE_DEPTH, %0d, N_BANKS, %0d, REG_BANK_DEPTH, %0d, PIPE_STAGES, %0d, DATA_MEM_ADDR_L, %0d, INSTR_L, %0d, BIT_L, %0d,latency (clock cycles), %.2f, clock period (ns), %.2f", this.net, N_TREE, TREE_DEPTH, N_BANKS, BANK_DEPTH, PIPE_STAGES, DATA_MEM_ADDR_L, INSTR_L, BIT_L, run_time/CLK_PERIOD, CLK_PERIOD);
    $fclose(fd);

  endtask : execute_ping_pong

  task execute(input real CLK_PERIOD);
    time start_time;
    time end_time;
    time run_time;
    int fd;
     
    int rand_delay;
    std::randomize(rand_delay) with {
      rand_delay < 100;
      rand_delay > 2;
    };

    $display("Starting execution!\n");
    start_time= $time;
    repeat(1) @(this.ifc.cb);
    repeat(1) @(this.ifc.cb);
    ifc.cb.enable_execution <= 1;

    fork
      begin
        forever begin
          repeat(100) @(this.ifc.cb);
          $display("cycles: %.2f", (($time - start_time)/CLK_PERIOD));
        end
      end
      begin
        @(ifc.cb iff (ifc.cb.current_instr_rd_addr > target_last_instr_addr));
      end
      begin
        repeat(350000) @(this.ifc.cb); // avoid infinite loops
      end
    join_any

    repeat(PIPE_STAGES + 5) @(this.ifc.cb); //flush
    ifc.cb.enable_execution <= 0;

    end_time= $time;
    run_time = end_time - start_time;

    $display("########################");
    $display("Done execution!");
    $display("########################");
    $display("\n");
    $display("Time statistics: ");

    $display("start time, end time, run_time: %t, %t, %t\n", start_time, end_time, run_time);
    $display("clk_cycles,%0d\n", (run_time/CLK_PERIOD));

    `ifdef GATE_NETLIST
      fd= $fopen("latency_netlist_2.txt", "a");
    `else
      /* fd= $fopen("latency.txt", "a"); */
      fd= $fopen("../../out/reports/rtl_latency.txt", "a+");
    `endif
    $fdisplay(fd, "NET, %s, N_TREE, %0d, TREE_DEPTH, %0d, N_BANKS, %0d, REG_BANK_DEPTH, %0d, PIPE_STAGES, %0d, DATA_MEM_ADDR_L, %0d, INSTR_L, %0d, BIT_L, %0d,latency (clock cycles), %0d, clock period (ns), %.2f", this.net, N_TREE, TREE_DEPTH, N_BANKS, BANK_DEPTH, PIPE_STAGES, DATA_MEM_ADDR_L, INSTR_L, BIT_L, run_time/CLK_PERIOD, CLK_PERIOD);
    $fclose(fd);
  endtask : execute


  task check_final_output();
    logic [$clog2(N_BANKS) + DATA_MEM_ADDR_L - 1 : 0] addr;
    logic [$clog2(N_BANKS) - 1 : 0] bank;
    word_t golden_data, simulated_data;
    word_t golden [0 : 1];

    string prefix= get_fname_prefix();
    int fd;

    $readmemh({prefix, "_golden.txt"}, golden);
    
    bank = golden[0];
    golden_data = golden[1];
    assert (!$isunknown(golden_data));
    assert (!$isunknown(bank));

    addr = (bank << DATA_MEM_ADDR_L) + 0;  // 0 is the mem addr
    ifc.cb.init_data_addr <= addr;
    ifc.cb.init_data_re <= 1;
    repeat(1) @(ifc.cb);
    repeat(1) @(ifc.cb);
    simulated_data = ifc.cb.init_data_out;
    assert (!$isunknown(simulated_data));

    ifc.cb.init_data_re <= 0;
    ifc.cb.init_data_addr <= 'x;

    `ifdef GATE_NETLIST
      fd= $fopen("functionality_netlist_2.txt", "a");
    `else
      fd= $fopen("functionality_tb.txt", "a");
    `endif

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
  endtask : check_final_output

endclass : Init


`endif //INIT_DEF
