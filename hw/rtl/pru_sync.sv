//=======================================================================
// Created by         : KU Leuven
// Filename           : /users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/src/pru_sync.sv
// Author             : Nimish Shah
// Created On         : 2019-10-21 16:53
// Last Modified      : 
// Update Count       : 2019-10-21 16:53
// Description        : 
//                      
//=======================================================================
`ifndef TOP_DEF
  `define TOP_DEF
  
`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

`include "datapath_top.sv"
`include "control_top.sv"
`include "mem_model.sv"

`ifdef INSTR_PING_PONG
module pru_sync #(
  parameter DATA_INIT_FILE_PATH_PREFIX = "dummy",
  parameter INSTR_INIT_FILE_PATH_PREFIX = "dummy"
)
  (
  input clk, rst,

  input enable_execution,

  input [INSTR_L-1 : 0] init_instr,
  input instr_addr_t init_instr_addr,
  input init_instr_we,
  input io_ping_wr,
  output instr_addr_t current_instr_rd_addr,

  input word_t init_data_in,
  output word_t init_data_out,
  input [$clog2(N_BANKS) + DATA_MEM_ADDR_L - 1 : 0] init_data_addr,
  input init_data_we,
  input init_data_re
 
  /* input instr_addr_t  base_instr_addr  , */
  /* output instr_addr_t instr_mem_rd_addr, */
  /* input [INSTR_L-1 : 0] instr, */
  /* input instr_vld, */
  /* output instr_rdy, */

  /* output word_t [N_BANKS - 1 : 0] mem_inputs, */
  /* input word_t [N_BANKS - 1 : 0] mem_outputs, */
  /* output ram_addr_t ram_addr, */
  /* output ram_we_t        ram_we      , */
  /* output ram_re_t        ram_re */      
  ); 

  // Signals
  alu_mode_t      alu_mode    ;
  crossbar_sel_t  crossbar_sel;
  logic           pipe_en     ;
  logic [N_BANKS - 1: 0] crossbar_flop_en;
  logic [N_BANKS - 1: 0] crossbar_pipe_en;
  reg_wr_mode_t   reg_wr_mode ;
  reg_wr_sel_t    reg_wr_sel  ;
  reg_rd_addr_t   reg_rd_addr ;
  reg_we_t        reg_we      ;
  reg_re_t        reg_re      ;
  reg_inv_t       reg_inv     ;

  // signals between logic and mem
  instr_addr_t  base_instr_addr  ;
  instr_addr_t logic_instr_mem_rd_addr;
  instr_addr_t instr_mem_addr_ping; 
  instr_addr_t instr_mem_addr_pong; 
  logic [INSTR_L-1 : 0] instr;
  logic [INSTR_L-1 : 0] instr_ping;
  logic [INSTR_L-1 : 0] instr_pong;
  logic instr_vld;
  logic instr_rdy;
  logic instr_re;
  logic [INSTR_MEM_RD_LATENCY - 1 : 0] instr_re_w_latency;
  
  word_t [N_BANKS - 1 : 0] logic_mem_inputs;
  ram_addr_t logic_ram_addr;
  ram_we_t   logic_ram_we;
  ram_re_t   logic_ram_re;

  word_t [N_BANKS - 1 : 0] mem_inputs;
  word_t [N_BANKS - 1 : 0] mem_outputs;
  ram_addr_t ram_addr;
  ram_we_t        ram_we;
  ram_re_t        ram_re;
  assign base_instr_addr = '0;
  /* assign instr_vld = enable_execution & instr_re_w_latency[INSTR_MEM_RD_LATENCY - 1]; */
  assign instr_vld = enable_execution & instr_re;
  assign instr_re = instr_rdy & enable_execution;
  assign instr_re_ping = ~io_ping_wr ? instr_re : 0;
  assign instr_re_pong = io_ping_wr ? instr_re : 0;
  assign instr_we_ping = io_ping_wr & init_instr_we;
  assign instr_we_pong = (~io_ping_wr) & init_instr_we;
  assign instr_mem_addr_ping = io_ping_wr ? init_instr_addr : logic_instr_mem_rd_addr;
  assign instr_mem_addr_pong = ~io_ping_wr ? init_instr_addr : logic_instr_mem_rd_addr;
  assign current_instr_rd_addr = instr_vld ? logic_instr_mem_rd_addr : '0;
  /* assign current_instr_rd_addr = logic_instr_mem_rd_addr; */
  assign instr = io_ping_wr ? instr_pong : instr_ping;
  
  // signal from/to TB
  logic [$clog2(N_BANKS) - 1 : 0] init_data_bank;
  word_t [N_BANKS - 1 : 0] init_mem_inputs;
  ram_addr_t init_ram_addr;
  ram_we_t   init_ram_we;
  ram_re_t   init_ram_re;
  
  assign init_data_bank = init_data_addr [DATA_MEM_ADDR_L +: $clog2(N_BANKS)];
  assign init_ram_addr = init_data_addr [0 +: DATA_MEM_ADDR_L];
  assign init_data_out = mem_outputs[init_data_bank];

  always_comb begin
    init_ram_we = '0;
    init_ram_re = '0;
    foreach ( init_mem_inputs[i]) begin
      init_mem_inputs[i] = '0;
    end
    init_ram_we[init_data_bank] = init_data_we;
    init_ram_re[init_data_bank] = init_data_re;
    init_mem_inputs[init_data_bank] = init_data_in;
  end

  // switch to init signals whenever asked
  assign mem_inputs = (init_data_we | init_data_re) ? init_mem_inputs : logic_mem_inputs;
  assign ram_addr = (init_data_we | init_data_re) ? init_ram_addr : logic_ram_addr;
  assign ram_we = (init_data_we | init_data_re) ? init_ram_we : logic_ram_we;
  assign ram_re = (init_data_we | init_data_re) ? init_ram_re : logic_ram_re;
  

  // Instances
  control_top control_top_ins(
  .clk              (clk              ),
  .rst              (rst              ),
  .base_instr_addr  (base_instr_addr  ),
  .instr_mem_rd_addr(logic_instr_mem_rd_addr),
  .instr_mem_rd_rdy (instr_mem_rd_rdy ),
  .instr            (instr            ),
  .instr_vld        (instr_vld        ),
  .instr_rdy        (instr_rdy        ),

  .alu_mode    (alu_mode    ) , 
  .crossbar_sel(crossbar_sel) , 
  .pipe_en     (pipe_en     ) , 
  .crossbar_flop_en (crossbar_flop_en),
  .crossbar_pipe_en (crossbar_pipe_en),
  .reg_wr_mode (reg_wr_mode ) , 
  .reg_wr_sel  (reg_wr_sel  ) , 
  .ram_addr    (logic_ram_addr    ) , 
  .ram_we      (logic_ram_we      ) , 
  .ram_re      (logic_ram_re      ) , 
  .reg_rd_addr (reg_rd_addr ) , 
  .reg_we      (reg_we      ) , 
  .reg_re      (reg_re      ) , 
  .reg_inv     (reg_inv     )  
);


  datapath_top datapath_top_ins (
    .clk        (clk          ), 
    .rst        (rst          ), 
    .alu_mode    (alu_mode    ) ,
    .crossbar_sel(crossbar_sel) ,
    .pipe_en     (pipe_en     ) ,
    .crossbar_flop_en (crossbar_flop_en),
    .crossbar_pipe_en (crossbar_pipe_en),
    .reg_wr_mode (reg_wr_mode ) ,
    .reg_wr_sel  (reg_wr_sel  ) ,
    .reg_rd_addr (reg_rd_addr ) ,
    .reg_we      (reg_we      ) ,
    .reg_re      (reg_re      ) ,
    .reg_inv     (reg_inv     ) ,

    .mem_inputs (logic_mem_inputs   ), 
    .mem_outputs(mem_outputs  )
  ); 


  // Data memory
  my_memory #(
    .DATA_L     (BIT_L),
    .ADDR_L     (DATA_MEM_ADDR_L),
    .N_BANKS    (N_BANKS),
    .INIT_FILE_PATH_PREFIX (DATA_INIT_FILE_PATH_PREFIX),
    .RD_LATENCY (DATA_MEM_RD_LATENCY)
  ) data_mem
  (
    .clk     (clk     ),
    .rst     (rst     ),

    .wr_en   (ram_we),
    .addr    (ram_addr),
    .wr_data (mem_inputs),
    .rd_data (mem_outputs),
    .rd_en   (ram_re)
  );

  // Instr memory
  my_memory #(
    .DATA_L     (INSTR_L),
    .ADDR_L     (INSTR_MEM_ADDR_L),
    .N_BANKS    (1),
    .INIT_FILE_PATH_PREFIX (INSTR_INIT_FILE_PATH_PREFIX),
    .RD_LATENCY (INSTR_MEM_RD_LATENCY)
  ) instr_mem_ping 
  (
    .clk     (clk     ),
    .rst     (rst     ),

    .wr_en   (instr_we_ping),
    .addr    (instr_mem_addr_ping),
    .wr_data (init_instr),
    .rd_data (instr_ping),
    .rd_en   (instr_re_ping)
  );

  my_memory #(
    .DATA_L     (INSTR_L),
    .ADDR_L     (INSTR_MEM_ADDR_L),
    .N_BANKS    (1),
    .INIT_FILE_PATH_PREFIX (INSTR_INIT_FILE_PATH_PREFIX),
    .RD_LATENCY (INSTR_MEM_RD_LATENCY)
  ) instr_mem_pong 
  (
    .clk     (clk     ),
    .rst     (rst     ),

    .wr_en   (instr_we_pong),
    .addr    (instr_mem_addr_pong),
    .wr_data (init_instr),
    .rd_data (instr_pong),
    .rd_en   (instr_re_pong)
  );



  /* always_ff @(posedge clk or negedge rst) begin */
  /*   if (rst== RESET_STATE) begin */
  /*     instr_re_w_latency <= 'x; */
  /*   end else begin */
  /*     instr_re_w_latency[0]<= instr_re; */
  /*     for (integer i=1; i< INSTR_MEM_RD_LATENCY; i=i+1) begin */
  /*       instr_re_w_latency[i] <= instr_re_w_latency[i-1]; */
  /*     end */
  /*   end */
  /* end */
  assert property (@(posedge clk) if (instr_we_ping) (!instr_re_ping)) else $warning("Cannot perform both write and read");
  assert property (@(posedge clk) if (instr_we_pong) (!instr_re_pong)) else $warning("Cannot perform both write and read");
  assert property (@(posedge clk) if (instr_re_ping) (!instr_we_ping)) else $warning("Cannot perform both write and read");
  assert property (@(posedge clk) if (instr_re_pong) (!instr_we_pong)) else $warning("Cannot perform both write and read");


endmodule

`else
module pru_sync #(
  parameter DATA_INIT_FILE_PATH_PREFIX = "dummy",
  parameter INSTR_INIT_FILE_PATH_PREFIX = "dummy"
)
  (
  input clk, rst,

  input enable_execution,

  input [INSTR_L-1 : 0] init_instr,
  input instr_addr_t init_instr_addr,
  input init_instr_we,
  output instr_addr_t current_instr_rd_addr,

  input word_t init_data_in,
  output word_t init_data_out,
  input [$clog2(N_BANKS) + DATA_MEM_ADDR_L - 1 : 0] init_data_addr,
  input init_data_we,
  input init_data_re
 
  /* input instr_addr_t  base_instr_addr  , */
  /* output instr_addr_t instr_mem_rd_addr, */
  /* input [INSTR_L-1 : 0] instr, */
  /* input instr_vld, */
  /* output instr_rdy, */

  /* output word_t [N_BANKS - 1 : 0] mem_inputs, */
  /* input word_t [N_BANKS - 1 : 0] mem_outputs, */
  /* output ram_addr_t ram_addr, */
  /* output ram_we_t        ram_we      , */
  /* output ram_re_t        ram_re */      
  ); 

  // Signals
  alu_mode_t      alu_mode    ;
  crossbar_sel_t  crossbar_sel;
  logic           pipe_en     ;
  logic [N_BANKS - 1: 0] crossbar_flop_en;
  logic [N_BANKS - 1: 0] crossbar_pipe_en;
  reg_wr_mode_t   reg_wr_mode ;
  reg_wr_sel_t    reg_wr_sel  ;
  reg_rd_addr_t   reg_rd_addr ;
  reg_we_t        reg_we      ;
  reg_re_t        reg_re      ;
  reg_inv_t       reg_inv     ;

  // signals between logic and mem
  instr_addr_t  base_instr_addr  ;
  instr_addr_t logic_instr_mem_rd_addr;
  instr_addr_t init_instr_mem_rd_addr;
  instr_addr_t instr_mem_addr;
  logic [INSTR_L-1 : 0] instr;
  logic instr_vld;
  logic instr_rdy;
  logic instr_re;
  logic [INSTR_MEM_RD_LATENCY - 1 : 0] instr_re_w_latency;

  word_t [N_BANKS - 1 : 0] logic_mem_inputs;
  ram_addr_t logic_ram_addr;
  ram_we_t   logic_ram_we;
  ram_re_t   logic_ram_re;

  word_t [N_BANKS - 1 : 0] mem_inputs;
  word_t [N_BANKS - 1 : 0] mem_outputs;
  ram_addr_t ram_addr;
  ram_we_t        ram_we;
  ram_re_t        ram_re;
  assign base_instr_addr = '0;
  /* assign instr_vld = enable_execution & instr_re_w_latency[INSTR_MEM_RD_LATENCY - 1]; */
  assign instr_vld = enable_execution & instr_re;
  assign instr_re = instr_rdy & enable_execution;
  
  // signal from/to TB
  logic [$clog2(N_BANKS) - 1 : 0] init_data_bank;
  word_t [N_BANKS - 1 : 0] init_mem_inputs;
  ram_addr_t init_ram_addr;
  ram_we_t   init_ram_we;
  ram_re_t   init_ram_re;
  
  assign init_data_bank = init_data_addr [DATA_MEM_ADDR_L +: $clog2(N_BANKS)];
  assign init_ram_addr = init_data_addr [0 +: DATA_MEM_ADDR_L];
  assign init_data_out = mem_outputs[init_data_bank];

  always_comb begin
    init_ram_we = '0;
    init_ram_re = '0;
    foreach ( init_mem_inputs[i]) begin
      init_mem_inputs[i] = '0;
    end
    init_ram_we[init_data_bank] = init_data_we;
    init_ram_re[init_data_bank] = init_data_re;
    init_mem_inputs[init_data_bank] = init_data_in;
  end

  // switch to init signals whenever asked
  assign mem_inputs = (init_data_we | init_data_re) ? init_mem_inputs : logic_mem_inputs;
  assign ram_addr = (init_data_we | init_data_re) ? init_ram_addr : logic_ram_addr;
  assign ram_we = (init_data_we | init_data_re) ? init_ram_we : logic_ram_we;
  assign ram_re = (init_data_we | init_data_re) ? init_ram_re : logic_ram_re;
  
  assign instr_mem_addr = init_instr_we ? init_instr_addr : logic_instr_mem_rd_addr;
  assign current_instr_rd_addr = instr_vld ? logic_instr_mem_rd_addr : '0;

  // Instances
  control_top control_top_ins(
  .clk              (clk              ),
  .rst              (rst              ),
  .base_instr_addr  (base_instr_addr  ),
  .instr_mem_rd_addr(logic_instr_mem_rd_addr),
  .instr_mem_rd_rdy (instr_mem_rd_rdy ),
  .instr            (instr            ),
  .instr_vld        (instr_vld        ),
  .instr_rdy        (instr_rdy        ),

  .alu_mode    (alu_mode    ) , 
  .crossbar_sel(crossbar_sel) , 
  .pipe_en     (pipe_en     ) , 
  .crossbar_flop_en (crossbar_flop_en),
  .crossbar_pipe_en (crossbar_pipe_en),
  .reg_wr_mode (reg_wr_mode ) , 
  .reg_wr_sel  (reg_wr_sel  ) , 
  .ram_addr    (logic_ram_addr    ) , 
  .ram_we      (logic_ram_we      ) , 
  .ram_re      (logic_ram_re      ) , 
  .reg_rd_addr (reg_rd_addr ) , 
  .reg_we      (reg_we      ) , 
  .reg_re      (reg_re      ) , 
  .reg_inv     (reg_inv     )  
);


  datapath_top datapath_top_ins (
    .clk        (clk          ), 
    .rst        (rst          ), 
    .alu_mode    (alu_mode    ) ,
    .crossbar_sel(crossbar_sel) ,
    .pipe_en     (pipe_en     ) ,
    .crossbar_flop_en (crossbar_flop_en),
    .crossbar_pipe_en (crossbar_pipe_en),
    .reg_wr_mode (reg_wr_mode ) ,
    .reg_wr_sel  (reg_wr_sel  ) ,
    .reg_rd_addr (reg_rd_addr ) ,
    .reg_we      (reg_we      ) ,
    .reg_re      (reg_re      ) ,
    .reg_inv     (reg_inv     ) ,

    .mem_inputs (logic_mem_inputs   ), 
    .mem_outputs(mem_outputs  )
  ); 


  // Data memory
  my_memory #(
    .DATA_L     (BIT_L),
    .ADDR_L     (DATA_MEM_ADDR_L),
    .N_BANKS    (N_BANKS),
    .INIT_FILE_PATH_PREFIX (DATA_INIT_FILE_PATH_PREFIX),
    .RD_LATENCY (DATA_MEM_RD_LATENCY)
  ) data_mem
  (
    .clk     (clk     ),
    .rst     (rst     ),

    .wr_en   (ram_we),
    .addr    (ram_addr),
    .wr_data (mem_inputs),
    .rd_data (mem_outputs),
    .rd_en   (ram_re)
  );

  // Instr memory
  my_memory #(
    .DATA_L     (INSTR_L),
    .ADDR_L     (INSTR_MEM_ADDR_L),
    .N_BANKS    (1),
    .INIT_FILE_PATH_PREFIX (INSTR_INIT_FILE_PATH_PREFIX),
    .RD_LATENCY (INSTR_MEM_RD_LATENCY)
  ) instr_mem 
  (
    .clk     (clk     ),
    .rst     (rst     ),

    .wr_en   (init_instr_we),
    .addr    (instr_mem_addr),
    .wr_data (init_instr),
    .rd_data (instr),
    .rd_en   (instr_re)
  );


  /* always_ff @(posedge clk or negedge rst) begin */
  /*   if (rst== RESET_STATE) begin */
  /*     instr_re_w_latency <= 'x; */
  /*   end else begin */
  /*     instr_re_w_latency[0]<= instr_re; */
  /*     for (integer i=1; i< INSTR_MEM_RD_LATENCY; i=i+1) begin */
  /*       instr_re_w_latency[i] <= instr_re_w_latency[i-1]; */
  /*     end */
  /*   end */
  /* end */


endmodule
`endif

`endif //TOP_DEF
