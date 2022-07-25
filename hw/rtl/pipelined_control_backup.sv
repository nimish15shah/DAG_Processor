
`ifndef PIPELINED_CONTROL_DEF
  `define PIPELINED_CONTROL_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

interface pipelined_control_if( input clk, rst, controller_if.datapath_mp io);
  /* Control to Datapath */
  var alu_mode_t      alu_mode;
  var crossbar_sel_t  crossbar_sel; // Always use packed array as casting from packed to unpacked is not straightforward.
  logic               pipe_en;
  var reg_wr_mode_t   reg_wr_mode;
  var reg_wr_sel_t    reg_wr_sel;

  /* Control to RAM */
  var ram_addr_t      ram_addr;
  var ram_we_t        ram_we;
  var ram_re_t        ram_re;

  /* Control to Regbank */ 
  var reg_rd_addr_t reg_rd_addr;
  var reg_we_t      reg_we;
  var reg_inv_t     reg_inv;


  /* redefining inputs (needed for modport) */
  var alu_mode_t      in_alu_mode;
  var crossbar_sel_t  in_crossbar_sel; // Always use packed array as casting from packed to unpacked is not straightforward.
  logic               in_pipe_en;
  var reg_wr_mode_t   in_reg_wr_mode;
  var reg_wr_sel_t    in_reg_wr_sel;
  var ram_addr_t      in_ram_addr;
  var ram_we_t        in_ram_we;
  var ram_re_t        in_ram_re;
  var reg_rd_addr_t in_reg_rd_addr;
  var reg_we_t      in_reg_we;
  var reg_inv_t     in_reg_inv;

  assign in_alu_mode     = io.alu_mode;
  assign in_pipe_en      = io.pipe_en;
  assign in_ram_addr     = io.ram_addr;
  assign in_ram_we       = io.ram_we;
  assign in_ram_re       = io.ram_re;
  assign in_reg_wr_mode  = io.reg_wr_mode;
  assign in_reg_wr_sel   = io.reg_wr_sel;
  assign in_reg_rd_addr  = io.reg_rd_addr;
  assign in_reg_we       = io.reg_we;
  assign in_reg_inv      = io.reg_inv;
  assign in_crossbar_sel = io.crossbar_sel;

  modport datapath_mp ( input alu_mode, crossbar_sel, pipe_en, ram_addr, ram_we, ram_re, reg_wr_mode, reg_wr_sel, reg_rd_addr, reg_we, reg_inv);
  modport pipelined_control_mp ( 
            output   alu_mode,
               crossbar_sel,
               pipe_en,
               ram_addr,
               ram_we,
               ram_re,
               reg_wr_mode,
               reg_wr_sel,
               reg_rd_addr,
               reg_we,
               reg_inv,
 
            input clk, rst, 
            input in_alu_mode,
               in_crossbar_sel,
               in_pipe_en,
               in_ram_addr,
               in_ram_we,
               in_ram_re,
               in_reg_wr_mode,
               in_reg_wr_sel,
               in_reg_rd_addr,
               in_reg_we,
               in_reg_inv
      );
endinterface

module pipelined_control (
  pipelined_control_if.pipelined_control_mp io
  ); 

  import hw_config_pkg::PIPE_STAGES;

  alu_mode_t alu_mode;
  crossbar_sel_t crossbar_sel;
  logic pipe_en;
  ram_addr_t ram_addr;
  ram_we_t ram_we;
  ram_re_t ram_re;
  reg_wr_mode_t reg_wr_mode;
  reg_wr_sel_t reg_wr_sel;
  reg_rd_addr_t reg_rd_addr;
  reg_we_t reg_we;
  reg_inv_t reg_inv;

  var alu_mode_t     [PIPE_STAGES-1 : 0] pipe_alu_mode;     
  var crossbar_sel_t [PIPE_STAGES-1 : 0] pipe_crossbar_sel; 
  logic              [PIPE_STAGES-1 : 0] pipe_pipe_en;
  var reg_wr_mode_t  [PIPE_STAGES-1 : 0] pipe_reg_wr_mode;
  var reg_wr_sel_t   [PIPE_STAGES-1 : 0] pipe_reg_wr_sel;
  var ram_addr_t     [PIPE_STAGES-1 : 0] pipe_ram_addr;
  var ram_we_t       [PIPE_STAGES-1 : 0] pipe_ram_we;
  var ram_re_t       [PIPE_STAGES-1 : 0] pipe_ram_re;
  var reg_rd_addr_t  [PIPE_STAGES-1 : 0] pipe_reg_rd_addr;
  var reg_we_t       [PIPE_STAGES-1 : 0] pipe_reg_we;
  var reg_inv_t      [PIPE_STAGES-1 : 0] pipe_reg_inv;    


  always_ff @(posedge io.clk) begin
    if (io.rst== RESET_STATE) begin
      foreach ( pipe_alu_mode[i,j]) begin
        for (integer k=0; k< N_ALU_PER_TREE; k=k+1) begin
          pipe_alu_mode[i][j][k] <= PASS_0;
        end
      end
      foreach ( pipe_reg_wr_sel[i,j]) begin
        pipe_reg_wr_sel[i][j] <=  '0;
      end
      foreach ( pipe_reg_rd_addr[i,j]) begin
        pipe_reg_rd_addr[i][j] <=  '0;
      end
      foreach ( pipe_crossbar_sel[i,j]) begin
        pipe_crossbar_sel[i][j] <=  '0;
      end
      pipe_pipe_en      <=  '0;
      pipe_reg_wr_mode  <=  ALU;
      pipe_ram_addr     <=  '0;
      pipe_ram_we       <=  '0;
      pipe_ram_re       <=  '0;
      pipe_reg_we       <=  '0;
      pipe_reg_inv      <=  '0;

    end else begin
      if (io.in_pipe_en == 1) begin // Enable the pipe only if in_pipe_en is 1
        pipe_alu_mode    [0] <= io.in_alu_mode     ;
        pipe_crossbar_sel[0] <= io.in_crossbar_sel ;
        pipe_pipe_en     [0] <= io.in_pipe_en      ;
        pipe_reg_wr_mode [0] <= io.in_reg_wr_mode  ;
        pipe_reg_wr_sel  [0] <= io.in_reg_wr_sel   ;
        pipe_ram_addr    [0] <= io.in_ram_addr     ;
        pipe_ram_we      [0] <= io.in_ram_we       ;
        pipe_ram_re      [0] <= io.in_ram_re       ;
        pipe_reg_rd_addr [0] <= io.in_reg_rd_addr  ;
        pipe_reg_we      [0] <= io.in_reg_we       ;
        pipe_reg_inv     [0] <= io.in_reg_inv      ;

        for (integer i=1; i<PIPE_STAGES; i= i+1) begin
          pipe_alu_mode    [i] <= pipe_alu_mode    [i-1];  
          pipe_crossbar_sel[i] <= pipe_crossbar_sel[i-1];
          pipe_pipe_en     [i] <= pipe_pipe_en     [i-1];
          pipe_reg_wr_mode [i] <= pipe_reg_wr_mode [i-1];
          pipe_reg_wr_sel  [i] <= pipe_reg_wr_sel  [i-1];
          pipe_ram_addr    [i] <= pipe_ram_addr    [i-1];
          pipe_ram_we      [i] <= pipe_ram_we      [i-1];
          pipe_ram_re      [i] <= pipe_ram_re      [i-1];
          pipe_reg_rd_addr [i] <= pipe_reg_rd_addr [i-1];
          pipe_reg_we      [i] <= pipe_reg_we      [i-1];
          pipe_reg_inv     [i] <= pipe_reg_inv     [i-1];
        end
      end
    end
  end

  /* Assign outputs */
  // 0th Pipe stage
  assign io.crossbar_sel = io.in_crossbar_sel;
  assign io.pipe_en      = io.in_pipe_en;
  assign io.ram_addr     = io.in_ram_addr;
  assign io.ram_re       = io.in_ram_re;
  assign io.reg_rd_addr  = io.in_reg_rd_addr;
  assign io.reg_inv      = io.in_reg_inv;
  assign io.ram_we       = io.in_ram_we;

  // pipe stages inside trees
  genvar tree_depth_i, tree_i, alu_i;
  generate
    for (tree_i=0; tree_i < N_TREE ; tree_i= tree_i + 1) begin: tree_loop      
      for (tree_depth_i=0; tree_depth_i< TREE_DEPTH ; tree_depth_i=tree_depth_i+1) begin: tree_depth_loop
        for (alu_i= (2**(TREE_DEPTH - tree_depth_i - 1)) - 1; alu_i< (2**(TREE_DEPTH-tree_depth_i)) - 1; alu_i= alu_i + 1) begin: alu_loop
          assign io.alu_mode[tree_i][alu_i] = pipe_alu_mode[tree_depth_i][tree_i][alu_i];
        end
      end
    end
  endgenerate

  // write back stage
  assign io.reg_we      = pipe_reg_we     [TREE_DEPTH]; 
  assign io.reg_wr_sel  = pipe_reg_wr_sel [TREE_DEPTH]; 
  assign io.reg_wr_mode = pipe_reg_wr_mode[TREE_DEPTH]; 

endmodule


`endif //PIPELINED_CONTROL_DEF
