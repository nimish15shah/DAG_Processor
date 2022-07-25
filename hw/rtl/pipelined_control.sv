
`ifndef PIPELINED_CONTROL_DEF
  `define PIPELINED_CONTROL_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

module pipelined_control (
  input clk,rst,
  input                 in_pipe_en        , 
  input [N_BANKS - 1: 0]in_crossbar_pipe_en, 

  input alu_mode_t      in_alu_mode       , 
  input crossbar_sel_t  in_crossbar_sel   , 
  input reg_wr_mode_t   in_reg_wr_mode    , 
  input reg_wr_sel_t    in_reg_wr_sel     , 
  input ram_addr_t      in_ram_addr       , 
  input ram_we_t        in_ram_we         , 
  input ram_re_t        in_ram_re         , 
  input reg_rd_addr_t   in_reg_rd_addr    , 
  input reg_we_t        in_reg_we         , 
  input reg_re_t        in_reg_re         , 
  input reg_inv_t       in_reg_inv        , 

  output alu_mode_t      out_alu_mode     , 
  output crossbar_sel_t  out_crossbar_sel , 
  output                 out_pipe_en      , 
  output [N_BANKS - 1: 0]out_crossbar_pipe_en, 
  output reg_wr_mode_t   out_reg_wr_mode  , 
  output reg_wr_sel_t    out_reg_wr_sel   , 
  output ram_addr_t      out_ram_addr     , 
  output ram_we_t        out_ram_we       , 
  output ram_re_t        out_ram_re       , 
  output reg_rd_addr_t   out_reg_rd_addr  , 
  output reg_we_t        out_reg_we       , 
  output reg_re_t        out_reg_re       , 
  output reg_inv_t       out_reg_inv

  ); 

  import hw_config_pkg::PIPE_STAGES;

  logic [PIPE_STAGES-1 : 0] [N_BANKS - 1 : 0] pipe_crossbar_pipe_en;     
  var alu_mode_t     [PIPE_STAGES-1 : 0] pipe_alu_mode;     
  var crossbar_sel_t [PIPE_STAGES-1 : 0] pipe_crossbar_sel; 
  var reg_wr_mode_t  [PIPE_STAGES-1 : 0] pipe_reg_wr_mode;
  var reg_wr_sel_t   [PIPE_STAGES-1 : 0] pipe_reg_wr_sel;
  var ram_addr_t     [PIPE_STAGES-1 : 0] pipe_ram_addr;
  var ram_we_t       [PIPE_STAGES-1 : 0] pipe_ram_we;
  var ram_re_t       [PIPE_STAGES-1 : 0] pipe_ram_re;
  var reg_rd_addr_t  [PIPE_STAGES-1 : 0] pipe_reg_rd_addr;
  var reg_we_t       [PIPE_STAGES-1 : 0] pipe_reg_we;
  var reg_re_t       [PIPE_STAGES-1 : 0] pipe_reg_re;
  var reg_inv_t      [PIPE_STAGES-1 : 0] pipe_reg_inv;    


  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      foreach ( pipe_crossbar_sel[i,j]) begin
        pipe_crossbar_sel[i][j] <= '0;
      end
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
      pipe_reg_wr_mode  <=  ALU;
      pipe_ram_addr     <=  '0;
      pipe_ram_we       <=  '0;
      pipe_ram_re       <=  '0;
      pipe_reg_we       <=  '0;
      pipe_reg_re       <=  '0;
      pipe_reg_inv      <=  '0;

    end else begin
      if (in_pipe_en == 1) begin // Enable the pipe only if in_pipe_en is 1
        pipe_crossbar_pipe_en[0] <= in_crossbar_pipe_en     ;
        pipe_alu_mode    [0] <= in_alu_mode     ;
        pipe_crossbar_sel[0] <= in_crossbar_sel ;
        pipe_reg_wr_mode [0] <= in_reg_wr_mode  ;
        pipe_reg_wr_sel  [0] <= in_reg_wr_sel   ;
        pipe_ram_addr    [0] <= in_ram_addr     ;
        pipe_ram_we      [0] <= in_ram_we       ;
        pipe_ram_re      [0] <= in_ram_re       ;
        pipe_reg_rd_addr [0] <= in_reg_rd_addr  ;
        pipe_reg_we      [0] <= in_reg_we       ;
        pipe_reg_re      [0] <= in_reg_re       ;
        pipe_reg_inv     [0] <= in_reg_inv      ;

        for (integer i=1; i<PIPE_STAGES; i= i+1) begin
          pipe_crossbar_pipe_en [i] <= pipe_crossbar_pipe_en [i-1];  
          pipe_alu_mode    [i] <= pipe_alu_mode    [i-1];  
          pipe_crossbar_sel[i] <= pipe_crossbar_sel[i-1];
          pipe_reg_wr_mode [i] <= pipe_reg_wr_mode [i-1];
          pipe_reg_wr_sel  [i] <= pipe_reg_wr_sel  [i-1];
          pipe_ram_addr    [i] <= pipe_ram_addr    [i-1];
          pipe_ram_we      [i] <= pipe_ram_we      [i-1];
          pipe_ram_re      [i] <= pipe_ram_re      [i-1];
          pipe_reg_rd_addr [i] <= pipe_reg_rd_addr [i-1];
          pipe_reg_we      [i] <= pipe_reg_we      [i-1];
          pipe_reg_re      [i] <= pipe_reg_re      [i-1];
          pipe_reg_inv     [i] <= pipe_reg_inv     [i-1];
        end
      end
    end
  end

  /* Assign outputs */
  // 0th Pipe stage
  assign out_pipe_en      = in_pipe_en;
  assign out_reg_rd_addr  = in_reg_rd_addr;
  assign out_reg_inv      = in_reg_inv;
  assign out_reg_re       = in_reg_re; 
  /* assign out_ram_addr     = in_ram_addr; */
  /* assign out_ram_re       = in_ram_re; */
  /* assign out_ram_we       = in_ram_we; */

`ifdef SRAM_RF
  // 1st stage
  assign out_crossbar_sel = pipe_crossbar_sel[0];
`else
  // 0th stage
  assign out_crossbar_sel = in_crossbar_sel;
`endif // SRAM_RF

  // 1st stage
  assign out_crossbar_pipe_en = pipe_crossbar_pipe_en[0];

  // pipe stages inside trees
  genvar tree_depth_i, tree_i, alu_i;
  generate
    for (tree_i=0; tree_i < N_TREE ; tree_i= tree_i + 1) begin: tree_loop      
      for (tree_depth_i=0; tree_depth_i< TREE_DEPTH ; tree_depth_i=tree_depth_i+1) begin: tree_depth_loop
        for (alu_i= (2**(TREE_DEPTH - tree_depth_i - 1)) - 1; alu_i< (2**(TREE_DEPTH-tree_depth_i)) - 1; alu_i= alu_i + 1) begin: alu_loop
`ifdef SRAM_RF
          assign out_alu_mode[tree_i][alu_i] = pipe_alu_mode[tree_depth_i+1][tree_i][alu_i];
`else
          assign out_alu_mode[tree_i][alu_i] = pipe_alu_mode[tree_depth_i][tree_i][alu_i];
`endif
        end
      end
    end
  endgenerate

  // Memory control signals
  // RD_LATENCY cycles before write back stage
  assign out_ram_addr     = pipe_ram_addr [TREE_DEPTH - DATA_MEM_RD_LATENCY];
  assign out_ram_re       = pipe_ram_re   [TREE_DEPTH - DATA_MEM_RD_LATENCY];
  assign out_ram_we       = pipe_ram_we   [TREE_DEPTH - DATA_MEM_RD_LATENCY];

  // write back stage
`ifdef SRAM_RF
  assign out_reg_we      = pipe_reg_we     [TREE_DEPTH+1]; 
  assign out_reg_wr_sel  = pipe_reg_wr_sel [TREE_DEPTH+1]; 
  assign out_reg_wr_mode = pipe_reg_wr_mode[TREE_DEPTH+1]; 
`else
  assign out_reg_we      = pipe_reg_we     [TREE_DEPTH]; 
  assign out_reg_wr_sel  = pipe_reg_wr_sel [TREE_DEPTH]; 
  assign out_reg_wr_mode = pipe_reg_wr_mode[TREE_DEPTH]; 
`endif

  // Assertions
  assert property (@(posedge clk) if (out_ram_we) ((out_ram_we & out_ram_re) == 0)) else $warning("Both re and we cannot be high since ram is single ported: %0b, %0b\n", out_ram_re, out_ram_we);

endmodule


`endif //PIPELINED_CONTROL_DEF
