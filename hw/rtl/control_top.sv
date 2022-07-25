
`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"

`include "pipelined_control.sv" // To be changed to control top eventually
`include "instr_decd.sv"

`ifndef CONTROL_TOP_DEF
  `define CONTROL_TOP_DEF

module control_top (
  input clk, rst,

  input instr_addr_t  base_instr_addr  ,
  output instr_addr_t instr_mem_rd_addr,
  output instr_mem_rd_rdy,
  input [INSTR_L-1 : 0] instr          ,
  input instr_vld                      ,
  output instr_rdy                     ,

  output alu_mode_t      alu_mode     , 
  output crossbar_sel_t  crossbar_sel , 
  output                 pipe_en      , 
  output [N_BANKS - 1 : 0] crossbar_flop_en, 
  output [N_BANKS - 1 : 0] crossbar_pipe_en, 
  output reg_wr_mode_t   reg_wr_mode  , 
  output reg_wr_sel_t    reg_wr_sel   , 
  output ram_addr_t      ram_addr     , 
  output ram_we_t        ram_we       , 
  output ram_re_t        ram_re       , 
  output reg_rd_addr_t   reg_rd_addr  , 
  output reg_we_t        reg_we       , 
  output reg_re_t        reg_re       , 
  output reg_inv_t       reg_inv       

  ); 
  // Signals

  // Interfaces
  controller_if controller_if_ins(
    .clk            (clk      ),
    .rst            (rst      ),
    .instr          (instr    ),
    .instr_vld      (instr_vld),
    .base_instr_addr(base_instr_addr)
  );

  // Instances
  controller controller_ins(controller_if_ins);
  assign crossbar_flop_en = controller_if_ins.crossbar_flop_en;
  pipelined_control pipelined_control_ins(
  .clk (clk),
  .rst (rst),
  .in_alu_mode      (controller_if_ins.alu_mode     ) , 
  .in_crossbar_sel  (controller_if_ins.crossbar_sel ) , 
  .in_pipe_en       (controller_if_ins.pipe_en      ) , 
  .in_crossbar_pipe_en(controller_if_ins.crossbar_pipe_en) , 
  .in_reg_wr_mode   (controller_if_ins.reg_wr_mode  ) , 
  .in_reg_wr_sel    (controller_if_ins.reg_wr_sel   ) , 
  .in_ram_addr      (controller_if_ins.ram_addr     ) , 
  .in_ram_we        (controller_if_ins.ram_we       ) , 
  .in_ram_re        (controller_if_ins.ram_re       ) , 
  .in_reg_rd_addr   (controller_if_ins.reg_rd_addr  ) , 
  .in_reg_we        (controller_if_ins.reg_we       ) , 
  .in_reg_re        (controller_if_ins.reg_re       ) , 
  .in_reg_inv       (controller_if_ins.reg_inv      ) , 

  .out_alu_mode     (alu_mode    ) , 
  .out_crossbar_sel (crossbar_sel) , 
  .out_pipe_en      (pipe_en     ) , 
  .out_crossbar_pipe_en(crossbar_pipe_en) , 
  .out_reg_wr_mode  (reg_wr_mode ) , 
  .out_reg_wr_sel   (reg_wr_sel  ) , 
  .out_ram_addr     (ram_addr    ) , 
  .out_ram_we       (ram_we      ) , 
  .out_ram_re       (ram_re      ) , 
  .out_reg_rd_addr  (reg_rd_addr ) , 
  .out_reg_we       (reg_we      ) , 
  .out_reg_re       (reg_re      ) , 
  .out_reg_inv      (reg_inv     ) 
  ); 

  assign instr_rdy = controller_if_ins.controller_rdy;
  assign instr_mem_rd_addr = controller_if_ins.instr_mem_rd_addr;
  assign instr_mem_rd_rdy = controller_if_ins.instr_mem_rd_rdy;
endmodule


`endif //CONTROL_TOP_DEF
