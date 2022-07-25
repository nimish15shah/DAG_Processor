//=======================================================================
// Created by         : KU Leuven
// Filename           : instr_decd.sv
// Author             : Nimish Shah
// Created On         : 2019-09-25 16:27
// Last Modified      : 
// Update Count       : 2019-09-25 16:27
// Description        : Instruction decoder
//                      
//=======================================================================

`ifndef INSTR_DECD_DEF
  `define INSTR_DECD_DEF

`include "common_pkg.sv"
`include "utils_pkg.sv"
`include "instr_decd_pkg.sv"
`include "instr_mem_addr_gen.sv"

//===========================================
//       Interfaces
//===========================================
interface controller_if (input clk, rst, input [INSTR_L-1:0] instr, input instr_vld, input instr_addr_t base_instr_addr);
  import instr_decd_pkg::*;
  import hw_config_pkg::*;
  import controller_type_defs::*;
  
  logic        controller_rdy;
  instr_addr_t instr_mem_rd_addr;
  logic instr_mem_rd_rdy;
  
  /* Control to Datapath */
  var   alu_mode_t     alu_mode;
  var   crossbar_sel_t crossbar_sel; // Always use packed array as casting from packed to unpacked is not straightforward.
  logic pipe_en;
  logic [N_BANKS - 1: 0] crossbar_flop_en;
  logic [N_BANKS - 1 : 0] crossbar_pipe_en;
  var   reg_wr_mode_t  reg_wr_mode;
  var   reg_wr_sel_t   reg_wr_sel;

  /* Control to RAM */
  var ram_addr_t ram_addr;
  var ram_we_t   ram_we;
  var ram_re_t   ram_re;

  /* Control to Regbank */ 
  var reg_rd_addr_t reg_rd_addr;
  var reg_we_t      reg_we;
  var reg_re_t      reg_re;
  var reg_inv_t     reg_inv;

  modport controller_mp (input clk, rst, instr_vld, instr, base_instr_addr,
                         output controller_rdy, alu_mode, crossbar_sel, pipe_en, crossbar_flop_en, crossbar_pipe_en, ram_addr, ram_we, ram_re, reg_wr_mode, reg_wr_sel, reg_rd_addr, reg_we, reg_re, reg_inv, instr_mem_rd_addr, instr_mem_rd_rdy);
//  modport instr_mem_mp (input clk, rst, controller_rdy, output instr_vld, instr);
//  modport pipelined_control_mp ( input alu_mode, crossbar_sel, pipe_en, ram_addr, ram_we, ram_re, reg_wr_mode, reg_wr_sel, reg_rd_addr, reg_we, reg_inv);
endinterface

//===========================================
//       Modules
//===========================================

module controller ( controller_if.controller_mp io); 
  import instr_decd_pkg::*;
  import hw_config_pkg::*;
  import controller_type_defs::*;
  
  logic [INSTR_L -1 : 0] fetcher_instr; 
  logic fetcher_instr_vld;
  logic decd_rdy;
  logic out_fetcher_rdy;
  logic instr_vld_to_fetcher;

  instr_decd instr_decd_ins(
      .clk               ( io.clk),
      .rst               ( io.rst),
      .fetcher_instr     ( fetcher_instr),
      .fetcher_instr_vld ( fetcher_instr_vld),
      .decd_rdy          ( decd_rdy),
      .out_alu_mode          ( io.alu_mode),
      .out_crossbar_sel      ( io.crossbar_sel),
      .out_pipe_en           ( io.pipe_en),
      .out_crossbar_flop_en  ( io.crossbar_flop_en),
      .out_crossbar_pipe_en  ( io.crossbar_pipe_en),
      .out_ram_addr          ( io.ram_addr),
      .out_ram_we            ( io.ram_we),
      .out_ram_re            ( io.ram_re),
      .out_reg_wr_mode       ( io.reg_wr_mode),
      .out_reg_wr_sel        ( io.reg_wr_sel),
      .out_reg_rd_addr       ( io.reg_rd_addr),
      .out_reg_we            ( io.reg_we),
      .out_reg_re            ( io.reg_re),
      .out_reg_inv           ( io.reg_inv)
  );

  instr_fetcher instr_fetcher_ins(
      .clk               ( io.clk),
      .rst               ( io.rst),
      .instr_vld         ( instr_vld_to_fetcher),
      .instr             ( io.instr),
      .decd_rdy          ( decd_rdy),
      .fetcher_instr_vld ( fetcher_instr_vld),
      .fetcher_instr     ( fetcher_instr),
      .out_fetcher_rdy   (out_fetcher_rdy)
   );
  instr_mem_addr_gen instr_mem_addr_gen_ins(
    .clk              (io.clk              ),
    .rst              (io.rst              ),
    .fetcher_rdy      (out_fetcher_rdy     ),
    .instr_vld        (io.instr_vld        ),
    .base_instr_addr  (io.base_instr_addr  ),
    .instr_mem_rd_addr(io.instr_mem_rd_addr),
    .instr_mem_rd_rdy (io.instr_mem_rd_rdy ),
    .instr_vld_to_fetcher(instr_vld_to_fetcher)
  ); 

  assign io.controller_rdy= out_fetcher_rdy;

  initial instr_decd_pkg_asserts();
endmodule

module instr_decd(
  input clk, rst, [INSTR_L -1 : 0] fetcher_instr, 
  input fetcher_instr_vld, 
  output decd_rdy, 
  output alu_mode_t      out_alu_mode     , 
  output crossbar_sel_t  out_crossbar_sel , 
  output                 out_pipe_en      , 
  output [N_BANKS - 1 : 0] out_crossbar_flop_en, 
  output [N_BANKS - 1 : 0] out_crossbar_pipe_en, 
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
  import controller_type_defs::*;
  import opcodes::*;
  import instr_decd_pkg::*;

  alu_mode_t     alu_mode;
  crossbar_sel_t crossbar_sel;
  logic          pipe_en;
  logic [N_BANKS - 1: 0] crossbar_flop_en;
  logic [N_BANKS - 1: 0] crossbar_pipe_en;
  ram_addr_t     ram_addr;
  ram_we_t       ram_we;
  ram_re_t       ram_re;
  reg_wr_mode_t  reg_wr_mode;
  reg_wr_sel_t   reg_wr_sel;
  reg_rd_addr_t  reg_rd_addr;
  reg_we_t       reg_we;
  reg_re_t       reg_re;
  reg_inv_t      reg_inv;
  
  logic [INSTR_L -1 : 0] instr;
  assign instr= fetcher_instr;

  logic [OPCODE_L-1 :0 ] opcode;
  assign opcode = instr[OPCODE_L-1 : 0];

  //===========================================
  //       Helping signals
  //===========================================
  logic [N_BANKS-1 : 0] decompress_vect;
  always_comb begin
    case (opcode)
      ST_4_OPCODE: decompress_vect = instr[ST_4_EN_L + ST_4_EN_S - 1 : ST_4_EN_S];
      ST_8_OPCODE: decompress_vect = instr[ST_8_EN_L + ST_8_EN_S - 1 : ST_8_EN_S];
      CP_8_OPCODE: decompress_vect = instr[CP_8_EN_L + CP_8_EN_S - 1 : CP_8_EN_S];
      CP_4_OPCODE: decompress_vect = instr[CP_4_EN_L + CP_4_EN_S - 1 : CP_4_EN_S];
      CP_2_OPCODE: decompress_vect = instr[CP_2_EN_L + CP_2_EN_S - 1 : CP_2_EN_S];
      default : decompress_vect = '0;
    endcase
  end

  logic [N_BANKS-1 : 0] [$clog2(MAX_COMPRESSED_BANKS) : 0] decompress_sel; // Minimum compression allowed: x4
  always_comb begin
    decompress_sel[0] = decompress_vect[0] - 1;
    for (integer i=1; i< N_BANKS ; i=i+1) begin
      decompress_sel[i] = decompress_sel[i-1] + decompress_vect[i];
    end
  end

  logic [MAX_COMPRESSED_BANKS -1 : 0] [$clog2(BANK_DEPTH) - 1 : 0] compressed_addrs;
  logic [MAX_COMPRESSED_BANKS -1 : 0] [$clog2(N_BANKS) - 1 : 0] compressed_crossbar_sels;
  logic [MAX_COMPRESSED_BANKS -1 : 0] compressed_invalids;
  always_comb begin
    //compressed_addrs = {MAX_COMPRESSED_BANKS{(($clog2(BANK_DEPTH)){1'bx}}};    
    foreach ( compressed_addrs[i]) begin
      compressed_addrs[i] = '0;    
    end
    foreach ( compressed_crossbar_sels[i]) begin
      compressed_crossbar_sels[i] = '0;    
    end
    compressed_invalids = '0;
    case (opcode)
      ST_4_OPCODE: begin
        if (N_BANKS >= 16) begin
          for (integer i=0; i< 4 && i< MAX_COMPRESSED_BANKS; i=i+1) begin
            compressed_addrs[i] = instr[(i+1)*BANK_ADDR_L + ST_4_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
          end
        end
      end 
      ST_8_OPCODE: begin
        if (N_BANKS >= 32) begin
          for (integer i=0; i< 8 && i< MAX_COMPRESSED_BANKS; i=i+1) begin
            compressed_addrs[i] = instr[(i+1)*BANK_ADDR_L + ST_8_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
          end
        end
      end 
      CP_8_OPCODE: begin
        /* compressed_invalids[7:0] = instr[CP_8_INVLD_S + CP_8_INVLD_L - 1 -: 8]; */
        if (N_BANKS >= 32) begin
          for (integer i=0; i< 8 && i< MAX_COMPRESSED_BANKS; i=i+1) begin
            compressed_addrs[i] = instr[(i+1)*BANK_ADDR_L + CP_8_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
            compressed_crossbar_sels[i] = instr[(i+1)*CROSSBAR_SEL_L + CP_8_CROSS_SEL_S - 1 -: CROSSBAR_SEL_L];
            compressed_invalids[i] = instr[CP_8_INVLD_S + i];
          end
        end
      end 
      CP_4_OPCODE: begin
        /* compressed_invalids[4:0] = instr[CP_4_INVLD_S + CP_4_INVLD_L - 1 -: 4]; */
        if (N_BANKS >= 16) begin
          for (integer i=0; i< 4 && i< MAX_COMPRESSED_BANKS; i=i+1) begin
            compressed_addrs[i] = instr[(i+1)*BANK_ADDR_L + CP_4_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
            compressed_crossbar_sels[i] = instr[(i+1)*CROSSBAR_SEL_L + CP_4_CROSS_SEL_S - 1 -: CROSSBAR_SEL_L];
            compressed_invalids[i] = instr[CP_4_INVLD_S + i];
          end
        end
      end 
      CP_2_OPCODE: begin
        /* compressed_invalids[2:0] = instr[CP_2_INVLD_S + CP_2_INVLD_L - 1 -: 2]; */
        if (N_BANKS >= 8) begin
          for (integer i=0; i< 2 && i< MAX_COMPRESSED_BANKS; i=i+1) begin
            compressed_addrs[i] = instr[(i+1)*BANK_ADDR_L + CP_2_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
            compressed_crossbar_sels[i] = instr[(i+1)*CROSSBAR_SEL_L + CP_2_CROSS_SEL_S - 1 -: CROSSBAR_SEL_L];
            compressed_invalids[i] = instr[CP_2_INVLD_S + i];
          end
        end
      end 
      default : begin
        foreach ( compressed_addrs[i]) begin
          compressed_addrs[i] = '0;    
        end
        foreach ( compressed_crossbar_sels[i]) begin
          compressed_crossbar_sels[i] = '0;    
        end
        compressed_invalids = '0;
      end
    endcase
  end

  //===========================================
  //    Generate control signals
  //===========================================
  always_comb begin
    /* Default values */
    for (integer i=0; i< N_BANKS; i=i+1) begin
      /* crossbar_sel[i] = '0; */
      crossbar_sel[i] = i;
    end
    for (integer i=0; i< N_TREE ; i=i+1) begin
      for (integer j=0; j< N_ALU_PER_TREE ; j=j+1) begin
//        alu_mode[i][j] = $cast(alu_mode[i][j], 'b0);
        alu_mode[i][j] = PASS_1;
      end
    end
    pipe_en = '0;
    crossbar_flop_en= '0;
    crossbar_pipe_en= '0;
    ram_addr = '0;
    ram_we = '0;
    ram_re = '0;
    reg_wr_mode = ALU;
    foreach (reg_wr_sel[i]) begin
      reg_wr_sel[i] = '0;
    end

    for (integer i=0; i< N_BANKS; i=i+1) begin
      reg_rd_addr[i] = '0;
    end
    reg_we = '0;
    reg_re = '0;
    reg_inv = '0;
    
    if (fetcher_instr_vld == 1) begin // Decode only if instruction valid
      case (opcode)
        NOP_OPCODE: begin
          pipe_en= 1'b1;
        end

        NOP_STALL_OPCODE: begin
          // FIXME: stall the pipe. Unstalled now for testing.
          pipe_en= 1'b1;
        end

        LD_OPCODE: begin
          pipe_en = 1'b1;
          reg_wr_mode= RAM;

          ram_re= instr[LD_EN_S + LD_EN_L -1 : LD_EN_S];
          reg_we= instr[LD_EN_S + LD_EN_L -1 : LD_EN_S];

          ram_addr= instr[LD_MEM_ADDR_S + LD_MEM_ADDR_L - 1: LD_MEM_ADDR_S];
        end

        ST_OPCODE: begin
          pipe_en = 1'b1;
          ram_we= instr[ST_EN_S + ST_EN_L - 1: ST_EN_S];
          reg_re= instr[ST_EN_S + ST_EN_L - 1: ST_EN_S];
          reg_inv = instr[ST_EN_S + ST_EN_L - 1: ST_EN_S];

          ram_addr= instr[ST_MEM_ADDR_S + ST_MEM_ADDR_L - 1: ST_MEM_ADDR_S];
          
          for (integer i=0; i< N_BANKS ; i=i+1) begin
            reg_rd_addr[i]= instr[ (i+1)*BANK_ADDR_L + ST_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
          end
          crossbar_pipe_en = ram_we;
          crossbar_flop_en = ram_we;
        end
        ST_8_OPCODE: begin
          pipe_en = 1'b1;
          if (N_BANKS >= 32) begin
            ram_we= instr[ST_8_EN_S + ST_8_EN_L - 1: ST_8_EN_S];
            reg_re= instr[ST_8_EN_S + ST_8_EN_L - 1: ST_8_EN_S];
            reg_inv = instr[ST_8_EN_S + ST_8_EN_L - 1: ST_8_EN_S];
            ram_addr= instr[ST_8_MEM_ADDR_S + ST_8_MEM_ADDR_L - 1: ST_8_MEM_ADDR_S];

            for (integer i=0; i< N_BANKS ; i=i+1) begin
              if (reg_re[i] == 1) begin
                reg_rd_addr[i] = compressed_addrs[decompress_sel[i]];
              end
            end
            crossbar_pipe_en = ram_we;
            crossbar_flop_en = ram_we;
          end else begin
            $error("Incorrect opcode %0d", opcode);
          end
        end
        ST_4_OPCODE: begin
          pipe_en = 1'b1;
          if (N_BANKS >= 16) begin
            ram_we= instr[ST_4_EN_S + ST_4_EN_L - 1: ST_4_EN_S];
            reg_re= instr[ST_4_EN_S + ST_4_EN_L - 1: ST_4_EN_S];
            reg_inv = instr[ST_4_EN_S + ST_4_EN_L - 1: ST_4_EN_S];
            ram_addr= instr[ST_4_MEM_ADDR_S + ST_4_MEM_ADDR_L - 1: ST_4_MEM_ADDR_S];

            // TODO: Pontetial of using less muxes in decompression
            for (integer i=0; i< N_BANKS ; i=i+1) begin
              if (reg_re[i] == 1) begin
                reg_rd_addr[i] = compressed_addrs[decompress_sel[i]];
              end
            end
            crossbar_pipe_en = ram_we;
            crossbar_flop_en = ram_we;
          end else begin
            $error("Incorrect opcode %0d", opcode);
          end
        end
        CP_OPCODE: begin
          pipe_en = 1'b1;
          reg_wr_mode= CROSSBAR;
          reg_we = instr[CP_WR_EN_S + CP_WR_EN_L - 1: CP_WR_EN_S];
          reg_re = instr[CP_RD_EN_S + CP_RD_EN_L - 1: CP_RD_EN_S];
          reg_inv = instr[CP_INVLD_S + CP_INVLD_L - 1: CP_INVLD_S];

          for (integer i=0; i< N_BANKS ; i=i+1) begin
            reg_rd_addr[i]= instr[ (i+1)*BANK_ADDR_L + CP_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
            crossbar_sel[i] = instr[(i+1)*CROSSBAR_SEL_L + CP_CROSS_SEL_S - 1 -: CROSSBAR_SEL_L];
          end
          crossbar_pipe_en = reg_we;
          crossbar_flop_en = reg_we;
        end
        CP_8_OPCODE: begin
          pipe_en = 1'b1;
          if (N_BANKS >= 32) begin
            reg_wr_mode= CROSSBAR;
            reg_we = decompress_vect;
            
            for (integer i=0; i< 8 && i < MAX_COMPRESSED_BANKS; i=i+1) begin
              if (compressed_invalids[i] == 1) begin
                reg_inv[compressed_crossbar_sels[i]] = 1'b1;
              end
            end
            for (integer i=0; i< N_BANKS ; i=i+1) begin
              if (reg_we[i] == 1) begin
                crossbar_sel[i] = compressed_crossbar_sels[decompress_sel[i]]; // Rd bank
                reg_rd_addr[crossbar_sel[i]] = compressed_addrs[decompress_sel[i]];
                reg_re[crossbar_sel[i]] = 1'b1;
              end
            end
            crossbar_pipe_en = reg_we;
            crossbar_flop_en = reg_we;
          end else begin
            $error("Incorrect opcode %0d", opcode);
          end
          
        end
        CP_4_OPCODE: begin
          pipe_en = 1'b1;
          if (N_BANKS >= 16) begin
            reg_wr_mode= CROSSBAR;
            reg_we = decompress_vect;
            
            for (integer i=0; i< 4 && i < MAX_COMPRESSED_BANKS; i=i+1) begin
              if (compressed_invalids[i] == 1) begin
                reg_inv[compressed_crossbar_sels[i]] = 1'b1;
              end
            end
            for (integer i=0; i< N_BANKS ; i=i+1) begin
              /* crossbar_sel[i] = compressed_crossbar_sels[decompress_sel[i]]; */
              /* reg_rd_addr[i] = compressed_addrs[decompress_sel[i]]; */
              /* if (reg_we[i] == 1) begin */
              /*   reg_re[crossbar_sel[i]] = 1'b1; */
              /* end */
              if (reg_we[i] == 1) begin
                crossbar_sel[i] = compressed_crossbar_sels[decompress_sel[i]]; // Rd bank
                reg_rd_addr[crossbar_sel[i]] = compressed_addrs[decompress_sel[i]];
                reg_re[crossbar_sel[i]] = 1'b1;
              end
            end
            crossbar_pipe_en = reg_we;
            crossbar_flop_en = reg_we;
          end else begin
            $error("Incorrect opcode %0d", opcode);
          end
          
        end
        CP_2_OPCODE: begin
          pipe_en = 1'b1;
          if (N_BANKS >= 8) begin
            reg_wr_mode= CROSSBAR;
            reg_we = decompress_vect;
            
            for (integer i=0; i< 2 && i < MAX_COMPRESSED_BANKS; i=i+1) begin
              reg_re[compressed_crossbar_sels[i]] = 1'b1;
              if (compressed_invalids[i] == 1) begin
                reg_inv[compressed_crossbar_sels[i]] = 1'b1;
              end
            end
            for (integer i=0; i< N_BANKS ; i=i+1) begin
              /* crossbar_sel[i] = compressed_crossbar_sels[decompress_sel[i]]; */
              /* reg_rd_addr[i] = compressed_addrs[decompress_sel[i]]; */
              /* if (reg_we[i] == 1) begin */
              /*   reg_re[crossbar_sel[i]] = 1'b1; */
              /* end */
              if (reg_we[i] == 1) begin
                crossbar_sel[i] = compressed_crossbar_sels[decompress_sel[i]]; // Rd bank
                reg_rd_addr[crossbar_sel[i]] = compressed_addrs[decompress_sel[i]];
                reg_re[crossbar_sel[i]] = 1'b1;
              end
            end
            crossbar_pipe_en = reg_we;
            crossbar_flop_en = reg_we;
          end else begin
            $error("Incorrect opcode %0d", opcode);
          end
          
        end
        
        BB_OPCODE: begin
          pipe_en = 1'b1;
          reg_wr_mode = ALU;
          reg_inv = instr[BB_INVLD_L + BB_INVLD_S - 1: BB_INVLD_S];
          for (integer i=0; i< N_BANKS ; i=i+1) begin
            crossbar_sel[i] = instr[(i+1)*CROSSBAR_SEL_L + BB_CROSS_SEL_S - 1 -: CROSSBAR_SEL_L];
            reg_rd_addr[i]= instr[ (i+1)*BANK_ADDR_L + BB_BANK_RD_ADDR_S -1 -: BANK_ADDR_L];
            reg_wr_sel[i]= instr[(i+1)*$bits(reg_wr_sel[i]) + BB_BANK_WR_SEL_S -1 -: $bits(reg_wr_sel[i])];
          end

          // NOTE: This loop depends on the previous loop, so do NOT interchange the order of loops
          for (integer i=0; i< N_BANKS ; i=i+1) begin
            // FIXME : reg_re enabled for bank=0 by default
            reg_re[crossbar_sel[i]] = 1'b1;
            if (reg_wr_sel[i] == 0) begin // Indicate not to write output
              reg_we[i] = 1'b0;
            end else begin
              reg_we[i] = 1'b1;
            end
          end
          
          for (integer i=0; i< N_TREE ; i=i+1) begin
            for (integer j=0; j< N_ALU_PER_TREE ; j=j+1) begin
              // Take appropriate slice out of instr and cast it to alu_mode_enum_t
              alu_mode[i][j] = op2alu_mode(instr[BB_ARITH_OP_S + (i*N_ALU_PER_TREE + j+1)*$bits(alu_mode_enum_t) -1 -: $bits(alu_mode_enum_t)]);
            end
          end
          crossbar_flop_en= '1;
        end

        default: begin 
          pipe_en = '0;
          assert (0);
        end
      endcase
    end
  end

  assign out_alu_mode     = alu_mode;
  assign out_crossbar_sel = crossbar_sel;
  assign out_pipe_en      = pipe_en;
  assign out_crossbar_flop_en      = crossbar_flop_en;
  assign out_crossbar_pipe_en      = crossbar_pipe_en;
  assign out_ram_addr     = ram_addr;
  assign out_ram_we       = ram_we;
  assign out_ram_re       = ram_re;
  assign out_reg_wr_mode  = reg_wr_mode;
  assign out_reg_wr_sel   = reg_wr_sel;
  assign out_reg_rd_addr  = reg_rd_addr;
  assign out_reg_we       = reg_we;
  assign out_reg_re       = reg_re;
  assign out_reg_inv      = reg_inv;    
  assign decd_rdy     = 1;


  `ifdef DEBUG
  initial begin
    forever begin
      @(posedge clk);
      if (rst == RESET_STATE) break;
    end
    forever begin
      @(posedge clk);
      if (opcode != '0) break;
    end

    forever begin
    $display("INSTR_DECD: opcode %0d, reg_we %0b, reg_inv %0b, reg_re %0b, reg_wr_mode : %0d, ram_we: %0b, ram_re: %0b, ram_addr %0d, time %0t\n", opcode, reg_we, reg_inv, reg_re, reg_wr_mode, ram_we, ram_re, ram_addr, $time);
    @(posedge clk);
    end
  end
  `endif

endmodule
  

//===========================================
//       Instruction Fetcher
// A module that prefetches 1 instruction and shifts 
// according to varibale instruction length
// 
//  instr_reg_q
//  2*L_______________________0
//  |_________________________| -> shift this side
//   /\                   \/
//  new_instr         instr to decoder 
//  enter from           [L-1 : 0]
//  here
//===========================================

module instr_fetcher (
   input clk,
   input rst,
   input instr_vld,
   input [INSTR_L-1 : 0]instr,
   input decd_rdy,
   
   output fetcher_instr_vld,
   output [INSTR_L-1 : 0] fetcher_instr,
   output out_fetcher_rdy
  ); 

  logic fetcher_rdy;

  /* Internal signals */
  logic [2*INSTR_L-1 : 0] instr_reg_q;
  logic [2*INSTR_L-1 : 0] instr_reg_d;
  logic [2*INSTR_L-1 : 0] instr_reg_shifted;
  logic [2*INSTR_L-1 : 0] new_instr_shifted;

  logic [OPCODE_L-1 :0 ] opcode;
  assign opcode = instr_reg_q[OPCODE_L-1 : 0];

  logic [$clog2(2*INSTR_L) : 0] first_nonvalid_bit_q; // One extra bit to represent numbers higher than 2*INSTR_L, 
                                                    // as first non valid bit could be beyond 2*INSTR_L if every bit is valid
  logic [$clog2(2*INSTR_L) : 0] first_nonvalid_bit_d;

  typedef enum logic[1:0] {OLD, SHIFTED, NEW} mux_sel_enum_t;
  mux_sel_enum_t [2*INSTR_L - 1 : 0] mux_sel;

  logic [$clog2(INSTR_L) : 0] curr_instr_len;
  logic [$clog2(INSTR_L) : 0] new_instr_shift_cnt;


  //===========================================
  //       Instansiation
  //===========================================
  instr_len_decode instr_len_decode_ins(.opcode(opcode), .curr_instr_len(curr_instr_len));

  //===========================================
  //      Flow control 
  //===========================================
  logic fetcher_2_decd_vld;
  logic send_2_decd_en, fetcher_gets_new_instr;

  always_comb begin
    if (curr_instr_len < first_nonvalid_bit_q) begin // There are some valid bits
      fetcher_2_decd_vld = 1;
    end else begin
      fetcher_2_decd_vld = 0;
    end
  end

  always_comb begin
    if ( (send_2_decd_en && (first_nonvalid_bit_q-curr_instr_len <= ($bits(instr_reg_q) - $bits(instr)))) || (first_nonvalid_bit_q < ($bits(instr_reg_q) - $bits(instr))) ) begin // There is space to pop from FIFO
      fetcher_rdy = 1;
    end else begin
      fetcher_rdy= 0;
    end
  end

  assign send_2_decd_en = fetcher_2_decd_vld & decd_rdy;
  assign fetcher_gets_new_instr = fetcher_rdy & instr_vld;
  
  //===========================================
  //       Combinational blocks
  //===========================================
  // first_nonvalid_bit_d, new_instr_shift_cnt, and MUX select
  always_comb begin
    // Initialize
    foreach (mux_sel[i]) begin
      mux_sel[i] = OLD; // Default
    end
    new_instr_shift_cnt = 0;

    if (send_2_decd_en) begin // Instruction consumed by decoder
      new_instr_shift_cnt =   $bits(instr_reg_q) - $bits(instr) - first_nonvalid_bit_q + curr_instr_len;
      foreach (mux_sel[i]) begin
        mux_sel[i] = SHIFTED; 
      end
      if (fetcher_gets_new_instr) begin // New instruction coming in 
        first_nonvalid_bit_d = first_nonvalid_bit_q - curr_instr_len + $bits(instr);
        mux_sel[first_nonvalid_bit_q -curr_instr_len +: $bits(instr)]= {$bits(instr){NEW}};
      end else begin // New instruction NOT coming in
        first_nonvalid_bit_d = first_nonvalid_bit_q - curr_instr_len;
      end

      /* assert (curr_instr_len <= first_nonvalid_bit_q) else $warning(1); */
      /* assert ($bits(instr_reg_q) - $bits(instr) - first_nonvalid_bit_q + curr_instr_len >= 0); */

    end else begin // Instruction not cosumed by decoder
      new_instr_shift_cnt=  $bits(instr_reg_q) - $bits(instr) - first_nonvalid_bit_q; 
      if (fetcher_gets_new_instr) begin // New instruction coming in
        first_nonvalid_bit_d = first_nonvalid_bit_q + $bits(instr);
        mux_sel[first_nonvalid_bit_q +: $bits(instr)]= {$bits(instr){NEW}};
      end else begin // New instruction NOT coming in
        first_nonvalid_bit_d = first_nonvalid_bit_q;
      end
      assert ($bits(instr_reg_q) - $bits(instr) - first_nonvalid_bit_q >= 0);
    end // if (send_2_decd_en)
    /* assert (first_nonvalid_bit_d <= 2*INSTR_L) else $warning(1, "first_nonvalid_bit_d not set properly"); */
    /* assert (first_nonvalid_bit_q <= 2*INSTR_L) else $warning(1, "first_nonvalid_bit_q not set properly"); */
  end

  // instr_reg_d based on mux_sel
  always_comb begin
    for (integer i=0; i< $bits(instr_reg_d); i=i+1) begin
      case (mux_sel[i]) 
        OLD     : instr_reg_d[i] = instr_reg_q[i];
        SHIFTED : instr_reg_d[i] = instr_reg_shifted[i];
        NEW     : instr_reg_d[i] = new_instr_shifted[i];
        default : begin 
            instr_reg_d[i] = 1'bx;
            /* $warning(1); */
          end
      endcase
    end
  end
  
  // Shifting 

  always_comb begin
    instr_reg_shifted = instr_reg_q;
    case (curr_instr_len) 
      NOP_L      : instr_reg_shifted = instr_reg_q >> NOP_L      ; //begin end
      NOP_STALL_L: instr_reg_shifted = instr_reg_q >> NOP_STALL_L; //begin end
      LD_L       : instr_reg_shifted = instr_reg_q >> LD_L       ; //begin end
      CP_8_L     : instr_reg_shifted = instr_reg_q >> CP_8_L     ; //begin end
      CP_L       : instr_reg_shifted = instr_reg_q >> CP_L       ; //begin end
      CP_4_L     : instr_reg_shifted = instr_reg_q >> CP_4_L     ; //begin end
      CP_2_L     : instr_reg_shifted = instr_reg_q >> CP_2_L     ; //begin end
      ST_4_L     : instr_reg_shifted = instr_reg_q >> ST_4_L     ; //begin end
      ST_8_L     : instr_reg_shifted = instr_reg_q >> ST_8_L     ; //begin end
      ST_L       : instr_reg_shifted = instr_reg_q >> ST_L       ; //begin end
      BB_L       : instr_reg_shifted = instr_reg_q >> BB_L       ; //begin end      
      /* default : $warning; */
    endcase
  end

  assign new_instr_shifted= instr << ($bits(instr_reg_q) - $bits(instr) - new_instr_shift_cnt);
  /* assign instr_reg_shifted= instr_reg_q >> curr_instr_len; */

  //===========================================
  //       Sequential elements
  //===========================================
  // instr_reg_q
  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      instr_reg_q <= '0;
    end else begin
      instr_reg_q <= instr_reg_d;
    end
  end

  // first_nonvalid_bit_q
  always_ff @(posedge clk) begin
    if (rst== RESET_STATE) begin
      first_nonvalid_bit_q <= '0;
    end else begin
      first_nonvalid_bit_q <= first_nonvalid_bit_d;
    end
  end

  /* Outputs */
  assign fetcher_instr_vld = fetcher_2_decd_vld;
  assign fetcher_instr= instr_reg_q[INSTR_L-1 : 0];
  assign out_fetcher_rdy= fetcher_rdy;
endmodule //instr_decd

module instr_len_decode( input [OPCODE_L-1 : 0] opcode, output [$clog2(INSTR_L) : 0 ] curr_instr_len);
  logic [$clog2(INSTR_L) : 0] curr_instr_len_pre;

  always_comb begin
    curr_instr_len_pre = NOP_L;
    case (opcode) 
      NOP_OPCODE  : curr_instr_len_pre       = NOP_L      ;
      NOP_STALL_OPCODE  : curr_instr_len_pre = NOP_STALL_L;
      LD_OPCODE   : curr_instr_len_pre       = LD_L       ;
      CP_8_OPCODE : curr_instr_len_pre       = CP_8_L     ;
      CP_OPCODE : curr_instr_len_pre         = CP_L       ;
      CP_4_OPCODE : curr_instr_len_pre       = CP_4_L     ;
      CP_2_OPCODE : curr_instr_len_pre       = CP_2_L     ;
      ST_4_OPCODE : curr_instr_len_pre       = ST_4_L     ;
      ST_8_OPCODE : curr_instr_len_pre       = ST_8_L     ;
      ST_OPCODE   : curr_instr_len_pre       = ST_L       ;
      BB_OPCODE   : curr_instr_len_pre       = BB_L       ;
      /* default : begin */ 
      /*   $warning(1, "Unrecognized opcode"); */
      /* end */
    endcase
  end

  assign curr_instr_len = curr_instr_len_pre;

endmodule

`endif //INSTR_DECD_DEF

