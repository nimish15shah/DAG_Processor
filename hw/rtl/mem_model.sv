//=======================================================================
// Created by         : KU Leuven
// Filename           : mem_model.sv
// Author             : Nimish Shah
// Created On         : 2019-12-18 21:33
// Last Modified      : 
// Update Count       : 2019-12-18 21:33
// Description        : 
//                      
//=======================================================================

`ifndef MEM_MODEL_DEF
  `define MEM_MODEL_DEF

`ifdef USE_SRAM_MACRO
module my_memory #(
  parameter DATA_L= 8,
  parameter ADDR_L= 8,
  parameter N_BANKS= 8,
  parameter INIT_FILE_PATH_PREFIX= "dummy_prefix",
  parameter RD_LATENCY
)
(
  input clk,
  input rst,

  input [ADDR_L - 1 : 0] addr,
  input [N_BANKS -1 : 0] [DATA_L - 1 : 0] wr_data,
  input [N_BANKS - 1 : 0]                 wr_en,
  input [N_BANKS - 1 : 0]                 rd_en, 
  output [N_BANKS - 1 : 0] [DATA_L - 1 : 0] rd_data
);

  logic [N_BANKS - 1 : 0] ch_en;
  assign ch_en = wr_en | rd_en;

  localparam int SRAM_MACRO_WIDTH= `SRAM_MACRO_WIDTH;
  localparam int N_PARALLEL_MACROS= (DATA_L + SRAM_MACRO_WIDTH - 1)/SRAM_MACRO_WIDTH;

  logic [N_BANKS - 1 : 0] [N_PARALLEL_MACROS - 1 : 0] [SRAM_MACRO_WIDTH - 1 : 0] wr_data_macro_wise;
  logic [N_BANKS - 1 : 0] [N_PARALLEL_MACROS - 1 : 0] [SRAM_MACRO_WIDTH - 1 : 0] rd_data_macro_wise;
  logic [N_BANKS - 1 : 0] [DATA_L - 1 : 0] rd_data_d;

  always_comb begin
    wr_data_macro_wise = 'x;
    rd_data_d= 'x;
    for (integer i=0; i< N_BANKS ; i=i+1) begin
      wr_data_macro_wise [i] = wr_data[i];
      rd_data_d[i] = rd_data_macro_wise[i];
    end
  end

  assign rd_data= rd_data_d;

  generate
    genvar mem_i, par_macro_i;
    for (mem_i=0; mem_i< N_BANKS ; mem_i=mem_i+1) begin: mem_loop
      for (par_macro_i=0; par_macro_i< N_PARALLEL_MACROS ; par_macro_i=par_macro_i+1) begin: par_loop
        /* fname.hextoa(mem_i); */
        sp_mem_model #(
          .DATA_L (SRAM_MACRO_WIDTH),
          .ADDR_L (ADDR_L),
          .RD_LATENCY (RD_LATENCY)
        ) SP_MEM_MODEL_INS
        (
          .clk     (clk     ),
          .rst     (rst     ),

          .slp     (1'b0),
          .sd      (1'b0),
                            
          .wr_en   (wr_en  [mem_i]),
          .addr    (addr   ),
          .wr_data (wr_data_macro_wise[mem_i][par_macro_i]),
          .rd_data (rd_data_macro_wise[mem_i][par_macro_i]),
          .ch_en   (ch_en  [mem_i])
        ); 
      end
    end
    
  endgenerate

  `ifdef DEBUG
  initial begin
    forever begin
      @(posedge clk);
      if (rst == RESET_STATE) break;
    end
    forever begin
      @(posedge clk);
      if ((wr_en | rd_en) != '0) break;
    end
    forever begin
      if (((wr_en | rd_en) != '0) || 1) begin
        $display("MEM: addr: %0d, wr_en %0b, rd_en %0b, time: %0t",addr, wr_en, rd_en, $time);
        foreach ( wr_data[i]) begin
          /* if (wr_en[i]) begin */
            $write("bank: %0d, wr_data: %0h,", i, wr_data[i]);
          /* end */
        end
        $display(" ");
        foreach (rd_data[i]) begin
          if (rd_en[i]) begin
            $write("bank: %0d, rd_data: %0h,", i, rd_data[i]);
          end
        end
        $display(" ");
      end
    @(posedge clk);
    end
  end
  `endif

endmodule

`define MEM_CONNECTION SRAM_INS\
      (\
        .SLP   (1'b0),\
        .SD    (1'b0),\
        .CLK   (clk),\
        .CEB   (~ch_en_per_macro [ macro_i ]),\
        .WEB   (~wr_en_per_macro [ macro_i ]),\
        .AWT   (1'b0),\
        .A     (addr_per_macro [ macro_i ]),\
        .D     (wr_data_per_macro [ macro_i ]),\
        .BWEB  (32'b0),\
        .Q     (rd_data_per_macro [ macro_i ])\
      );

module sp_mem_model #(
  parameter DATA_L= 8,
  parameter ADDR_L= 8,
  parameter RD_LATENCY
)
(
  input clk,
  input rst,
  
  input slp,
  input sd,
  
  input wr_en, // active high
  input ch_en, // active high
  input [ADDR_L - 1 : 0] addr,
  input [DATA_L - 1 : 0] wr_data,
  output [DATA_L - 1 : 0] rd_data
);
  localparam ADDR_L_PER_MACRO = 
    (ADDR_L >= 13) ? 13 :
    ((ADDR_L >= 10) ? 10 :
    ((ADDR_L >= 9) ? 9 :
    ((ADDR_L >= 8) ? 8 :
    -1
  )));

  localparam N_MACROS = 2**(ADDR_L - ADDR_L_PER_MACRO);
  
  logic [N_MACROS - 1 : 0] [DATA_L - 1 : 0]rd_data_per_macro;
  logic [N_MACROS - 1 : 0] [DATA_L - 1 : 0]wr_data_per_macro;
  logic [N_MACROS - 1 : 0] [ADDR_L_PER_MACRO - 1 : 0] addr_per_macro;
  logic [N_MACROS - 1 : 0] wr_en_per_macro;
  logic [N_MACROS - 1 : 0] ch_en_per_macro;
  logic [$clog2(N_MACROS) - 1 : 0] active_macro_id;
  logic [RD_LATENCY - 1 : 0] [$clog2(N_MACROS) - 1 : 0] active_macro_id_w_latency;

  generate
    if (ADDR_L != ADDR_L_PER_MACRO) begin: MULTIPLE_MACRO
      assign active_macro_id = addr[ADDR_L_PER_MACRO +: $clog2(N_MACROS)];
      assign rd_data = rd_data_per_macro[active_macro_id_w_latency[RD_LATENCY - 1]];
    end else begin: SINGLE_MACRO
      assign active_macro_id = '0;
      assign rd_data = rd_data_per_macro[0];
    end
  endgenerate

  always_comb begin
    wr_en_per_macro = '0;
    ch_en_per_macro = '0;
    for (integer i=0; i< N_MACROS ; i=i+1) begin
      // Zero gating
      wr_data_per_macro[i] = '0;
      addr_per_macro[i] = '0;
      
      // No zero gating
      /* wr_data_per_macro[i] = wr_data; */
      /* addr_per_macro[active_macro_id] = addr[0 +: ADDR_L_PER_MACRO]; */
    end
    wr_en_per_macro[active_macro_id] = wr_en;
    ch_en_per_macro[active_macro_id] = ch_en;
    wr_data_per_macro[active_macro_id] = wr_data;
    addr_per_macro[active_macro_id] = addr[0 +: ADDR_L_PER_MACRO];
  end


  always_ff @(posedge clk or negedge rst) begin
    if (rst== RESET_STATE) begin
      active_macro_id_w_latency <= 'x;
    end else begin
      if (ch_en) begin
        active_macro_id_w_latency[0]<= active_macro_id;
        for (integer i=1; i< RD_LATENCY; i=i+1) begin
          active_macro_id_w_latency[i] <= active_macro_id_w_latency[i-1];
        end
      end
    end
  end

  generate
    genvar macro_i;
    for (macro_i=0; macro_i< N_MACROS ; macro_i=macro_i+1) begin: macro_loop
      if (ADDR_L_PER_MACRO == 13) begin: mem_8196x32
        TS1N28HPCPHVTB8192X32M8SWASO 
        `MEM_CONNECTION
      end else if (ADDR_L_PER_MACRO == 10) begin: mem_1024x32
        TS1N28HPCPHVTB1024X32M4SWASO 
        `MEM_CONNECTION
      end else if (ADDR_L_PER_MACRO == 9) begin: mem_512x32
        TS1N28HPCPHVTB512X32M4SWASO 
        `MEM_CONNECTION
      end else if (ADDR_L_PER_MACRO == 8) begin: mem_256x32
        TS1N28HPCPHVTB256X32M4SWASO 
        `MEM_CONNECTION
      end else begin: error_loop
        initial begin
          $fatal("FATAL!!!!!!!!!!!!!!!!!!"); // undefine name
        end
      end
    end

  endgenerate


  // asserts
  initial begin
    assert (N_MACROS >= 1) else $fatal("PROBLEM");
    assert (ADDR_L >= 8) else $fatal("PROBLEM");
    assert (ADDR_L_PER_MACRO <= ADDR_L) else $fatal("PROBLEM");
    assert (ADDR_L_PER_MACRO >= 1) else $fatal("PROBLEM");
    assert (DATA_L == 32) else $fatal("PROBLEM");
  end

endmodule

/* module sp_mem_model #( */
/*   parameter DATA_L= 8, */
/*   parameter ADDR_L= 8, */
/*   parameter INIT_FILE_PATH_PREFIX= "dummy_prefix", */
/*   parameter RD_LATENCY */
/* ) */
/* ( */
/*   input clk, */
/*   input rst, */
  
/*   input slp, */
/*   input sd, */
  
/*   input wr_en, // active high */
/*   input ch_en, // active high */
/*   input [ADDR_L - 1 : 0] addr, */
/*   input [DATA_L - 1 : 0] wr_data, */
/*   output [DATA_L - 1 : 0] rd_data */
/* ); */ 
  
/*   logic [31 : 0] wr_data_resized_32; */
/*   logic [23 : 0] wr_data_resized_24; */
/*   logic [31 : 0] rd_data_resized_32; */
/*   logic [23 : 0] rd_data_resized_24; */
/*   logic [DATA_L - 1 : 0] rd_data_pre; */

/*   assign wr_data_resized_32 = wr_data; */
/*   assign wr_data_resized_24 = wr_data; */


/*   generate */
/*     if (DATA_L >24 && DATA_L <= 32 && ADDR_L == 10) begin: mem_1024x32 */
/*       assign rd_data_pre = rd_data_resized_32; */
/*       TS1N28HPCPHVTB1024X32M4SWBASO SRAM_1024x32 ( */
/*         .SLP   (slp), */
/*         .SD    (sd), */
/*         .CLK   (clk), */
/*         .CEB   (~ch_en), // active low */
/*         .WEB   (~wr_en), //active low */
/*         .CEBM  (1'b1), */
/*         .WEBM  (1'b1), */
/*         .AWT   (1'b0), */
/*         .A     (addr), */
/*         .D     (wr_data_resized_32), */
/*         .BWEB  ('0), */
/*         .AM    ('0), */
/*         .DM    ('0), */
/*         .BWEBM ({32{1'b1}}), */
/*         .BIST  (1'b0), */
/*         .Q     (rd_data_resized_32) */
/*       ); */
/*     end else if (DATA_L <= 24 && ADDR_L == 10) begin: mem_1024x24 */
/*       assign rd_data_pre = rd_data_resized_24; */
/*       TS1N28HPCPHVTB1024X24M4SWBASO SRAM_1024x24 ( */
/*         .SLP   (slp), */
/*         .SD    (sd), */
/*         .CLK   (clk), */
/*         .CEB   (~ch_en), // active low */
/*         .WEB   (~wr_en), //active low */
/*         .CEBM  (1'b1), */
/*         .WEBM  (1'b1), */
/*         .AWT   (1'b0), */
/*         .A     (addr), */
/*         .D     (wr_data_resized_24), */
/*         .BWEB  ('0), */
/*         .AM    ('0), */
/*         .DM    ('0), */
/*         .BWEBM ({24{1'b1}}), */
/*         .BIST  (1'b0), */
/*         .Q     (rd_data_resized_24) */
/*       ); */
/*     end else if (DATA_L <= 24 && ADDR_L == 9) begin: mem_512x24 */
/*       assign rd_data_pre = rd_data_resized_24; */
/*       TS1N28HPCPHVTB512X24M4SWBASO SRAM_512x24 ( */
/*         .SLP   (slp), */
/*         .SD    (sd), */
/*         .CLK   (clk), */
/*         .CEB   (~ch_en), // active low */
/*         .WEB   (~wr_en), //active low */
/*         .CEBM  (1'b1), */
/*         .WEBM  (1'b1), */
/*         .AWT   (1'b0), */
/*         .A     (addr), */
/*         .D     (wr_data_resized_24), */
/*         .BWEB  ('0), */
/*         .AM    ('0), */
/*         .DM    ('0), */
/*         .BWEBM ({24{1'b1}}), */
/*         .BIST  (1'b0), */
/*         .Q     (rd_data_resized_24) */
/*       ); */
/*     end else if (DATA_L > 24 && DATA_L <= 32 && ADDR_L == 9) begin: mem_512x32 */
/*       assign rd_data_pre = rd_data_resized_32; */
/*       TS1N28HPCPHVTB512X32M4SWBASO SRAM_512x32 ( */
/*         .SLP   (slp), */
/*         .SD    (sd), */
/*         .CLK   (clk), */
/*         .CEB   (~ch_en), // active low */
/*         .WEB   (~wr_en), //active low */
/*         .CEBM  (1'b1), */
/*         .WEBM  (1'b1), */
/*         .AWT   (1'b0), */
/*         .A     (addr), */
/*         .D     (wr_data_resized_32), */
/*         .BWEB  ('0), */
/*         .AM    ('0), */
/*         .DM    ('0), */
/*         .BWEBM ({32{1'b1}}), */
/*         .BIST  (1'b0), */
/*         .Q     (rd_data_resized_32) */
/*       ); */
/*     end */
/*   endgenerate */
  
/*   assign rd_data = rd_data_pre; */

/*   initial begin */
/*     //assert (DATA_L == 32 || DATA_L == 24); */
/*     assert (ADDR_L == 10 || ADDR_L == 9 ); //|| ADDR_L == 8); */
/*   end */

/*   assert property (@(posedge clk) if (wr_en) (!$isunknown(wr_data))) else $warning("%h", wr_data); */
/*   assert property (@(posedge clk) if (ch_en) (!$isunknown(addr))) else $warning("%h", addr); */

/*   assert property (@(posedge clk) (ch_en && ~wr_en) |=> (!$isunknown(rd_data_pre))) else $warning("rd_data_pre cannot be unknow after a read request: %h", rd_data_pre); */
/* endmodule */

`else // do not use USE_SRAM_MACRO
  //===========================================
  //       Functional model for memory
  //===========================================
  //===========================================
  //       Functional model for memory
  //===========================================
module sp_mem_model #(
  parameter DATA_L= 8,
  parameter ADDR_L= 8,
  parameter RD_LATENCY
)
(
  input clk,
  input rst,

  input slp,
  input sd,
  
  input wr_en,
  input ch_en,
  input [ADDR_L - 1 : 0] addr,
  input [DATA_L - 1 : 0] wr_data,
  output [DATA_L - 1 : 0] rd_data
); 
  
  logic [2**ADDR_L - 1 : 0] [DATA_L - 1 : 0] data_array;
  logic [DATA_L - 1 : 0] rd_data_based_on_en;
  logic [RD_LATENCY - 1 : 0] [DATA_L - 1 : 0] rd_data_delayed;

  always_ff @(posedge clk or negedge rst) begin
    if (rst== RESET_STATE) begin
      data_array <= 'x;
    end else begin
      if (wr_en) begin
        data_array[addr] <= wr_data;
      end
    end
  end

  always_ff @(posedge clk or negedge rst) begin
    if (rst== RESET_STATE) begin
      rd_data_delayed <= 'x;
    end else begin
      if (ch_en) begin
        rd_data_delayed[0]<= rd_data_based_on_en;
        for (integer i=1; i< RD_LATENCY; i=i+1) begin
          rd_data_delayed[i] <= rd_data_delayed[i-1];
        end
      end
    end
  end
  
  assign rd_data_based_on_en= wr_en ? 'x : data_array[addr];

  assign rd_data = rd_data_delayed[RD_LATENCY - 1];
endmodule



module my_memory #(
  parameter DATA_L= 8,
  parameter ADDR_L= 8,
  parameter N_BANKS= 8,
  parameter INIT_FILE_PATH_PREFIX= "dummy_prefix",
  parameter RD_LATENCY
)
(
  input clk,
  input rst,

  input [ADDR_L - 1 : 0] addr,
  input [N_BANKS -1 : 0] [DATA_L - 1 : 0] wr_data,
  input [N_BANKS - 1 : 0]                 wr_en,
  input [N_BANKS - 1 : 0]                 rd_en, 
  output [N_BANKS - 1 : 0] [DATA_L - 1 : 0] rd_data
);

  logic [N_BANKS - 1 : 0] ch_en;
  assign ch_en = wr_en | rd_en;

  generate
    genvar mem_i;
    for (mem_i=0; mem_i< N_BANKS ; mem_i=mem_i+1) begin: mem_loop
      /* fname.hextoa(mem_i); */
      sp_mem_model #(
        .DATA_L (DATA_L),
        .ADDR_L (ADDR_L),
        .RD_LATENCY (RD_LATENCY)
      ) SP_MEM_MODEL_INS
      (
        .clk     (clk     ),
        .rst     (rst     ),

        .slp     (1'b0),
        .sd      (1'b0),
                          
        .wr_en   (wr_en  [mem_i]),
        .addr    (addr   ),
        .wr_data (wr_data[mem_i]),
        .rd_data (rd_data[mem_i]),
        .ch_en   (ch_en  [mem_i])
      ); 
    end
  endgenerate
endmodule

`endif


`endif //MEM_MODEL_DEF
