// Content: SV modules for handling variable length instructions 
// 
// Author: Nimish Shah
// Date: 18/05/2019
// AC processor

`include "common_pkg.sv"

`include "basic_blocks.sv"


////////////////////////////////////////
///////// Decoder to get instr len from instr bits
////////////////////////////////////////
module instr_len_decd #(parameter WORD_LEN= 32) (input [0: WORD_LEN-1] first_instr_word, output [$clog2(instr_pkg::max_instr_len)-1:0] curr_instr_len);
  

  localparam idx_len= $clog2(instr_pkg::n_instr_len);

  initial assert (instr_pkg::max_instr_len == instr_pkg::instr_len[instr_pkg::n_instr_len-1]) else $fatal("Update max_instr_len according to data in instr_len");

  logic [idx_len-1:0] len_code;
  assign len_code= first_instr_word[0:idx_len-1];
  assign curr_instr_len = instr_pkg::instr_len[len_code]; 

endmodule

///////////////////////////////////////
////// Barrel shifter with output of same length as input 
///////////////////////////////////////
/*
  Shifts only by predefined values and not any possible value
  parameters:
    WORD_LEN : the size of each input words
    N_IN_WORD : Number of input words
    N_OUT_WORD : Number of output words. Concretely, N_OUT_WORD < N_IN_WORD + max_shift_val. But here max_shift_val will be at max == N_IN_WORD, so we will assert N_OUT_WORD < 2*N_IN_WORD
    N_SHIFTS : Number of possible shifts. Should be <= N_IN_WORD

*/

module barrel_shift # ( parameter WORD_LEN= 32, N_IN_WORD=32, N_OUT_WORD=32, N_SHIFTS= 4) (
  input [0: N_IN_WORD-1][0:WORD_LEN-1] in,
  input [$clog2(N_SHIFTS)-1:0] shift_code,
  input shift_en,
  output [0: N_OUT_WORD-1][0:WORD_LEN-1] out
  );
  localparam shift_0=  instr_pkg::instr_len[0]; 
  localparam shift_1=  instr_pkg::instr_len[1];
  localparam shift_2=  instr_pkg::instr_len[2];

  initial 
  begin
    assert (N_OUT_WORD <= 2*N_IN_WORD) else $fatal("N_OUT_WORD should not be more than 2*N_IN_WORD");
    assert (N_SHIFTS <= N_IN_WORD) else $fatal("Do not want to shift more than the original length of the input");
  end

  integer i;
  logic [0: 2*N_IN_WORD-1][0:WORD_LEN-1] out_pre; // NOTE: length is 2*N_IN_WORD and NOT N_OUT_WORD
   


  always_comb
  begin
    // Initialize out_pre= 0
    for (i=0;i<2*N_IN_WORD;i=i+1) begin
      out_pre[i]= 0;
    end
    
    if (shift_en) begin
     if (shift_code == 0) begin
       out_pre[shift_0: shift_0+N_IN_WORD-1 ]= in;
     end else if (shift_code == 1) begin
       out_pre[ shift_1: shift_1+N_IN_WORD-1]= in;
     end else if (shift_code == 2) begin
       out_pre[ shift_2: shift_2+N_IN_WORD-1]= in;
     end else begin
       $fatal(1, "Not defined yet!");
       out_pre[ 0: N_IN_WORD-1 ]= in;
     end
    end else begin
      out_pre[ 0: N_IN_WORD-1 ]= in;
    end
  end

  assign out = out_pre[0:N_OUT_WORD-1];
endmodule: barrel_shift 

interface instr_reader_if #(parameter WORD_LEN= 32, INSTR_FIFO_WIDTH=16) (input clk, input rst);

  // fifo --> reader --> decoder

  logic [0: INSTR_FIFO_WIDTH-1][0:WORD_LEN-1] instr_from_fifo ;
  logic reader_2_fifo_rdy;
  logic fifo_2_reader_vld;
  logic [0: INSTR_FIFO_WIDTH-1][0:WORD_LEN-1]instr_2_decd;
  logic reader_2_decd_vld;
  logic decd_2_reader_rdy;

  modport fifo (input reader_2_fifo_rdy, output fifo_2_reader_vld, instr_from_fifo);
  modport reader (input clk,rst,instr_from_fifo, fifo_2_reader_vld, decd_2_reader_rdy, output reader_2_fifo_rdy, reader_2_decd_vld, instr_2_decd);
  modport decd (input instr_2_decd, reader_2_decd_vld, output decd_2_reader_rdy);

endinterface: instr_reader_if

module instr_reader #(parameter WORD_LEN= 32, INSTR_FIFO_WIDTH=16) ( instr_reader_if.reader i);
  // Declare variables
  
  //NOTE: DO not  delete this
  //localparam INSTR_FIFO_WIDTH= $bits(i.instr_from_fifo); // NOTE: Synthesis tool doesn't allow directly accessing interface parameters
  
  localparam TOTAL_REG_W= 2*(INSTR_FIFO_WIDTH); // This has to be 2x, it cannot be directly made 3x or anything else
  localparam LENGTH_TRACK_BITS= $clog2(TOTAL_REG_W);
  localparam MAX_INSTR_LEN= instr_pkg::max_instr_len;
  localparam N_INSTR_LEN= instr_pkg::n_instr_len;

  reg  [0 : TOTAL_REG_W-1][0:WORD_LEN-1] instr_reg_q; 
  logic [0: TOTAL_REG_W-1][0:WORD_LEN-1] instr_reg_d ;
  logic [0: TOTAL_REG_W-1][0:WORD_LEN-1] instr_reg_shifted ;

  logic [0: TOTAL_REG_W-1] mux_sel;
  logic [LENGTH_TRACK_BITS:0] first_nonvalid_word_q;
  logic [LENGTH_TRACK_BITS:0] first_nonvalid_word_d;
  logic [LENGTH_TRACK_BITS-1:0] fifo_word_shift_cnt;
  
  logic [$clog2(MAX_INSTR_LEN)-1:0] curr_instr_len;
  logic [0: TOTAL_REG_W-1][0:WORD_LEN-1] fifo_word_shifted;
  
  logic reader_2_decd_vld, reader_2_fifo_rdy;
  logic send_2_decd_en, pop_fifo_en;
  
  logic [0:$clog2(N_INSTR_LEN)-1] shift_code;
  
  always_comb begin
    if (send_2_decd_en) begin
      if (pop_fifo_en) begin
        first_nonvalid_word_d = first_nonvalid_word_q - curr_instr_len + INSTR_FIFO_WIDTH;
      end else begin
        first_nonvalid_word_d = first_nonvalid_word_q - curr_instr_len;
      end
    end else begin
      if (pop_fifo_en) begin
        first_nonvalid_word_d = first_nonvalid_word_q + INSTR_FIFO_WIDTH;
      end else begin
        first_nonvalid_word_d = first_nonvalid_word_q;
      end
    end
    assert (first_nonvalid_word_q < TOTAL_REG_W + 1) else $fatal("first_nonvalid_word_q not set properly");
  end
  register #(.L(LENGTH_TRACK_BITS+1)) REG_first_nonvalid_word(.clk(i.clk), .rst(i.rst), .in(first_nonvalid_word_d), .out(first_nonvalid_word_q));

  // Register to store 2 instruction of maximum size
  // Choose between shifted version and new instruction based on select signal
  always_ff @(posedge i.clk) begin
    if (i.rst) begin
      for (integer itr_i=0; itr_i< TOTAL_REG_W; itr_i++) begin
        instr_reg_q[itr_i] <= 0;
      end
    end else begin
      instr_reg_q <= instr_reg_d;
    end
  end
  
  always_comb begin
    for (integer itr_j=0; itr_j<TOTAL_REG_W; itr_j++) begin
      if (mux_sel[itr_j]) begin // if 1 choose new word from fifo
        instr_reg_d[itr_j] = fifo_word_shifted[itr_j];
      end else begin // if 0 choose either the original or the shifted original
        if (send_2_decd_en) begin
          instr_reg_d[itr_j] = instr_reg_shifted[itr_j];
        end else begin
          instr_reg_d[itr_j] = instr_reg_q[itr_j];
        end
      end
    end
  end
  
  // MUX select
  always_comb begin
    mux_sel= 0; // All 1s
    if (pop_fifo_en) begin
      if (send_2_decd_en) begin
        mux_sel[(first_nonvalid_word_q-curr_instr_len) +: INSTR_FIFO_WIDTH] = '1; // All 1s
        assert (first_nonvalid_word_q - curr_instr_len + INSTR_FIFO_WIDTH <= TOTAL_REG_W) else $fatal("first_nonvalid_word_q+ INSTR_FIFO_WIDTH - curr_instr_len should be less than equal to TOTAL_REG_W in this condition");

      end else begin
        mux_sel[first_nonvalid_word_q +: INSTR_FIFO_WIDTH] = '1; // All 1s
        assert (first_nonvalid_word_q + INSTR_FIFO_WIDTH <= TOTAL_REG_W) else $fatal("first_nonvalid_word_q+ INSTR_FIFO_WIDTH should be less than equal to TOTAL_REG_W when pop_fifo_en is high and send_2_decd_en is low");
      end
    end else begin 
      mux_sel= 0;
    end
  end

  assign shift_code= instr_reg_q[0][0: $clog2(N_INSTR_LEN)-1];
  // Shift words in instr_reg_q
  // TODO: Generate shift code by decoding the instruction opcode and figuring out it's length, and not based on some bits in the opcode
  // TODO: Verify that endian-ness does not create problem for shift_code 
  barrel_shift #(.WORD_LEN(WORD_LEN), .N_IN_WORD(TOTAL_REG_W), .N_OUT_WORD(TOTAL_REG_W), .N_SHIFTS(N_INSTR_LEN)) shifter (.in(instr_reg_q), .out(instr_reg_shifted), .shift_code(shift_code), .shift_en(send_2_decd_en));
  

  assert property (@(posedge i.clk) (shift_code < instr_pkg::n_instr_len)) else $fatal("shift code cannot have more values than maximum number of possible instructions");
  
  always_comb begin
    fifo_word_shift_cnt = 0;
    if (send_2_decd_en) begin // Current words in the register will be shifted out to send instructions to decd
      assert (curr_instr_len - first_nonvalid_word_q + INSTR_FIFO_WIDTH <= INSTR_FIFO_WIDTH);
      fifo_word_shift_cnt = curr_instr_len - first_nonvalid_word_q + INSTR_FIFO_WIDTH; 
    end else begin
      assert (- first_nonvalid_word_q + INSTR_FIFO_WIDTH <= INSTR_FIFO_WIDTH);
      fifo_word_shift_cnt= - first_nonvalid_word_q + INSTR_FIFO_WIDTH;
    end
  end
  
  assign fifo_word_shifted = i.instr_from_fifo << fifo_word_shift_cnt;

  instr_len_decd #(.WORD_LEN(WORD_LEN)) instr_2_decd_INST ( .first_instr_word(instr_reg_q[0]), .curr_instr_len(curr_instr_len));

  // Flow control -------------------------------
  always_comb begin
    if ( (send_2_decd_en && (first_nonvalid_word_q-curr_instr_len < INSTR_FIFO_WIDTH)) || (first_nonvalid_word_q < INSTR_FIFO_WIDTH) ) begin // There is space to pop from FIFO
      reader_2_fifo_rdy = 1;
    end else begin
      reader_2_fifo_rdy= 0;
    end
  end

  always_comb begin
    if (curr_instr_len < first_nonvalid_word_q) begin // There are some valid words
      reader_2_decd_vld = 1;
    end else begin
      reader_2_decd_vld = 0;
    end
  end

  assign send_2_decd_en = reader_2_decd_vld & i.decd_2_reader_rdy;
  assign pop_fifo_en= reader_2_fifo_rdy & i.fifo_2_reader_vld;
  
  // Assign outputs -------------------------
  assign i.reader_2_fifo_rdy = reader_2_fifo_rdy;
  assign i.reader_2_decd_vld = reader_2_decd_vld;
  assign i.instr_2_decd = instr_reg_q[0: INSTR_FIFO_WIDTH-1];

endmodule: instr_reader

module try(input clk,rst, fifo_2_reader_vld, decd_2_reader_rdy, [0:15][31:0] instr_from_fifo, [63:0]mux_sel, output reader_2_fifo_rdy, reader_2_decd_vld, [0:15][31:0] instr_2_decd);
  
  instr_reader_if i(clk,rst);
  instr_reader instr_reader_0(.i(i.reader));
  
  reg [0:15][31:0] instr_from_fifo_q;
  
  always_ff @(posedge clk) begin
    instr_from_fifo_q <= instr_from_fifo;
  end

  assign reader_2_fifo_rdy= i.reader_2_fifo_rdy;
  assign reader_2_decd_vld= i.reader_2_decd_vld;
  assign instr_2_decd= i.instr_2_decd;
  assign i.instr_from_fifo= instr_from_fifo_q;
  assign i.decd_2_reader_rdy= decd_2_reader_rdy;
  assign i.fifo_2_reader_vld= fifo_2_reader_vld;
endmodule:try  
