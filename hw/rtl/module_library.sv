
`ifndef MODULE_LIB_DEF
  `define MODULE_LIB_DEF


///////////////////////////////
////// Floating Adder  ////////
///////////////////////////////
module flt_add #(
  parameter EXP_W = 8,
  parameter MAN_W = 23
)(
  input [EXP_W+MAN_W-1:0] in1,
  input [EXP_W+MAN_W-1:0] in2,
  output reg [EXP_W+MAN_W-1:0] out
 );

  /************************************************
   ************** INTERNAL VARIABLES **************
   ************************************************/
  logic unsigned [MAN_W-1:0] man1;
  logic unsigned [EXP_W-1:0] exp1;
  logic unsigned [MAN_W-1:0] man2;
  logic unsigned [EXP_W-1:0] exp2;
  logic unsigned [MAN_W-1:0] manO;
  logic unsigned [EXP_W-1:0] expO;


  logic signed [EXP_W:0] exp1S;
  logic signed [EXP_W:0] exp2S;
  logic signed [EXP_W:0] dExpS;
  logic signed [EXP_W:0] dExp;
  logic unsigned [EXP_W:0] shiftExp;

  logic unsigned [EXP_W-1:0] expSel;
  logic unsigned [EXP_W:0] expInc;
  logic unsigned increment;
  logic unsigned [EXP_W:0] expPreOverflow;

  logic unsigned [MAN_W:0] man1E; // MSB Extended man1
  logic unsigned [MAN_W:0] man2E; // MSB Extended man2
  logic unsigned [MAN_W+1:0] manPreShift;
  logic unsigned [MAN_W+1:0] manShift;
  logic unsigned [MAN_W+1:0] manNormal;
  logic unsigned [MAN_W+2:0] man_raw;
  logic unsigned [MAN_W+1:0] manPreRound;
  logic unsigned [MAN_W+1:0] manRound;
  logic unsigned [MAN_W:0] manPreOverflow;

  /************************************************
   ***************** FUNCTIONALITY ****************
   ************************************************/
  always_comb begin
    // Exponent difference calculation
    man1 = in1[MAN_W-1:0];
    exp1= in1[EXP_W + MAN_W-1: MAN_W];
    man2 = in2[MAN_W-1:0];
    exp2= in2[EXP_W + MAN_W-1: MAN_W];

    exp1S = $signed({1'b0,exp1});
    exp2S = -$signed({1'b0,exp2});
    dExpS = exp1S + exp2S;
    if (dExpS[EXP_W]==1)
      shiftExp = $unsigned(-dExpS);
    else
      shiftExp = $unsigned(dExpS);
    
    // Mantissa calculation
    if (in1 == 0) begin // ** SPECIAL SUPPORT FOR 0 **
      man1E = {1'b0, man1};
    end else begin
      man1E = {1'b1, man1};
    end
    
    if (in2 == 0) begin // ** SPECIAL SUPPORT FOR 0 **
      man2E = {1'b0, man2};
    end else begin
      man2E = {1'b1, man2};
    end

    if (dExpS[EXP_W]==1) begin
      manPreShift = {man1E,1'b0};
      manNormal = {man2E,1'b0};
    end else begin
      manPreShift = {man2E,1'b0};
      manNormal = {man1E,1'b0};
    end
    manShift = manPreShift>>>shiftExp[EXP_W-1:0];
    man_raw = manShift + manNormal;
    if (man_raw[MAN_W+2]) begin
      manPreRound = man_raw[MAN_W+2:1];
    end else begin
      manPreRound = man_raw[MAN_W+1:0]; 
    end

    manRound = manPreRound[MAN_W+1:1] + manPreRound[0]; // Rounding
    
    if (manRound[MAN_W+1]) begin
      manPreOverflow = manRound[MAN_W+1:1];
    end else begin
      manPreOverflow = manRound[MAN_W:0];
    end

    // Exponent calculation
//    if (exp1<exp2) begin
    if (dExpS[EXP_W]) begin
      expSel = exp2;
    end else begin
      expSel = exp1;
    end
    
    /*expInc = expSel+1;
    // Steven's condition:
    // increment = ((expSel=='b0 && man_raw[MAN_W+1])||(expSel!='b0&&(man_raw[MAN_W+2]==1||manRound[MAN_W+1]==1)));
    increment = man_raw[MAN_W+2]==1 || manRound[MAN_W+1]==1;
    if (increment)
      expPreOverflow = expInc;
    else
      expPreOverflow = {1'b0,expSel};
    */  
    expPreOverflow = {1'b0,expSel} + (man_raw[MAN_W+2]==1 || manRound[MAN_W+1]==1);

    // Overflow detection
    if (expPreOverflow[EXP_W]) begin
      expO = {EXP_W{1'b1}};
      manO = {MAN_W{1'b1}};
    end else begin
      expO = expPreOverflow[EXP_W-1:0];
      manO = manPreOverflow[0+:MAN_W];
    end
    out= {expO, manO};
  end
endmodule

///////////////////////////////
//////    Float Mult    / ////
///////////////////////////////
module flt_mul # (parameter EXP_L= 8, MNT_L=23)( //EXP_L should not be less than 3
    input [EXP_L+MNT_L-1:0] in1, 
    input [EXP_L+MNT_L-1:0] in2, 
    output [EXP_L+MNT_L-1:0] out 
  );
logic unsigned [MNT_L:0] sigX ;
logic unsigned [MNT_L:0] sigY ;
logic unsigned [2*MNT_L+1:0] sigProd ;
logic unsigned [2*MNT_L+1:0] sigProdExt_pre;
logic unsigned [2*MNT_L+1:0] sigProdExt ;

logic unsigned [EXP_L-1:0] expX ;
logic unsigned [EXP_L-1:0] expY ;
logic unsigned [EXP_L:0] expSumPreSub ;
logic unsigned [EXP_L-2:0] bias_dummy;
logic unsigned [EXP_L:0] bias ;
logic unsigned [EXP_L:0] expSum ;
logic unsigned [EXP_L:0] expPostNorm;
logic unsigned [EXP_L:0] expPostNorm_fin;

logic unsigned [EXP_L+MNT_L:0] expSig; 
logic unsigned [EXP_L+MNT_L:0] expSigPostRound;

logic unsigned norm;
logic unsigned sticky ;
logic unsigned guard  ;
logic unsigned round  ;
 
  assign expX = in1[EXP_L+MNT_L-1 : MNT_L];
  assign expY = in2[EXP_L+MNT_L-1 : MNT_L];
  assign sigX = {1'b1 , in1[MNT_L-1:0]};
  assign sigY = {1'b1 , in2[MNT_L-1:0]};
  
  assign sigProd= sigX*sigY; 
  assign norm= sigProd[2*MNT_L+1];

  assign expSumPreSub = {1'b0, expX} + {1'b0, expY};
  assign bias_dummy= '1;  // All ones 
  assign bias = {2'b00, bias_dummy}; //CONV_STD_LOGIC_VECTOR(127,10);
  assign expSum = expSumPreSub + norm - bias;
  assign expPostNorm= expSum;
  
  // significand normalization shift
  always @(*)
  begin
    if (norm == 1) begin
      sigProdExt_pre = {sigProd[2*MNT_L : 0], 1'b0};
    end else begin
      sigProdExt_pre = {sigProd[2*MNT_L-1 : 0], 2'b00};
    end
  end
  
  always @(*)
  begin
    if ((bias > (expSumPreSub+norm)) || (in1==0) || (in2==0)) begin // Underflow check and check if any of the operand is zero ** SPECIAL SUPPORT FOR 0 **
      expPostNorm_fin= '0; // All zeroes
      sigProdExt= '0;
    end else if (expPostNorm[EXP_L]) begin // Overflow
      expPostNorm_fin= '1; // All ones
      sigProdExt= { {MNT_L{1'b1}} , {2+MNT_L{1'b0}} }; // First MNT_L bits toward MSB are 1, rest 0s
    end else begin
      expPostNorm_fin= expPostNorm;
      sigProdExt= sigProdExt_pre;
    end
  end
  
  assign expSig = {expPostNorm_fin , sigProdExt[2*MNT_L+1: MNT_L+2]};
  assign sticky = sigProdExt[MNT_L+1];
  
  /*
  always @(*) begin
    if (sigProdExt[MNT_L: 0]== 0) begin
      guard = 0;
    end else begin
      guard = 1;
    end
  end
  
  assign round = sticky & ( (guard & ~(sigProdExt[MNT_L+2])) | (sigProdExt[MNT_L+2] ))  ; // sticky bit is the MSB of the bits being stipped off. guard thing tries to round 0.5 to 0 instead of 1
  assign expSigPostRound= round+ expSig; // This will not overflow Mantissa because there will always be a zero in Mantissa to absorb the propogation of 1, hence no overflow check required for this
  */
  assign expSigPostRound= sticky+ expSig; // This will not overflow Mantissa because there will always be a zero in Mantissa to absorb the propogation of 1, hence no overflow check required for this

  assign out = expSigPostRound[EXP_L+MNT_L-1:0];
  
endmodule


///////////////////////////////
////// Shift Reg       ////////
///////////////////////////////
module shift_reg #(parameter DEPTH=1, WORD_LEN=32)(
  input clk, rst,
  input [WORD_LEN-1:0] in,
  output [WORD_LEN-1:0] out
  );

  reg [DEPTH-1 : 0][WORD_LEN-1 : 0] shift ;
  integer i;
  assign out= shift[DEPTH-1];
  
  always @ ( posedge clk)
  begin: SHIFT_REG
    if (rst) begin
      for (i=0; i<DEPTH; i=i+1) begin
        shift[i] <= 0;
      end
    end else begin
      shift[0]<= in;
      for (i=1; i<DEPTH; i= i+1) begin
        shift[i] <= shift[i-1];
      end
    end
  end
endmodule

///////////////////////////////
////// Shift Reg 2D    ////////
///////////////////////////////
module shift_reg_2d #(parameter DEPTH=1, INNER_DIM=32, OUTER_DIM= 8)(
  input clk, rst,
  input [OUTER_DIM - 1: 0][INNER_DIM-1:0] in,
  output [OUTER_DIM - 1: 0][INNER_DIM-1:0] out
  );

  reg [DEPTH-1 : 0][OUTER_DIM - 1: 0][INNER_DIM-1:0] shift;
  integer i;
  assign out= shift[DEPTH-1];
  
  always @ ( posedge clk)
  begin: SHIFT_REG_2D
    if (rst) begin
      for (i=0; i<DEPTH; i=i+1) begin
        for (integer j=0; j<OUTER_DIM; j=j+1) begin
            shift[i][j] <= '0;
        end
      end
    end else begin
      shift[0]<= in;
      for (i=1; i<DEPTH; i= i+1) begin
        shift[i] <= shift[i-1];
      end
    end
  end
endmodule

////////////////////////////////////////
////// Parameterizable Crossbar ////////
////////////////////////////////////////
module crossbar #(parameter WORD_LEN=32,  IN_PORTS= 32, OUT_PORTS=32) (
  input [IN_PORTS-1: 0][WORD_LEN-1 : 0] input_words ,
  input [OUT_PORTS-1: 0][$clog2(IN_PORTS) - 1 : 0] sel ,
  output [OUT_PORTS-1: 0][WORD_LEN-1 : 0] output_words 
  );
  
  logic [OUT_PORTS-1: 0][WORD_LEN-1 : 0] output_words_d ;
  
  always_comb
  begin
    for (int i=0 ; i < OUT_PORTS; i++) begin
      output_words_d[i] <= input_words[sel[i]]; 
    end
  end

  assign output_words = output_words_d;
endmodule  

`endif //MODULE_LIB_DEF
