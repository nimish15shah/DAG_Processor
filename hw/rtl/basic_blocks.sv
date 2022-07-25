module register #(parameter L) ( input clk, rst, [L-1:0]in, output [L-1:0] out);
  reg [L-1:0]out_q;
  always_ff @(posedge clk) begin
    if (rst) begin
      out_q<=0;
    end else begin
      out_q<= in;
    end
  end
  assign out= out_q; 
endmodule: register
