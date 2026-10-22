/*
 * A whole handwritten digit classifier (64 to 10 linear layer, scikit-learn digits) as hardware.
 * The 640 INT4 weights are constants in the logic. Send an 8 x 8 image as 8 bytes of 1 bit pixels,
 * one byte per clock; the digit appears on the 7 segment display four clocks after the last byte.
 * SPDX-License-Identifier: Apache-2.0
 *
 *   ui_in[7:0]  PIXELS     8 pixels of one image row, pixel 0 (left) in bit 0
 *   uio[0] in   IN_VALID   take ui_in on this rising edge (8 bytes per image, back to back is fine)
 *   uio[1] in   ABORT      restart the current image
 *   uio[2] in   SCORES     0: uo_out drives the 7 segment display. 1: uo_out = score of class ui_in[3:0] for the last
 *                          image (signed 8 bit; held until the next image starts; class 10 to 15 read 0)
 *   uio[3] out  DONE       toggles each time a new digit is ready
 *   uio[7:4]    DIGIT      the predicted digit, binary
 *   uo_out      7 segment  a..g on uo_out[0..6] (blank until the first result), uo_out[7] = DONE
 */
`default_nettype none

module tt_um_joonsense_digit_classifier (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
  wire [3:0] digit;
  wire done_toggle, have_result, busy;
  wire [79:0] scores;

  // Inputs are registered once: each pixel bit feeds the tables of all 10 classes, and with the 4 ns input delay of the
  // Tiny Tapeout pads that path missed 50 MHz at the slow corner after routing (-0.65 ns). One clock of latency, same rate.
  reg [7:0] pixels;
  reg valid, restart;
  always @(posedge clk) begin
    pixels <= ui_in;
    valid <= rst_n && uio_in[0];
    restart <= rst_n && uio_in[1];
  end

  digit_core core (.clk(clk), .rst(!rst_n), .in_valid(valid), .in_data(pixels), .abort(restart),
                   .digit(digit), .done_toggle(done_toggle), .have_result(have_result), .busy(busy), .scores(scores));
  wire [7:0] score = ui_in[3:0] < 4'd10 ? scores[ui_in[3:0] * 8 +: 8] : 8'd0;

  reg [6:0] seg;
  always @(*) begin
    case (digit)
      4'd0: seg = 7'b0111111;
      4'd1: seg = 7'b0000110;
      4'd2: seg = 7'b1011011;
      4'd3: seg = 7'b1001111;
      4'd4: seg = 7'b1100110;
      4'd5: seg = 7'b1101101;
      4'd6: seg = 7'b1111101;
      4'd7: seg = 7'b0000111;
      4'd8: seg = 7'b1111111;
      4'd9: seg = 7'b1101111;
      default: seg = 7'b1000000;
    endcase
  end

  assign uo_out = uio_in[2] ? score : {done_toggle, have_result ? seg : 7'b0000000};
  assign uio_out = {digit, done_toggle, 3'b000};
  assign uio_oe = 8'b1111_1000;

  wire _unused = &{ena, uio_in[7:3], ui_in[7:4], busy, 1'b0};
endmodule
