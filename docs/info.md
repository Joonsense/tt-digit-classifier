## How it works

A complete handwritten digit classifier (one 64 to 10 linear layer trained on scikit-learn digits) implemented directly
in logic. The 640 weights (4 bit) and 10 biases are constants inside the design, so nothing has to be loaded.

An image is 8 x 8 pixels at 1 bit each: 8 bytes, one per clock, pixel 0 of each row in bit 0. Every clock the chip adds
that row's contribution to all 10 class scores at once (80 multiply accumulates per clock). Three clocks after the 8th byte
(inputs and per row sums are registered) the highest score wins (ties go to the lower digit), the digit appears on the 7 segment display and on `uio[7:4]`, and
`DONE` toggles. Images can follow back to back: one result every 8 clocks, 6.25 million images per second at 50 MHz.

Test set accuracy is 91.39% (360 images). Every score the chip computes matches the Python integer model bit for bit.

## How to test

Hold `uio[0]` (IN_VALID) high and put one image row per clock on `ui_in`, 8 rows per image. Watch the 7 segment display,
or read `uio[7:4]` after `uio[3]` (DONE) toggles. `uio[1]` (ABORT) restarts the current image.

With `uio[2]` (SCORES) high, `uo_out` shows the signed 8 bit score of class `ui_in[3:0]` for the last image instead of the
display, to check the arithmetic.

`host/classify.py` does this from the demo board's MicroPython (manual clocking), with a few built in test images.

## External hardware

None. The demo board's 7 segment display shows the digit.
