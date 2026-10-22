# SPDX-License-Identifier: Apache-2.0
# Tiny Tapeout demo board (MicroPython, ttboard SDK): classify 8 x 8 images with the digit classifier tile.
#   import classify; classify.main()
# Each image is 8 bytes, one row per byte, pixel 0 (left) in bit 0, 1 = ink.

IN_VALID, ABORT, SCORES = 1, 2, 4
# (test set index, label, bytes): real images from the scikit-learn digits test split, already in chip order
SAMPLES = [
    (24, 0, [0x0C, 0x1C, 0x36, 0x24, 0x20, 0x24, 0x34, 0x1C]),
    (16, 1, [0x08, 0x0C, 0x08, 0x1C, 0x10, 0x10, 0x30, 0x78]),
    (6, 2, [0x0C, 0x1C, 0x16, 0x10, 0x10, 0x08, 0x38, 0x3C]),
    (2, 3, [0x38, 0x3C, 0x30, 0x30, 0x60, 0x60, 0x28, 0x18]),
    (0, 7, [0x38, 0x2C, 0x24, 0x30, 0x78, 0x10, 0x18, 0x08]),
]


def classify(tt, rows):
    tt.uio_oe_pico.value = IN_VALID | ABORT | SCORES
    before = (int(tt.uio_out.value) >> 3) & 1
    for row in rows:
        tt.ui_in.value = row
        tt.uio_in.value = IN_VALID
        tt.clock_project_once()
    tt.uio_in.value = 0
    for _ in range(8):   # the digit is ready four clocks after the last row
        tt.clock_project_once()
        status = int(tt.uio_out.value)
        if (status >> 3) & 1 != before:
            break
    else:
        raise RuntimeError('no new result')
    digit = status >> 4
    tt.uio_in.value = SCORES
    scores = []
    for c in range(10):
        tt.ui_in.value = c
        raw = int(tt.uo_out.value)
        scores.append(raw - 256 if raw > 127 else raw)
    tt.uio_in.value = 0
    return digit, scores


def main():
    from ttboard.demoboard import DemoBoard
    tt = DemoBoard.get()
    tt.shuttle.tt_um_joonsense_digit_classifier.enable()
    tt.reset_project(True)
    tt.clock_project_once()
    tt.reset_project(False)
    for index, label, rows in SAMPLES:
        digit, scores = classify(tt, rows)
        print('test image %d, label %d -> chip %d %s' % (index, label, digit, scores))
