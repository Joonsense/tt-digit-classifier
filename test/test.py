# SPDX-License-Identifier: Apache-2.0
# Runs on the RTL and, in the Tiny Tapeout GL job, on the post layout gate netlist.
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer

DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'digits_images.json')))
SEG = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F]
IN_VALID, ABORT, SCORES = 1, 2, 4


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)


def state(dut):
    uio, uo = dut.uio_out.value.to_unsigned(), dut.uo_out.value.to_unsigned()
    return (uio >> 3) & 1, uio >> 4, uo & 0x7F


async def watch(dut, results):
    last = 0
    cycle = 0
    while True:
        await FallingEdge(dut.clk)
        cycle += 1
        await Timer(1, unit="ns")
        done, digit, seg = state(dut)
        if done != last:
            last = done
            results.append((cycle, digit, seg))


@cocotb.test()
async def test_all_test_images_back_to_back(dut):
    """All 360 test images, one byte per clock with no gaps: every digit must match the Python integer model."""
    await start(dut)
    assert state(dut)[2] == 0, "display must be blank before the first result"
    results = []
    cocotb.start_soon(watch(dut, results))
    for image in DATA['images']:
        for byte in image:
            dut.ui_in.value = byte
            dut.uio_in.value = IN_VALID
            await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 6)
    got = [digit for _, digit, _ in results]
    assert len(got) == len(DATA['expected']), f"{len(got)} results"
    mismatches = sum(g != w for g, w in zip(got, DATA['expected']))
    assert mismatches == 0, f"{mismatches} digits differ from the model"
    assert all(seg == SEG[digit] for _, digit, seg in results), "7 segment pattern does not match the digit"
    gaps = {b[0] - a[0] for a, b in zip(results, results[1:])}
    assert gaps == {8}, f"result interval {gaps}, want 8 clocks"
    correct = sum(g == label for g, label in zip(got, DATA['labels']))
    dut._log.info(f"360 images, 0 mismatches, accuracy {correct / 360:.4f}, one image every 8 clocks (80 MAC per clock)")


@cocotb.test()
async def test_abort_and_gaps(dut):
    """ABORT in the middle of an image restarts it; IN_VALID gaps inside an image do not change the answer."""
    await start(dut)
    results = []
    cocotb.start_soon(watch(dut, results))
    image, want = DATA['images'][7], DATA['expected'][7]
    for byte in DATA['images'][3][:5]:
        dut.ui_in.value = byte
        dut.uio_in.value = IN_VALID
        await FallingEdge(dut.clk)
    dut.uio_in.value = ABORT
    await FallingEdge(dut.clk)
    for byte in image:
        dut.uio_in.value = 0
        await ClockCycles(dut.clk, 2)
        await FallingEdge(dut.clk)
        dut.ui_in.value = byte
        dut.uio_in.value = IN_VALID
        await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 4)
    assert [d for _, d, _ in results] == [want], f"{results} != [{want}]"


@cocotb.test()
async def test_all_scores_bit_exact(dut):
    """For every test image, read all 10 class scores through the SCORES mode: each must equal bias + weights . pixels."""
    await start(dut)
    bad = 0
    for image, want in zip(DATA['images'], DATA['scores']):
        before = state(dut)[0]
        for byte in image:
            dut.ui_in.value = byte
            dut.uio_in.value = IN_VALID
            await FallingEdge(dut.clk)
        dut.uio_in.value = 0
        for _ in range(10):   # wait for DONE: the scores are final and held from then until the next image starts
            await FallingEdge(dut.clk)
            await Timer(1, unit="ns")
            if state(dut)[0] != before:
                break
        else:
            assert False, "no result"
        dut.uio_in.value = SCORES
        for c in range(10):
            dut.ui_in.value = c
            await Timer(1, unit="ns")
            raw = dut.uo_out.value.to_unsigned()
            bad += (raw - 256 if raw > 127 else raw) != want[c]
        dut.uio_in.value = 0
        await FallingEdge(dut.clk)   # finish reading before the next image's first byte
    dut.uio_in.value = 0
    assert bad == 0, f"{bad} of 3600 scores differ"
    dut._log.info("3600 scores bit identical to Python")
