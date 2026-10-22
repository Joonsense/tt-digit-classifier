"""S8: 숫자 분류기 한 층을 가중치 상수로 굳힌 칩. 픽셀 바이트만 받아 10 클래스 점수를 병렬 누산하고 argmax 를 낸다.

입력: 이미지 하나 = 64 픽셀 × bits → 64·bits/8 바이트, 클록당 1바이트 (바이트 안 첫 픽셀이 하위 비트).
점수: acc[c] = bias[c] + Σ W[c][i]·x[i] (첫 바이트에서 bias + 그 바이트 항으로 새로 시작). 마지막 바이트 다음 클록에 argmax 를 등록.
동점은 낮은 클래스 (numpy argmax 와 같음). 출력 uo_out: SEG=0 이면 {OUT_VALID, DONE토글, 0, class[3:0]}... (래퍼에서 결정).
"""
import json
from pathlib import Path


def model(path):
    d = json.loads(Path(path).read_text())
    return d['weights'], d['bias'], d['inputs'], d['labels'], d['act_bits']


def rtl(weights, bias, bits, group=4):
    per = 8 // bits
    nbytes = 64 // per
    iw = max(1, (nbytes - 1).bit_length())
    # 점수 범위: |bias| + Σ|W|·max(x) → 부호 폭
    top = (1 << bits) - 1
    span = max(abs(b) + sum(abs(w) * top for w in ws) for ws, b in zip(weights, bias))
    aw = span.bit_length() + 1
    lines = []
    G = group if bits == 1 else 1   # 1비트 픽셀은 G 개를 묶어 (위치, G비트) → 상수 표 하나로 (곱셈·작은 덧셈을 표가 대신)
    for c in range(10):
        terms = []
        for g in range(per // G):
            name = f"t{c}_{g}"
            rows = []
            for k in range(nbytes):
                for m in range(1 << (G * bits)):
                    xs = [(m >> (bits * j)) & ((1 << bits) - 1) for j in range(G)]
                    v = sum(weights[c][k * per + g * G + j] * xs[j] for j in range(G))
                    rows.append(f"      {{{iw}'d{k}, {G * bits}'d{m}}}: {name} = {v};")
            sel = f"in_data[{(g + 1) * G * bits - 1}:{g * G * bits}]"
            lines.append(f"  reg signed [{aw - 1}:0] {name};\n  always @(*) case ({{index, {sel}}})\n" + "\n".join(rows)
                         + f"\n      default: {name} = 0;\n    endcase")
            terms.append(name)
        lines.append(f"  wire signed [{aw - 1}:0] t{c} = {' + '.join(terms)};")
        lines.append(f"  reg signed [{aw - 1}:0] r{c};   // 바이트 하나의 클래스 합을 한 번 등록 (표·덧셈 경로와 누산을 나눈다)")
    accs = '\n'.join(f"  reg signed [{aw - 1}:0] acc{c};" for c in range(10))
    update = '\n'.join(f"        acc{c} <= r_first ? {bias[c]} + r{c} : acc{c} + r{c};" for c in range(10))
    capture = '\n'.join(f"        r{c} <= t{c};" for c in range(10))
    # argmax 트리: 왼쪽이 낮은 클래스, 오른쪽이 '크면' 이긴다 → 동점은 낮은 클래스
    def tree(ids):
        if len(ids) == 1:
            return f"acc{ids[0]}", f"4'd{ids[0]}", []
        mid = (len(ids) + 1) // 2
        lv, li, ld = tree(ids[:mid])
        rv, ri, rd = tree(ids[mid:])
        name = f"m{ids[0]}_{ids[-1]}"
        decl = ld + rd + [f"  wire right_{name} = {rv} > {lv};",
                          f"  wire signed [{aw - 1}:0] v_{name} = right_{name} ? {rv} : {lv};",
                          f"  wire [3:0] i_{name} = right_{name} ? {ri} : {li};"]
        return f"v_{name}", f"i_{name}", decl
    root, best, decls = tree(list(range(10)))
    # 최상위 비교의 값 쪽은 쓰지 않는다 (번호만 쓴다)
    decls = [f"  /* verilator lint_off UNUSEDSIGNAL */\n{d}\n  /* verilator lint_on UNUSEDSIGNAL */" if d.lstrip().startswith(f"wire signed [{aw - 1}:0] {root} ") else d for d in decls]
    return f"""module digit_core(input clk, input rst, input in_valid, input [7:0] in_data, input abort,
                  output reg [3:0] digit, output reg done_toggle, output reg have_result, output busy,
                  output [{10 * aw - 1}:0] scores);
  reg [{iw - 1}:0] index;
  reg pending, r_valid, r_first, r_last;
{accs}
{chr(10).join(lines)}
{chr(10).join(decls)}
  assign busy = index != 0 || r_valid || pending;
  assign scores = {{{', '.join(f'acc{c}' for c in reversed(range(10)))}}};   // 마지막 이미지의 점수 (다음 이미지 첫 바이트까지 유지)
  always @(posedge clk) begin
    if (rst) begin
      index <= 0; pending <= 1'b0; digit <= 4'd0; done_toggle <= 1'b0; have_result <= 1'b0;
      r_valid <= 1'b0; r_first <= 1'b0; r_last <= 1'b0;
{chr(10).join(f'      r{c} <= 0;' for c in range(10))}
{chr(10).join(f'      acc{c} <= 0;' for c in range(10))}
    end else begin
      pending <= 1'b0;
      r_valid <= 1'b0;
      if (pending) begin
        digit <= {best}; done_toggle <= ~done_toggle; have_result <= 1'b1;
      end
      if (r_valid) begin
{update}
        pending <= r_last;
      end
      if (abort) begin
        index <= 0;
      end else if (in_valid) begin
{capture}
        r_valid <= 1'b1; r_first <= index == 0; r_last <= index == {iw}'d{nbytes - 1};
        index <= index == {iw}'d{nbytes - 1} ? 0 : index + 1'b1;
      end
    end
  end
endmodule
""", nbytes, aw


def pack(px, bits):
    per = 8 // bits
    return [sum(px[i + j] << (bits * j) for j in range(per)) for i in range(0, 64, per)]


def scores(weights, bias, px):
    return [b + sum(w * x for w, x in zip(ws, px)) for ws, b in zip(weights, bias)]


def expected(weights, bias, px):
    scores = [b + sum(w * x for w, x in zip(ws, px)) for ws, b in zip(weights, bias)]
    return scores.index(max(scores))


def testbench(images, want, bits, nbytes):
    stream = [b for px in images for b in pack(px, bits)]
    data = ''.join(f'{b:02x}' for b in reversed(stream))
    digits = ''.join(f'{d:x}' for d in reversed(want))
    n = len(images)
    return f"""module tb;
  reg clk = 0, rst = 1, in_valid = 0, abort = 0; reg [7:0] in_data = 0;
  wire [3:0] digit; wire done_toggle, have_result, busy;
  digit_core d(clk, rst, in_valid, in_data, abort, digit, done_toggle, have_result, busy);
  reg [{8 * len(stream) - 1}:0] stream = {8 * len(stream)}'h{data};
  reg [{4 * n - 1}:0] want = {4 * n}'h{digits};
  integer sent, got, first_at, last_at, cycle; reg seen;
  always #5 clk = ~clk;
  always @(posedge clk) cycle <= rst ? 0 : cycle + 1;
  initial seen = 0;
  always @(posedge clk) if (!rst && done_toggle !== seen) begin
    #1; seen = done_toggle;
    if (digit !== want[got*4 +: 4]) $fatal(1, "image %0d: got %0d want %0d", got, digit, want[got*4 +: 4]);
    if (got == 0) first_at = cycle;
    last_at = cycle;
    got = got + 1;
  end
  initial begin
    sent = 0; got = 0; cycle = 0;
    repeat (3) @(negedge clk); rst = 0;
    // 쉼 없이 전부 흘린다 (클록당 1바이트)
    while (sent < {len(stream)}) begin @(negedge clk); in_data = stream[sent*8 +: 8]; in_valid = 1; sent = sent + 1; end
    @(negedge clk); in_valid = 0;
    repeat (5) @(negedge clk);
    if (got != {n}) $fatal(1, "got %0d results", got);
    if ((last_at - first_at) != {nbytes} * ({n} - 1)) $fatal(1, "interval %0d", (last_at - first_at) / ({n} - 1));
    // 이미지 도중 ABORT → 다음 이미지는 처음부터
    @(negedge clk); in_data = 8'hff; in_valid = 1; @(negedge clk); in_valid = 0; abort = 1; @(negedge clk); abort = 0;
    sent = 0; got = 0;
    while (sent < {nbytes}) begin @(negedge clk); in_data = stream[sent*8 +: 8]; in_valid = 1; sent = sent + 1; end
    @(negedge clk); in_valid = 0; repeat (4) @(negedge clk);
    if (got != 1) $fatal(1, "abort recovery");
    $display("PASS {n} images, one every {nbytes} clocks, {80 * 8 // nbytes} MAC per clock, abort recovery");
    $finish;
  end
endmodule
"""
