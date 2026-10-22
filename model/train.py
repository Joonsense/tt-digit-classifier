"""S3 응용 시연: 8×8 손글씨 숫자(scikit-learn 동봉, UCI optdigits 테스트셋) 분류 한 층을 INT8 로 양자화한다.

칩은 내적만 한다. 클래스마다 가중치 64개 · 픽셀 64개 = 길이 64 내적 → N=8 칩에서는 CHAIN 8조각.
바이어스는 호스트가 더한다. 이 스크립트는 numpy·scikit-learn 이 있는 파이썬에서 돌리고,
칩 쪽(cocotb·RTL) 이 읽을 digits.json 을 만든다 (칩 쪽은 numpy 없이 돈다).

실행: python3 examples/digits/train.py   (시스템 파이썬, scikit-learn 1.x)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

INT4 = '--int4' in sys.argv    # INT4 칩 판: 가중치 [-8, 7], 픽셀 0..7 (원소가 바이트 하나에 두 개씩 들어간다)
ACT_BITS = int(sys.argv[sys.argv.index('--act-bits') + 1]) if '--act-bits' in sys.argv else 4   # S7: 픽셀 2비트(0..3)·1비트(0..1), 가중치는 INT4
WIDTH = 20                     # 칩 결과 폭 (포화)
LIMIT = (1 << (WIDTH - 1)) - 1
PIXEL_SCALE = 7 / 16 if INT4 else 4   # INT8: 픽셀 0..16 → 0..64 (최대 |내적| 64·127·64 < 2^19), INT4: → 0..7
if ACT_BITS != 4:
    INT4, PIXEL_SCALE = True, ((1 << ACT_BITS) - 1) / 16   # 2비트: 0..3, 1비트: 픽셀 ≥ 8 이면 1
WEIGHT_MAX = 7 if INT4 else 127

digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = LogisticRegression(max_iter=5000, C=0.5)
model.fit(X_train / 16.0, y_train)
float_accuracy = float((model.predict(X_test / 16.0) == y_test).mean())

W, b = model.coef_, model.intercept_                     # (10, 64), (10,)   입력은 x/16
scale_w = WEIGHT_MAX / np.abs(W).max()
W_q = np.clip(np.round(W * scale_w), -WEIGHT_MAX - 1, WEIGHT_MAX).astype(int)
x_q = np.round(X_test * PIXEL_SCALE).astype(int)         # INT8 0..64 / INT4 0..7
# 실수 logits = W·(x/16) + b  ≈  (W_q·x_q) / (scale_w · 16/PIXEL_SCALE · ... ) 정리: W·x/16 = (W_q/scale_w)·(x_q/PIXEL_SCALE)/16
denom = scale_w * PIXEL_SCALE * 16
b_q = np.round(b * denom).astype(int)


def saturate(v):
    return max(-LIMIT - 1, min(LIMIT, int(v)))


dots = [[saturate(int(np.dot(W_q[c], x))) for c in range(10)] for x in x_q]
int_pred = [int(np.argmax(np.array(d) + b_q)) for d in dots]
int_accuracy = float(np.mean(np.array(int_pred) == y_test))

out = {
    'source': 'scikit-learn load_digits (UCI optdigits, CC BY 4.0) · test split 20% random_state=0 stratified',
    'int4': INT4, 'act_bits': ACT_BITS,
    'width': WIDTH, 'pixel_scale': PIXEL_SCALE, 'scale_w': scale_w,
    'weights': W_q.tolist(), 'bias': b_q.tolist(),
    'inputs': x_q.tolist(), 'labels': y_test.tolist(), 'dots': dots,
    'float_accuracy': float_accuracy, 'int8_accuracy': int_accuracy,
    'max_abs_dot': max(abs(v) for d in dots for v in d), 'saturated': sum(abs(v) >= LIMIT for d in dots for v in d),
}
name = f'digits-int4-a{ACT_BITS}.json' if ACT_BITS != 4 else ('digits-int4.json' if INT4 else 'digits.json')
Path(__file__).with_name(name).write_text(json.dumps(out))
print(f"{name} {'INT4' if INT4 else 'INT8'} test {len(y_test)}  float {float_accuracy:.4f}  quantized {int_accuracy:.4f}  max|dot| {out['max_abs_dot']}  saturated {out['saturated']}")
