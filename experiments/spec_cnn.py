"""SpecCNN-lite: 주파수 도메인 기반 이상 점수(의존성 최소화).

설계:
- 고정길이 윈도우(예: 128)로 구간을 나누고, 각 구간의 정규화된 스펙트럼 에너지를 계산.
- 저/중/고 3개 대역 에너지를 특징으로 사용.
- 간단한 1D '필터'(가중합)를 적용하여 점수화. (학습 대신 휴리스틱 가중치)
- 구간 점수를 원본 길이로 확장(윈도우 중심에 할당, 선형 보간)하여 포인트별 스코어 생성.

의존성: math, cmath만 사용 (numpy 미사용)
"""
from __future__ import annotations

from typing import List, Dict
import math
import cmath


def _hann(n: int) -> List[float]:
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / max(1, n - 1)) for i in range(n)]


def _dft_mag(x: List[float]) -> List[float]:
    N = len(x)
    mags: List[float] = []
    for k in range(N // 2 + 1):  # real-valued symmetry
        s = 0j
        for n, xn in enumerate(x):
            s += complex(xn, 0.0) * cmath.exp(-2j * math.pi * k * n / N)
        mags.append(abs(s))
    return mags


def _band_energy(mag: List[float], lo: float, hi: float) -> float:
    N = len(mag)
    ilo = max(0, int(lo * (N - 1)))
    ihi = min(N - 1, int(hi * (N - 1)))
    s = 0.0
    for i in range(ilo, ihi + 1):
        s += mag[i] * mag[i]
    return s / max(1, (ihi - ilo + 1))


def _window_scores(series: List[float], window: int, hop: int) -> List[float]:
    n = len(series)
    if n == 0:
        return []
    w = max(8, int(window))
    h = max(1, int(hop))
    win = _hann(w)
    scores: List[float] = []
    # heuristic filter weights: emphasize mid/high band rise and low band drop
    w_low, w_mid, w_high = -0.2, 0.6, 0.6
    for start in range(0, n - w + 1, h):
        frame = [series[start + i] * win[i] for i in range(w)]
        mag = _dft_mag(frame)
        # normalize magnitude by frame energy to reduce amplitude effect
        energy = sum(v * v for v in frame) / w
        scale = 1.0 / max(1e-8, energy)
        magn = [m * scale for m in mag]
        # three bands: [0,0.1], (0.1,0.3], (0.3,0.5]
        e_low = _band_energy(magn, 0.0, 0.1)
        e_mid = _band_energy(magn, 0.1, 0.3)
        e_high = _band_energy(magn, 0.3, 0.5)
        s = w_low * e_low + w_mid * e_mid + w_high * e_high
        scores.append(max(0.0, s))
    return scores


def _upsample_to_points(win_scores: List[float], n: int, hop: int, window: int) -> List[float]:
    if not win_scores:
        return [0.0] * n
    h = max(1, int(hop))
    w = max(1, int(window))
    # map each window score to its center index
    centers: List[int] = []
    for i, _ in enumerate(win_scores):
        start = i * h
        center = start + w // 2
        centers.append(min(n - 1, max(0, center)))
    # linear interpolation over centers
    out = [0.0] * n
    for i in range(len(centers) - 1):
        c0, c1 = centers[i], centers[i + 1]
        v0, v1 = win_scores[i], win_scores[i + 1]
        span = max(1, c1 - c0)
        for j in range(span):
            t = j / span
            out[c0 + j] = v0 * (1 - t) + v1 * t
    # fill tails
    if centers:
        for i in range(0, centers[0]):
            out[i] = win_scores[0]
        for i in range(centers[-1], n):
            out[i] = win_scores[-1]
    return out


def detect_speccnn(series: List[float], window: int = 128, hop: int = 16, quantile: float = 0.99) -> Dict:
    ws = _window_scores(series, window=window, hop=hop)
    scores = _upsample_to_points(ws, n=len(series), hop=hop, window=window)
    # quantile threshold
    # reuse ml_detector_knn._quantile to avoid re-implementing
    from .ml_detector_knn import _quantile

    thr = _quantile(scores, quantile)
    preds = [1 if s >= thr else 0 for s in scores]
    return {
        "scores": scores,
        "preds": preds,
        "params": {
            "method": "speccnn_lite",
            "window": int(window),
            "hop": int(hop),
            "quantile": float(quantile),
        },
    }

