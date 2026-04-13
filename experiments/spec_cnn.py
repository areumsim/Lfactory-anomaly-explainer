"""SpecCNN-lite: 주파수 도메인 기반 이상 점수.

설계:
- 고정길이 윈도우(예: 128)로 구간을 나누고, 각 구간의 스펙트럼 밴드 에너지를 계산.
- 저/중/고 3개 대역 에너지를 특징으로 사용.
- **Spectral Flux** 방식: 연속 프레임 간 밴드 에너지 변화량(half-wave rectified)을
  이상 점수로 사용. 정상 구간은 스펙트럼이 안정적이므로 flux ≈ 0, 이상 구간은 급격한
  주파수 변화로 flux가 커짐.
- 구간 점수를 원본 길이로 확장(윈도우 중심에 할당, 선형 보간)하여 포인트별 스코어 생성.

의존성: numpy (FFT 가속)
"""
from __future__ import annotations

from typing import List, Dict
import math

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _hann(n: int) -> List[float]:
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / max(1, n - 1)) for i in range(n)]


def _dft_mag_numpy(x: List[float]) -> List[float]:
    """FFT magnitude using numpy (O(N log N))."""
    arr = np.array(x, dtype=np.float64)
    mag = np.abs(np.fft.rfft(arr))
    return mag.tolist()


def _dft_mag_pure(x: List[float]) -> List[float]:
    """Pure-Python DFT magnitude (O(N^2) fallback)."""
    import cmath
    N = len(x)
    mags: List[float] = []
    for k in range(N // 2 + 1):
        s = 0j
        for n_idx, xn in enumerate(x):
            s += complex(xn, 0.0) * cmath.exp(-2j * math.pi * k * n_idx / N)
        mags.append(abs(s))
    return mags


def _dft_mag(x: List[float]) -> List[float]:
    if _HAS_NUMPY:
        return _dft_mag_numpy(x)
    return _dft_mag_pure(x)


def _band_energy(mag: List[float], lo: float, hi: float) -> float:
    N = len(mag)
    ilo = max(0, int(lo * (N - 1)))
    ihi = min(N - 1, int(hi * (N - 1)))
    s = 0.0
    for i in range(ilo, ihi + 1):
        s += mag[i] * mag[i]
    return s / max(1, (ihi - ilo + 1))


def _window_scores(
    series: List[float],
    window: int,
    hop: int,
    w_low: float = 0.3,
    w_mid: float = 0.4,
    w_high: float = 0.3,
) -> List[float]:
    """Compute per-window anomaly scores using spectral flux.

    Spectral flux measures the frame-to-frame change in band energies.
    Half-wave rectification keeps only energy increases (onset detection).
    """
    n = len(series)
    if n == 0:
        return []
    w = max(8, int(window))
    h = max(1, int(hop))
    win = _hann(w)

    # Phase 1: compute raw band energies for all windows
    raw_features: List[List[float]] = []  # each entry: [e_low, e_mid, e_high]
    for start in range(0, n - w + 1, h):
        frame = [series[start + i] * win[i] for i in range(w)]
        mag = _dft_mag(frame)
        e_low = _band_energy(mag, 0.0, 0.1)
        e_mid = _band_energy(mag, 0.1, 0.3)
        e_high = _band_energy(mag, 0.3, 0.5)
        raw_features.append([e_low, e_mid, e_high])

    if len(raw_features) < 2:
        return [0.0] * max(1, len(raw_features))

    # Phase 2: Spectral Flux — half-wave rectified band energy difference
    weights = [w_low, w_mid, w_high]
    scores: List[float] = [0.0]  # first window has no predecessor
    for i in range(1, len(raw_features)):
        flux = 0.0
        for b in range(3):
            diff = raw_features[i][b] - raw_features[i - 1][b]
            flux += weights[b] * max(0.0, diff)  # half-wave rectification
        scores.append(flux)

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


def _minmax_normalize(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range for consistency with other detectors."""
    if not scores:
        return scores
    lo = min(scores)
    hi = max(scores)
    rng = hi - lo
    if rng < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / rng for s in scores]


def detect_speccnn(
    series: List[float],
    window: int = 128,
    hop: int = 16,
    quantile: float = 0.99,
    w_low: float = 0.3,
    w_mid: float = 0.4,
    w_high: float = 0.3,
) -> Dict:
    ws = _window_scores(series, window=window, hop=hop, w_low=w_low, w_mid=w_mid, w_high=w_high)
    raw_scores = _upsample_to_points(ws, n=len(series), hop=hop, window=window)
    scores = _minmax_normalize(raw_scores)
    # quantile threshold on normalized [0,1] scores
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
            "w_low": float(w_low),
            "w_mid": float(w_mid),
            "w_high": float(w_high),
        },
    }
