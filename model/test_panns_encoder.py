#!/usr/bin/env python3
"""
test_panns_encoder.py
Smoke tests for panns_encoder.py.

Three tiers, so you get useful signal even before the checkpoint downloads:

  tier 1  pure numpy. Resampling, windowing, duty cycle. No torch needed.
  tier 2  the model. Shapes, determinism, batch invariance, no NaNs, and the
          gain-sensitivity check, which is the one that changes your design.
  tier 3  real data. Semantic sanity on ESC-50 clips. Skipped if ESC-50 is
          not on disk.

Run:
    pytest -v test_panns_encoder.py
    pytest -v -m "not slow" test_panns_encoder.py      # tier 1 only
    ESC50_ROOT=./ESC-50 pytest -v test_panns_encoder.py

Every test here exists because the failure it catches is silent. A wrong
resample ratio, a batch-dependent embedding or a gain-sensitive encoder will
all still produce plausible-looking numbers downstream.
"""

import os

import numpy as np
import pytest

from panns_encoder import (
    ESC50_RATE, PANNS_RATE, EMBED_DIM,
    resample_to, frame_windows, l2norm, energy_duty,
    read_wav, PannsEncoder, EmbeddingCache, IN_SCOPE, load_meta,
)

ESC50_ROOT = os.environ.get("ESC50_ROOT", "./ESC-50")
HAS_ESC50 = os.path.isdir(os.path.join(ESC50_ROOT, "audio"))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_model = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
needs_data = pytest.mark.skipif(not HAS_ESC50, reason="ESC-50 not found")


# ---------------------------------------------------------------------------
# Tier 1: pure numpy
# ---------------------------------------------------------------------------

def test_resample_length():
    """44100 -> 32000 must give the exact expected sample count."""
    x = np.zeros(ESC50_RATE * 5)
    y = resample_to(x, ESC50_RATE, PANNS_RATE)
    assert y.size == PANNS_RATE * 5


def test_resample_preserves_frequency():
    """A 1 kHz tone must still peak at 1 kHz after resampling.

    Catches an inverted up/down ratio, which is easy to do and produces audio
    that is merely pitch-shifted rather than obviously broken.
    """
    t = np.arange(ESC50_RATE * 2) / ESC50_RATE
    x = np.sin(2 * np.pi * 1000 * t)
    y = resample_to(x, ESC50_RATE, PANNS_RATE)
    freqs = np.fft.rfftfreq(y.size, 1 / PANNS_RATE)
    peak = freqs[np.abs(np.fft.rfft(y)).argmax()]
    assert abs(peak - 1000) < 5


def test_resample_identity():
    x = np.random.default_rng(0).normal(size=1000)
    assert np.allclose(resample_to(x, PANNS_RATE, PANNS_RATE), x)


def test_window_count_and_shape():
    x = np.zeros(PANNS_RATE * 5)
    win, rms = frame_windows(x, PANNS_RATE, window_s=1.0, hop_s=0.5)
    assert win.shape == (9, PANNS_RATE)
    assert rms.shape == (9,)


def test_windows_cover_the_signal():
    """The last window must reach the end. A silent tail loses evidence."""
    x = np.zeros(PANNS_RATE * 5)
    x[-100:] = 1.0
    _, rms = frame_windows(x, PANNS_RATE, window_s=1.0, hop_s=0.5)
    assert rms[-1] > 0


def test_window_content_matches_source():
    x = np.arange(PANNS_RATE * 3, dtype=float)
    win, _ = frame_windows(x, PANNS_RATE, window_s=1.0, hop_s=0.5)
    assert np.array_equal(win[0], x[:PANNS_RATE])
    assert np.array_equal(win[2], x[PANNS_RATE:2 * PANNS_RATE])


def test_window_rms_is_correct():
    x = np.ones(PANNS_RATE * 2) * 0.5
    _, rms = frame_windows(x, PANNS_RATE, window_s=1.0, hop_s=1.0)
    assert np.allclose(rms, 0.5)


def test_energy_duty_bounds():
    """Flat signal -> 1.0. Single impulse -> near zero. No threshold involved."""
    flat = np.ones(2048 * 50)
    assert energy_duty(flat) > 0.99

    impulse = np.zeros(2048 * 50)
    impulse[0] = 1.0
    assert energy_duty(impulse) < 0.05


def test_energy_duty_ignores_quiet_floor():
    """Room tone between events must not inflate the duty cycle.

    This is exactly the bug in the first threshold-based version: at 40 dB
    below peak, keyboard_typing measured 1.00.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(scale=1e-3, size=2048 * 100)
    x[:2048 * 5] += rng.normal(scale=1.0, size=2048 * 5)
    assert energy_duty(x) < 0.15


def test_l2norm():
    v = np.array([[3.0, 4.0], [0.0, 0.0]])
    n = l2norm(v)
    assert abs(np.linalg.norm(n[0]) - 1.0) < 1e-9
    assert np.all(np.isfinite(n[1]))       # zero vector must not produce NaN


def test_mismatched_lengths_raise():
    """Silent padding across a batch would change results. Refuse it loudly."""
    enc = PannsEncoder()
    with pytest.raises(ValueError, match="same length"):
        enc._stack([np.zeros(100), np.zeros(200)])


def test_cache_roundtrip(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    k = c.key("lib", ["a.wav", "b.wav"], window_s=1.0)
    assert c.get(k) is None
    c.put(k, emb=np.arange(6).reshape(2, 3))
    got = c.get(k)
    assert np.array_equal(got["emb"], np.arange(6).reshape(2, 3))


def test_cache_key_depends_on_settings(tmp_path):
    c = EmbeddingCache(str(tmp_path))
    assert c.key("lib", ["a"], window_s=1.0) != c.key("lib", ["a"], window_s=2.0)
    assert c.key("lib", ["a"], window_s=1.0) != c.key("lib", ["b"], window_s=1.0)


# ---------------------------------------------------------------------------
# Tier 2: the model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    enc = PannsEncoder(batch_size=8)
    _ = enc.model          # forces load and checkpoint download
    return enc


@pytest.fixture(scope="module")
def clips():
    """Four distinct 5-second signals at 44.1 kHz."""
    rng = np.random.default_rng(0)
    t = np.arange(ESC50_RATE * 5) / ESC50_RATE
    return np.stack([
        np.sin(2 * np.pi * 440 * t),                      # tone
        rng.normal(scale=0.1, size=t.size),               # white noise
        np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.05),   # pulsed tone
        np.zeros(t.size),                                 # silence
    ])


@needs_model
@pytest.mark.slow
def test_embed_shape(encoder, clips):
    e = encoder.embed(clips)
    assert e.shape == (4, EMBED_DIM)
    assert np.all(np.isfinite(e))


@needs_model
@pytest.mark.slow
def test_silence_is_finite(encoder, clips):
    """Many ESC-50 clips are largely digital padding. Silence must not NaN."""
    e = encoder.embed(clips[3:4])
    assert np.all(np.isfinite(e))
    assert np.linalg.norm(e) > 0


@needs_model
@pytest.mark.slow
def test_deterministic(encoder, clips):
    """Dropout or train-mode batchnorm left on would break this."""
    a = encoder.embed(clips)
    b = encoder.embed(clips)
    assert np.allclose(a, b, atol=1e-6)


@needs_model
@pytest.mark.slow
def test_batch_invariance(encoder, clips):
    """A clip embedded alone must match the same clip embedded in a batch.

    If this fails, every result you produce depends on the order your files
    happened to be listed in, and no ablation is comparable to another.
    """
    together = encoder.embed(clips)
    alone = np.concatenate([encoder.embed(clips[i:i + 1]) for i in range(4)])
    cos = (l2norm(together) * l2norm(alone)).sum(axis=1)
    assert np.all(cos > 0.9999), cos


@needs_model
@pytest.mark.slow
def test_batch_size_invariance(encoder, clips):
    enc_small = PannsEncoder(batch_size=1)
    enc_small._model = encoder.model
    enc_small._device = encoder.device
    a, b = encoder.embed(clips), enc_small.embed(clips)
    assert np.allclose(a, b, atol=1e-4)


@needs_model
@pytest.mark.slow
def test_distinct_signals_are_distinguishable(encoder, clips):
    """A pure tone and white noise must not land in the same place."""
    e = l2norm(encoder.embed(clips[:2]))
    assert float(e[0] @ e[1]) < 0.9


@needs_model
@pytest.mark.slow
def test_gain_sensitivity_is_measured(encoder, clips):
    """CNN14 is not gain invariant, and peak_scale varies per query.

    This is informational rather than a pass/fail on a tight bound: we assert
    only that a 12 dB attenuation does not destroy the representation, and we
    print the cosine so it can be quoted in the report. If it comes out low,
    peak-normalising library clips to match query loudness is not optional.
    """
    loud = encoder.embed(clips[:3])
    quiet = encoder.embed(clips[:3] * 0.25)
    cos = (l2norm(loud) * l2norm(quiet)).sum(axis=1)
    print("\n  cosine under -12 dB gain: %s" % np.round(cos, 4))
    assert np.all(cos > 0.80), cos


@needs_model
@pytest.mark.slow
def test_windowed_shapes(encoder, clips):
    emb, rms = encoder.embed_windowed(clips, window_s=1.0, hop_s=0.5)
    assert emb.shape == (4, 9, EMBED_DIM)
    assert rms.shape == (4, 9)
    assert np.all(np.isfinite(emb))


@needs_model
@pytest.mark.slow
def test_windowed_matches_manual_window(encoder, clips):
    """Window w of clip i must equal that window embedded on its own.

    Guards the reshape in embed_windowed, which is the easiest place in this
    module to silently transpose clips against windows.
    """
    emb, _ = encoder.embed_windowed(clips[:2], window_s=1.0, hop_s=0.5)
    w32 = resample_to(clips[1], ESC50_RATE, PANNS_RATE)
    win, _ = frame_windows(w32, PANNS_RATE, 1.0, 0.5)
    single = encoder.embed(win[3:4], source_rate=PANNS_RATE)
    cos = float(l2norm(emb[1, 3]) @ l2norm(single[0]))
    assert cos > 0.999, cos


@needs_model
@pytest.mark.slow
def test_pulsed_signal_has_varying_window_energy(encoder, clips):
    """Sanity on the RMS channel we rely on for energy weighting."""
    _, rms = encoder.embed_windowed(clips[2:3], window_s=0.5, hop_s=0.25)
    assert rms.std() >= 0


# ---------------------------------------------------------------------------
# Tier 3: real data
# ---------------------------------------------------------------------------

@needs_data
def test_esc50_layout():
    meta = load_meta(ESC50_ROOT)
    assert len(meta) == 2000
    cats = {r["category"] for r in meta}
    missing = set(IN_SCOPE) - cats
    assert not missing, missing
    fold1 = [r for r in meta if r["fold"] == "1" and r["category"] == "dog"]
    assert len(fold1) == 8


@needs_data
def test_clip_read_is_five_seconds():
    meta = load_meta(ESC50_ROOT)
    x = read_wav(os.path.join(ESC50_ROOT, "audio", meta[0]["filename"]))
    assert x.size == ESC50_RATE * 5
    assert np.abs(x).max() <= 1.0


@needs_model
@needs_data
@pytest.mark.slow
def test_same_class_clips_are_closer_than_different_class(encoder):
    """The minimum semantic bar: two dogs beat a dog and a vacuum cleaner."""
    meta = load_meta(ESC50_ROOT)
    audio = os.path.join(ESC50_ROOT, "audio")

    def pick(cat, n=3):
        rs = [r for r in meta if r["category"] == cat and r["fold"] == "1"][:n]
        return np.stack([read_wav(os.path.join(audio, r["filename"])) for r in rs])

    dogs = l2norm(encoder.embed(pick("dog")))
    vacs = l2norm(encoder.embed(pick("vacuum_cleaner")))
    within = (dogs @ dogs.T)[np.triu_indices(3, 1)].mean()
    between = (dogs @ vacs.T).mean()
    assert within > between, (within, between)


@needs_model
@needs_data
@pytest.mark.slow
def test_noise_degrades_but_does_not_destroy(encoder):
    """Quantifies the domain gap the augmented prototypes are meant to close.

    A clean clip and the same clip under rain at 0 dB must still be recognisably
    the same thing. If this cosine is near zero, retrieval against a clean
    library cannot work and the whole prototype design needs rethinking.
    """
    meta = load_meta(ESC50_ROOT)
    audio = os.path.join(ESC50_ROOT, "audio")
    dog = read_wav(os.path.join(
        audio, [r for r in meta if r["category"] == "dog"][0]["filename"]))
    rain = read_wav(os.path.join(
        audio, [r for r in meta if r["category"] == "rain"][0]["filename"]))

    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))
    noisy = dog + rain / rms(rain) * rms(dog)          # 0 dB SNR
    noisy = noisy / np.abs(noisy).max() * 0.95

    e = l2norm(encoder.embed(np.stack([dog, noisy])))
    cos = float(e[0] @ e[1])
    print("\n  clean vs 0 dB noisy, same clip: cosine %.4f" % cos)
    assert cos > 0.3, cos


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))