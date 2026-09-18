#!/usr/bin/env python3
"""
duty_cycle.py
How much of each query actually contains its target sources?

ESC-50 clips are Freesound extracts padded to a uniform five seconds, so many
of them are mostly silence. The generator computes rms(targets) over the whole
five seconds, which means a source that is only active for a fraction d of the
clip sits at roughly

    local SNR = nominal SNR - 10 * log10(d)

wherever it is actually present. A mouse_click clip labelled 0 dB can be at
+15 dB during the clicks themselves.

This script measures d for every target clip used in a split and reports:
  - the distribution of duty cycle over clips
  - the median duty cycle per class, which says which classes are sparse
  - the gap between nominal and local SNR per query

Usage:
  python duty_cycle.py --split dev
  python duty_cycle.py --split dev --thresh 40 --frame 2048

Requires numpy and matplotlib.
"""

import argparse
import csv
import os
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATE = 44100


def read_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit_length(x, n):
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size)])


def duty_cycle(x, frame=2048, thresh_db=40.0):
    """Fraction of frames whose RMS is within thresh_db of the loudest frame.

    A crude but honest activity detector. Digital padding sits at -inf dB and
    is always excluded; quiet room tone is excluded at 40 dB down and kept at
    60, which is why both numbers are reported.
    """
    n = (x.size // frame) * frame
    if n == 0:
        return 1.0
    f = x[:n].reshape(-1, frame)
    e = 20.0 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)
    return float((e > e.max() - thresh_db).mean())

def energy_duty(x, frame=2048,thresh_db=40.0):
    n = (x.size // frame) * frame
    p = (x[:n].reshape(-1, frame) ** 2).mean(axis=1)
    if p.sum() <= 0:
        return 1.0
    return float(p.sum() ** 2 / (len(p) * (p ** 2).sum()))

def parse_sources(field):
    out = []
    for item in field.split(";"):
        cls, fname, level = item.split("|")
        out.append((cls, fname, float(level)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esc50-root", default="./ESC-50")
    ap.add_argument("--queries-root", default="./field_queries")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "test_late"])
    ap.add_argument("--out", default="./inspect")
    ap.add_argument("--frame", type=int, default=2048)
    ap.add_argument("--thresh", type=float, default=40.0)
    args = ap.parse_args()

    audio_dir = os.path.join(args.esc50_root, "audio")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.queries_root, "manifest_%s.csv" % args.split),
              newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Duty cycle depends only on the clip, so cache it.
    cache40, cache60, audio_cache = {}, {}, {}

    def get_audio(fname):
        if fname not in audio_cache:
            audio_cache[fname] = read_wav(os.path.join(audio_dir, fname))
        return audio_cache[fname]

    def get_duty(fname):
        if fname not in cache40:
            x = get_audio(fname)
            cache40[fname] = energy_duty(x, args.frame, args.thresh)
            cache60[fname] = energy_duty(x, args.frame, 60.0)
        return cache40[fname], cache60[fname]

    per_class = {}
    query_rows = []

    for r in rows:
        sources = parse_sources(r["sources"])
        snr_db = float(r["snr_db"])

        # Duty cycle of the summed, level-scaled targets. This is the one that
        # sets the local SNR, because the noise is added against their joint RMS.
        n = None
        targets = None
        for cls, fname, level_db in sources:
            x = get_audio(fname)
            if n is None:
                n = x.size
                targets = np.zeros(n)
            targets = targets + fit_length(x, n) / rms(x) * (10.0 ** (level_db / 20.0))

        d_joint = energy_duty(targets, args.frame, args.thresh)
        local_snr = snr_db - 10.0 * np.log10(max(d_joint, 1e-6))

        for cls, fname, level_db in sources:
            d40, d60 = get_duty(fname)
            per_class.setdefault(cls, []).append(d40)

        query_rows.append({"k": int(r["k"]), "snr": snr_db,
                           "d_joint": d_joint, "local_snr": local_snr})

    all_d40 = np.array([d for v in cache40.values() for d in [v]])
    all_d60 = np.array(list(cache60.values()))
    d_joint = np.array([q["d_joint"] for q in query_rows])
    nominal = np.array([q["snr"] for q in query_rows])
    local = np.array([q["local_snr"] for q in query_rows])

    print("split %s   %d queries   %d distinct target clips"
          % (args.split, len(rows), len(cache40)))
    print("\nper-clip duty cycle")
    print("  threshold %2.0f dB below peak frame:  median %.3f   mean %.3f   below 0.5: %.0f%%"
          % (args.thresh, np.median(all_d40), all_d40.mean(), 100 * (all_d40 < 0.5).mean()))
    print("  threshold 60 dB below peak frame:  median %.3f   mean %.3f   below 0.5: %.0f%%"
          % (np.median(all_d60), all_d60.mean(), 100 * (all_d60 < 0.5).mean()))

    print("\nper-query joint target duty cycle: median %.3f" % np.median(d_joint))
    print("\nnominal vs local SNR, by nominal cell")
    for s in sorted(set(nominal.tolist()), reverse=True):
        sel = nominal == s
        print("  nominal %+5.1f dB   local median %+5.1f dB   (gain %+4.1f dB)"
              % (s, np.median(local[sel]), np.median(local[sel]) - s))

    order = sorted(per_class, key=lambda c: np.median(per_class[c]))
    print("\nmedian duty cycle by class, sparsest first")
    for c in order:
        print("  %-18s %.3f" % (c, np.median(per_class[c])))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    ax[0].hist(all_d40, bins=30)
    ax[0].axvline(np.median(all_d40), color="crimson", linestyle="--")
    ax[0].set_title("duty cycle per target clip\nmedian %.2f" % np.median(all_d40))
    ax[0].set_xlabel("fraction of clip active")

    meds = [np.median(per_class[c]) for c in order]
    ax[1].barh(range(len(order)), meds)
    ax[1].set_yticks(range(len(order)), order, fontsize=8)
    ax[1].axvline(0.5, color="crimson", linestyle="--", linewidth=0.8)
    ax[1].set_title("median duty cycle by class")
    ax[1].set_xlabel("fraction of clip active")

    for s in sorted(set(nominal.tolist()), reverse=True):
        sel = nominal == s
        ax[2].hist(local[sel], bins=25, alpha=0.6, label="nominal %+.0f dB" % s)
    ax[2].set_title("local SNR where the target is active")
    ax[2].set_xlabel("dB")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(args.out, "energy_duty_%s.png" % args.split)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("\nwrote %s" % png)


if __name__ == "__main__":
    main()