#!/usr/bin/env python3
"""
inspect_queries.py
Look at the field queries produced by make_field_queries.py.

Two modes:

  single   Rebuild one query from its manifest row and plot the mixture next to
           each individual source and the background noise. This is the useful
           one: it shows you exactly what your system has to pull apart, and
           how little of a -9 dB source survives under crickets at -5 dB SNR.

  summary  Sanity-check the whole split: how the queries fall across the nine
           (k, SNR) cells, what the true abundance values actually look like,
           and the oracle-uniform TVD baseline you will need to compare against.

Usage:
  python inspect_queries.py summary --split dev
  python inspect_queries.py single  --split dev --n 3
  python inspect_queries.py single  --split dev --query-id dev_00042 --write-stems

Paths default to the layout in your source_code folder:
  ./ESC-50/audio, ./ESC-50/meta, ./field_queries/...

Requires numpy and matplotlib only. Audio I/O uses the stdlib `wave` module so
that what you see is bit-identical to what the generator wrote.
"""

import argparse
import csv
import os
import wave

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RATE = 44100


# ---------------------------------------------------------------------------
# Audio helpers. Deliberately the same as make_field_queries.py.
# ---------------------------------------------------------------------------

def read_wav(path):
    """Read a mono 16-bit PCM wav as float64 in [-1, 1)."""
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def write_wav(path, x):
    pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit_length(x, n):
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size)])


# ---------------------------------------------------------------------------
# Manifest / ground truth
# ---------------------------------------------------------------------------

def load_manifest(queries_root, split):
    path = os.path.join(queries_root, "manifest_%s.csv" % split)
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_abundance(queries_root, split):
    """Return (classes, {query_id: np.array of abundances})."""
    path = os.path.join(queries_root, "abundance_%s.csv" % split)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        classes = header[1:]
        table = {row[0]: np.array([float(v) for v in row[1:]]) for row in reader}
    return classes, table


def parse_sources(field):
    """'dog|1-100032-A-0.wav|0.000;cat|...|-4.312' -> [(cls, file, level_db)]"""
    out = []
    for item in field.split(";"):
        cls, fname, level = item.split("|")
        out.append((cls, fname, float(level)))
    return out


# ---------------------------------------------------------------------------
# Rebuild one mixture from its manifest row
# ---------------------------------------------------------------------------

def rebuild(row, audio_dir):
    """Reconstruct the stems exactly as the generator built them.

    Returns a dict with the scaled target stems, the scaled noise stem, and the
    reconstructed mixture. Every stem already carries peak_scale, so they sum
    to the mixture and you can plot them on one shared amplitude axis.
    """
    sources = parse_sources(row["sources"])
    snr_db = float(row["snr_db"])
    peak_scale = float(row["peak_scale"])

    n = None
    stems = []
    targets = None
    for cls, fname, level_db in sources:
        x = read_wav(os.path.join(audio_dir, fname))
        if n is None:
            n = x.size
            targets = np.zeros(n)
        x = fit_length(x, n) / rms(x)
        x = x * (10.0 ** (level_db / 20.0))
        targets = targets + x
        stems.append({"class": cls, "file": fname, "level_db": level_db,
                      "audio": x * peak_scale})

    noise = fit_length(read_wav(os.path.join(audio_dir, row["noise_file"])), n)
    noise = noise / rms(noise)
    noise = noise * (rms(targets) / (10.0 ** (snr_db / 20.0)))

    mixture = (targets + noise) * peak_scale
    return {
        "stems": stems,
        "noise": {"class": row["noise_class"], "audio": noise * peak_scale},
        "mixture": mixture,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def panel(ax_wave, ax_spec, x, title, ylim):
    t = np.arange(x.size) / RATE
    ax_wave.plot(t, x, linewidth=0.4)
    ax_wave.set_ylim(-ylim, ylim)
    ax_wave.set_xlim(0, t[-1] if t.size else 1)
    ax_wave.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)
    ax_wave.set_yticks([])

    ax_spec.specgram(x, NFFT=1024, Fs=RATE, noverlap=512, cmap="magma",
                     vmin=-140, vmax=-20)
    ax_spec.set_yscale("symlog", linthresh=500)
    ax_spec.set_ylim(0, RATE / 2)
    ax_spec.set_yticks([])


def plot_query(row, truth, classes, audio_dir, out_dir, write_stems):
    parts = rebuild(row, audio_dir)
    qid = row["query_id"]

    n_rows = 1 + len(parts["stems"]) + 1
    ylim = float(np.max(np.abs(parts["mixture"]))) * 1.05 or 1.0

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 1.7 * n_rows),
                             gridspec_kw={"width_ratios": [2, 1]})
    if n_rows == 1:
        axes = np.array([axes])

    panel(axes[0][0], axes[0][1], parts["mixture"],
          "MIXTURE\n(what you get)", ylim)

    abund = {c: v for c, v in zip(classes, truth)}
    for i, stem in enumerate(parts["stems"], start=1):
        label = "%s\n%+.1f dB  a=%.3f" % (
            stem["class"], stem["level_db"], abund.get(stem["class"], 0.0))
        panel(axes[i][0], axes[i][1], stem["audio"], label, ylim)

    panel(axes[-1][0], axes[-1][1], parts["noise"]["audio"],
          "noise: %s\n(not in answer)" % parts["noise"]["class"], ylim)

    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")
    fig.suptitle("%s    k=%s    SNR=%s dB" % (qid, row["k"], row["snr_db"]),
                 fontsize=11)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])

    png = os.path.join(out_dir, "%s.png" % qid)
    fig.savefig(png, dpi=110)
    plt.close(fig)

    if write_stems:
        write_wav(os.path.join(out_dir, "%s_mixture.wav" % qid), parts["mixture"])
        for j, stem in enumerate(parts["stems"]):
            write_wav(os.path.join(out_dir, "%s_src%d_%s.wav" % (qid, j, stem["class"])),
                      stem["audio"])
        write_wav(os.path.join(out_dir, "%s_noise_%s.wav" % (qid, parts["noise"]["class"])),
                  parts["noise"]["audio"])

    return png, parts



# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_single(args):
    rows = load_manifest(args.queries_root, args.split)
    classes, truth = load_abundance(args.queries_root, args.split)
    audio_dir = os.path.join(args.esc50_root, "audio")
    os.makedirs(args.out, exist_ok=True)

    if args.query_id:
        chosen = [r for r in rows if r["query_id"] in args.query_id]
        if not chosen:
            raise SystemExit("No such query_id in %s" % args.split)
    else:
        rng = np.random.default_rng(args.seed)
        pool = rows
        if args.k:
            pool = [r for r in pool if int(r["k"]) == args.k]
        if args.snr is not None:
            pool = [r for r in pool if float(r["snr_db"]) == args.snr]
        if not pool:
            raise SystemExit("No queries match those filters.")
        idx = rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)
        chosen = [pool[i] for i in np.atleast_1d(idx)]

    for row in chosen:
        qid = row["query_id"]
        png, parts = plot_query(row, truth[qid], classes, audio_dir,
                                args.out, args.write_stems)

        on_disk = read_wav(os.path.join(args.queries_root, row["wav"]))
        err = float(np.max(np.abs(fit_length(parts["mixture"], on_disk.size) - on_disk)))

        print("%s  k=%s  snr=%s" % (qid, row["k"], row["snr_db"]))
        for cls, fname, lv in parse_sources(row["sources"]):
            a = dict(zip(classes, truth[qid])).get(cls, 0.0)
            print("    %-16s %+6.2f dB   true abundance %.3f" % (cls, lv, a))
        print("    noise            %s" % row["noise_class"])
        print("    rebuild error vs file on disk: %.2e  (16-bit step is 3.05e-05)" % err)
        print("    wrote %s" % png)
        print()


def mode_summary(args):
    rows = load_manifest(args.queries_root, args.split)
    classes, truth = load_abundance(args.queries_root, args.split)
    os.makedirs(args.out, exist_ok=True)

    ks = np.array([int(r["k"]) for r in rows])
    snrs = np.array([float(r["snr_db"]) for r in rows])
    A = np.array([truth[r["query_id"]] for r in rows])

    print("split %s   %d queries   %d classes" % (args.split, len(rows), len(classes)))
    print("\n(k, SNR) cell counts")
    uk, us = sorted(set(ks.tolist())), sorted(set(snrs.tolist()), reverse=True)
    print("        " + "".join("%8s" % ("%+.0f dB" % s) for s in us))
    for k in uk:
        print("  k=%d  " % k + "".join(
            "%8d" % int(((ks == k) & (snrs == s)).sum()) for s in us))

    # The baseline everything else must beat: perfect retrieval, no idea about
    # levels. Predict 1/k mass on each true class. TVD is then purely the cost
    # of not estimating the relative levels.
    print("\noracle-uniform TVD  (correct classes, equal weights)")
    for k in uk:
        if k == 1:
            continue
        sel = ks == k
        true = A[sel]
        pred = (true > 0).astype(float)
        pred = pred / pred.sum(axis=1, keepdims=True)
        tvd = 0.5 * np.abs(pred - true).sum(axis=1)
        print("  k=%d   mean %.4f   (n=%d)" % (k, tvd.mean(), sel.sum()))
    sel = ks >= 2
    true = A[sel]
    pred = (true > 0).astype(float) / (true > 0).sum(axis=1, keepdims=True)
    print("  k>=2  mean %.4f" % (0.5 * np.abs(pred - true).sum(axis=1)).mean())

    nz = A[A > 0]
    print("\nnon-zero true abundances: min %.3f  median %.3f  max %.3f"
          % (nz.min(), np.median(nz), nz.max()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(nz, bins=40)
    ax[0].set_title("true abundance values (non-zero)")
    ax[0].set_xlabel("abundance")
    counts = np.zeros((len(uk), len(us)))
    for i, k in enumerate(uk):
        for j, s in enumerate(us):
            counts[i, j] = ((ks == k) & (snrs == s)).sum()
    im = ax[1].imshow(counts, cmap="Blues")
    ax[1].set_xticks(range(len(us)), ["%+.0f" % s for s in us])
    ax[1].set_yticks(range(len(uk)), ["k=%d" % k for k in uk])
    ax[1].set_title("queries per (k, SNR) cell")
    ax[1].set_xlabel("SNR dB")
    for i in range(len(uk)):
        for j in range(len(us)):
            ax[1].text(j, i, "%d" % counts[i, j], ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    png = os.path.join(args.out, "summary_%s.png" % args.split)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print("\nwrote %s" % png)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["single", "summary"])
    ap.add_argument("--esc50-root", default="./ESC-50")
    ap.add_argument("--queries-root", default="./field_queries")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "test_late"])
    ap.add_argument("--out", default="./inspect")
    ap.add_argument("--query-id", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3, help="random queries to plot")
    ap.add_argument("--k", type=int, default=None, help="filter on k")
    ap.add_argument("--snr", type=float, default=None, help="filter on SNR dB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-stems", action="store_true",
                    help="also write the mixture and each stem as wav, to listen")
    args = ap.parse_args()

    matplotlib.use("Agg")
    if args.mode == "single":
        mode_single(args)
    else:
        mode_summary(args)


if __name__ == "__main__":
    main()