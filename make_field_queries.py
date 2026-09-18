#!/usr/bin/env python3
"""
make_field_queries.py
Esper Satellites - AI Engineer technical challenge.

Turns the clean ESC-50 query folds into field-like mixtures and writes the
ground truth alongside them.

Run this script unmodified for every result you report, so that your figures
and ours refer to the same data. Every draw is derived from SHA-256 rather than
from a library random number generator, so the recipe is fixed by this file
alone and does not depend on the version of numpy you have installed.

Usage:
    python make_field_queries.py --esc50-root /path/to/ESC-50-master --out ./field_queries

--esc50-root is the directory containing audio/ and meta/esc50.csv.

Requires numpy. Audio I/O uses the standard library `wave` module, so the
output does not depend on the version of any third-party audio decoder.
"""

import argparse
import csv
import hashlib
import os
import sys
import wave

import numpy as np

# --------------------------------------------------------------------------
# Fixed parameters. Do not change these for any result you report.
# --------------------------------------------------------------------------

MASTER_SEED = 20260915

NOISE_BANK = [
    "rain", "sea_waves", "wind", "crackling_fire", "crickets",
]

IN_SCOPE = [
    "dog", "rooster", "pig", "cow", "cat", "hen", "sheep", "frog",
    "door_wood_knock", "mouse_click", "keyboard_typing", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking", "siren", "car_horn", "train",
]

LATE_ADDITION = [
    "church_bells", "helicopter", "chainsaw", "hand_saw", "engine",
]

LIBRARY_FOLD = 1

# split name -> (folds, primary class pool, secondary class pool, split id)
# The split id enters the per-query seed, so the splits are independent of
# each other and of the order in which they are generated.
SPLITS = {
    "dev":       ((2, 3), IN_SCOPE, IN_SCOPE,               0),
    "test":      ((4, 5), IN_SCOPE, IN_SCOPE,               1),
    "test_late": ((4, 5), LATE_ADDITION, IN_SCOPE + LATE_ADDITION, 2),
}

K_CHOICES = (1, 2, 3)          # number of target sources
LEVEL_RANGE_DB = (-9.0, 0.0)   # level of each further source, rel. to the first
SNR_CHOICES_DB = (10.0, 0.0, -5.0)
# k and SNR are assigned so that the nine (k, SNR) cells are evenly filled
# within each split; which cell a given query falls in is shuffled.
REPEATS_PER_PRIMARY = 3        # queries generated from each primary clip
OUTPUT_PEAK = 0.95             # every mixture is scaled to this peak
MIN_RMS = 1e-5                 # clips quieter than this are not used

EXPECTED_RATE = 44100
EXPECTED_WIDTH = 2             # 16-bit PCM
EXPECTED_CHANNELS = 1


# --------------------------------------------------------------------------
# Audio I/O. Standard library only, so decoding is bit-exact everywhere.
# --------------------------------------------------------------------------

def read_wav(path):
    """Read a mono 16-bit PCM WAV file as float64 in [-1, 1)."""
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
            EXPECTED_CHANNELS, EXPECTED_WIDTH, EXPECTED_RATE
        ):
            raise ValueError(
                "%s is %d ch / %d-bit / %d Hz, expected mono / 16-bit / %d Hz. "
                "Use the ESC-50 audio as distributed, without resampling."
                % (path, w.getnchannels(), 8 * w.getsampwidth(),
                   w.getframerate(), EXPECTED_RATE)
            )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def write_wav(path, x):
    """Write float64 in [-1, 1] as mono 16-bit PCM."""
    pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(EXPECTED_CHANNELS)
        w.setsampwidth(EXPECTED_WIDTH)
        w.setframerate(EXPECTED_RATE)
        w.writeframes(pcm.tobytes())
    return pcm


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit_length(x, n):
    """Pad with zeros or truncate to n samples. ESC-50 clips are all the same
    length, so this is a guard rather than a routine operation."""
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size)])


# --------------------------------------------------------------------------
# Deterministic draws
# --------------------------------------------------------------------------

class Draws(object):
    """A random stream derived from SHA-256.

    NumPy does not guarantee that the Generator stream stays the same between
    releases (NEP 19), so seeding numpy would make the data depend on the
    version a candidate happens to have installed, and a mismatch would be
    silent. Deriving every draw from SHA-256 instead fixes the data for good:
    the output depends only on this file.
    """

    def __init__(self, *key):
        self._key = "|".join(str(k) for k in key)
        self._n = 0

    def _u64(self):
        digest = hashlib.sha256(
            ("%s|%d" % (self._key, self._n)).encode("utf-8")).digest()
        self._n += 1
        return int.from_bytes(digest[:8], "big")

    def unit(self):
        """Uniform in [0, 1), 53 bits of resolution."""
        return (self._u64() >> 11) * (2.0 ** -53)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.unit()

    def index(self, n):
        """Uniform integer in [0, n). Rejection sampling, so no modulo bias."""
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            v = self._u64()
            if v < limit:
                return v % n

    def choice(self, seq):
        seq = list(seq)
        return seq[self.index(len(seq))]

    def sample(self, seq, k):
        """k distinct items, in draw order."""
        pool = list(seq)
        return [pool.pop(self.index(len(pool))) for _ in range(k)]

    def permutation(self, n):
        """Fisher-Yates, drawing downwards."""
        out = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.index(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def load_corpus(root):
    """Return {category: {fold: [filenames sorted]}} for every ESC-50 clip."""
    meta = os.path.join(root, "meta", "esc50.csv")
    if not os.path.isfile(meta):
        raise SystemExit("Cannot find %s. Point --esc50-root at the directory "
                         "containing audio/ and meta/esc50.csv." % meta)
    corpus = {}
    with open(meta, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            corpus.setdefault(row["category"], {}).setdefault(
                int(row["fold"]), []).append(row["filename"])
    for cat in corpus:
        for fold in corpus[cat]:
            corpus[cat][fold].sort()
    missing = [c for c in NOISE_BANK + IN_SCOPE + LATE_ADDITION if c not in corpus]
    if missing:
        raise SystemExit("Classes missing from esc50.csv: %s" % ", ".join(missing))
    return corpus


def eligible(corpus, audio_dir, category, folds, cache):
    """Clips of `category` in `folds` whose RMS clears MIN_RMS, sorted.

    Near-silent clips are excluded so that normalising a source to unit RMS
    cannot amplify numerical noise. The exclusion is deterministic, so every
    candidate works from the same set.
    """
    key = (category, folds)
    if key in cache:
        return cache[key]
    out = []
    for fold in folds:
        for name in corpus[category].get(fold, []):
            path = os.path.join(audio_dir, name)
            x = read_wav(path)
            if rms(x) >= MIN_RMS:
                out.append(name)
    cache[key] = out
    return out


# --------------------------------------------------------------------------
# One query
# --------------------------------------------------------------------------

def build_query(draws, audio_dir, pools, noise_pool, label_space, k, snr_db):
    """Mix one field query. Returns (mixture, record).

    Levels are applied to unit-RMS sources, so the power contributed by
    source i is exactly 10 ** (level_i / 10) and the abundances follow in
    closed form. They do not depend on floating-point details of the audio.
    """
    primary_class, primary_file = pools["primary"]

    others = [c for c in pools["secondary_classes"] if c != primary_class]
    extra_classes = draws.sample(others, k - 1) if k > 1 else []

    sources = [(primary_class, primary_file, 0.0)]
    for cls in extra_classes:
        clips = pools["secondary_clips"][cls]
        sources.append((cls, draws.choice(clips),
                        round(draws.uniform(*LEVEL_RANGE_DB), 3)))

    noise_class, noise_clips = noise_pool
    noise_class = draws.choice(noise_class)
    noise_file = draws.choice(noise_clips[noise_class])

    # Sum the targets at their drawn levels.
    n = None
    targets = None
    for cls, name, level_db in sources:
        x = read_wav(os.path.join(audio_dir, name))
        if n is None:
            n = x.size
            targets = np.zeros(n)
        x = fit_length(x, n) / rms(x)
        targets += x * (10.0 ** (level_db / 20.0))

    # Add background noise at the drawn SNR.
    noise = fit_length(read_wav(os.path.join(audio_dir, noise_file)), n)
    noise = noise / rms(noise)
    target_rms = rms(targets)
    mixture = targets + noise * (target_rms / (10.0 ** (snr_db / 20.0)))

    # One scalar over the whole mixture: leaves SNR and abundances untouched.
    peak = float(np.max(np.abs(mixture)))
    peak_scale = OUTPUT_PEAK / peak if peak > 0 else 1.0
    mixture = mixture * peak_scale

    powers = np.array([10.0 ** (lv / 10.0) for _, _, lv in sources])
    fractions = powers / powers.sum()
    abundance = {cls: 0.0 for cls in label_space}
    for (cls, _, _), frac in zip(sources, fractions):
        abundance[cls] += float(frac)

    record = {
        "k": k,
        "snr_db": snr_db,
        "sources": ";".join("%s|%s|%.3f" % s for s in sources),
        "noise_class": noise_class,
        "noise_file": noise_file,
        "peak_scale": peak_scale,
        "abundance": abundance,
    }
    return mixture, record


# --------------------------------------------------------------------------
# One split
# --------------------------------------------------------------------------

def generate_split(split, corpus, audio_dir, out_root, cache):
    folds, primary_classes, secondary_classes, split_id = SPLITS[split]
    label_space = IN_SCOPE if split != "test_late" else IN_SCOPE + LATE_ADDITION

    secondary_clips = {c: eligible(corpus, audio_dir, c, folds, cache)
                       for c in secondary_classes}
    noise_clips = {c: eligible(corpus, audio_dir, c, folds, cache)
                   for c in NOISE_BANK}

    primaries = []
    for cls in primary_classes:
        for name in eligible(corpus, audio_dir, cls, folds, cache):
            primaries.append((cls, name))
    primaries.sort()
    # Each primary clip seeds several queries, so that the nine (k, SNR) cells
    # of the breakdown hold enough queries for the differences between them to
    # mean something.
    primaries = [(cls, name) for (cls, name) in primaries
                 for _ in range(REPEATS_PER_PRIMARY)]

    # Assign the nine (k, SNR) cells evenly across the queries, then shuffle.
    # An i.i.d. draw would leave the cells of the breakdown in section 3.5
    # unevenly filled, which matters most on the small test_late split.
    cells = [(k, snr) for k in K_CHOICES for snr in SNR_CHOICES_DB]
    order = Draws(MASTER_SEED, split_id, "schedule").permutation(len(primaries))
    schedule = [cells[order[i] % len(cells)] for i in range(len(primaries))]

    wav_dir = os.path.join(out_root, split)
    os.makedirs(wav_dir, exist_ok=True)

    audio_hash = hashlib.sha256()
    rows, abundance_rows = [], []

    for idx, (cls, name) in enumerate(primaries):
        draws = Draws(MASTER_SEED, split_id, idx)
        pools = {
            "primary": (cls, name),
            "secondary_classes": secondary_classes,
            "secondary_clips": secondary_clips,
        }
        k, snr_db = schedule[idx]
        mixture, rec = build_query(
            draws, audio_dir, pools, (NOISE_BANK, noise_clips), label_space,
            k, float(snr_db))

        query_id = "%s_%05d" % (split, idx)
        pcm = write_wav(os.path.join(wav_dir, query_id + ".wav"), mixture)
        audio_hash.update(pcm.tobytes())

        rows.append({
            "query_id": query_id,
            "wav": "%s/%s.wav" % (split, query_id),
            "k": rec["k"],
            "snr_db": "%.1f" % rec["snr_db"],
            "sources": rec["sources"],
            "noise_class": rec["noise_class"],
            "noise_file": rec["noise_file"],
            "peak_scale": "%.6f" % rec["peak_scale"],
        })
        abundance_rows.append(
            [query_id] + ["%.6f" % rec["abundance"][c] for c in label_space])

    manifest = os.path.join(out_root, "manifest_%s.csv" % split)
    write_csv(manifest, ["query_id", "wav", "k", "snr_db", "sources",
                         "noise_class", "noise_file", "peak_scale"],
              [[r[c] for c in ("query_id", "wav", "k", "snr_db", "sources",
                               "noise_class", "noise_file", "peak_scale")]
               for r in rows])

    abundance_path = os.path.join(out_root, "abundance_%s.csv" % split)
    write_csv(abundance_path, ["query_id"] + list(label_space), abundance_rows)

    template = os.path.join(out_root, "predictions_template_%s.csv" % split)
    write_csv(template, ["query_id"] + list(label_space),
              [[r[0]] + ["0.000000"] * len(label_space) for r in abundance_rows])

    return {
        "split": split,
        "n_queries": len(rows),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest),
        "audio_sha256": audio_hash.hexdigest(),
    }


def write_csv(path, header, rows):
    """Fixed newline and quoting, so the bytes are identical on every platform."""
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_library_manifests(corpus, out_root):
    """List the clean fold-1 clips that form the reference library.

    The in-scope and late-addition libraries are written separately, so that
    the late-addition classes stay out of development by construction.
    """
    for name, classes in (("library", IN_SCOPE), ("library_late", LATE_ADDITION)):
        rows = []
        for cls in classes:
            for fname in corpus[cls].get(LIBRARY_FOLD, []):
                rows.append([cls, "audio/%s" % fname])
        path = os.path.join(out_root, "manifest_%s.csv" % name)
        write_csv(path, ["category", "path"], rows)
        print("  %-24s %4d clips  %s" % (name, len(rows), path))


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esc50-root", required=True,
                    help="directory containing audio/ and meta/esc50.csv")
    ap.add_argument("--out", default="./field_queries",
                    help="output directory (default: ./field_queries)")
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "test_late"],
                    choices=sorted(SPLITS), help="splits to generate")
    args = ap.parse_args()

    audio_dir = os.path.join(args.esc50_root, "audio")
    if not os.path.isdir(audio_dir):
        raise SystemExit("Cannot find %s." % audio_dir)
    os.makedirs(args.out, exist_ok=True)

    corpus = load_corpus(args.esc50_root)
    cache = {}

    print("make_field_queries.py   seed=%d" % MASTER_SEED)
    print("Reference library (fold %d, clean, not degraded):" % LIBRARY_FOLD)
    write_library_manifests(corpus, args.out)

    print("Field queries:")
    summaries = []
    for split in args.splits:
        s = generate_split(split, corpus, audio_dir, args.out, cache)
        summaries.append(s)
        print("  %-24s %4d queries" % (s["split"], s["n_queries"]))

    lines = ["make_field_queries.py  seed=%d" % MASTER_SEED,
             "python %s  numpy %s" % (sys.version.split()[0], np.__version__),
             "",
             "The manifest checksum is the one to report. It is derived from the",
             "recipe alone. The audio checksum may differ in the last bit on a",
             "different CPU, which is audibly irrelevant and not a problem.",
             ""]
    for s in summaries:
        lines += [
            "split          %s" % s["split"],
            "queries        %d" % s["n_queries"],
            "manifest       sha256:%s" % s["manifest_sha256"],
            "audio          sha256:%s" % s["audio_sha256"],
            "",
        ]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "checksums.txt"), "w",
              newline="\n", encoding="utf-8") as fh:
        fh.write(text)

    print("\n" + "-" * 68)
    print(text.rstrip())
    print("-" * 68)
    print("Report the manifest checksums above in your documentation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
