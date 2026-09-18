#!/usr/bin/env python3
"""
panns_encoder.py
PANNs CNN14 feature extraction for the Esper cross-condition retrieval task.

Why CNN14: it is trained on AudioSet, which is multi-label and recorded in the
wild, so it has had to represent several overlapping sources at once. That is
the condition our queries are in. It is fully convolutional, so it accepts
variable-length input and we can embed short windows without retraining. It is
small and fast enough to embed the whole dataset in a couple of minutes on a
T4. The checkpoint is AudioSet-only, never fine-tuned on ESC-50, so folds 4
and 5 have not been seen.

What this module gives you:
  resample_to()          exact rational resampling 44100 -> 32000, no librosa
  frame_windows()        split a clip into overlapping windows + their RMS
  PannsEncoder.embed()   (N, 2048) clip embeddings
  PannsEncoder.embed_windowed()  (N, W, 2048) window embeddings + (N, W) RMS
  EmbeddingCache         npz-backed cache so you never recompute

Design notes:
  - Every waveform in a batch must be the same length. ESC-50 clips are all
    five seconds and windows are all the same size, so this costs nothing and
    it removes an entire class of padding bugs. Mixing lengths raises.
  - Embeddings are returned unnormalised. Call l2norm() where you want it.
  - Nothing here is trained. Adding a class means embedding its clips.

Install:
    pip install torch panns-inference scipy numpy
The checkpoint (~320 MB) downloads on first use to ~/panns_data/.

Probe (Probe A from the model-selection plan):
    python panns_encoder.py --esc50-root ./ESC-50 --probe
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ESC50_RATE = 44100
PANNS_RATE = 32000
EMBED_DIM = 2048


# ---------------------------------------------------------------------------
# Pure numpy helpers. No torch import here, so these are testable on their own.
# ---------------------------------------------------------------------------

def read_wav(path: str) -> np.ndarray:
    """Read a mono 16-bit PCM wav as float64 in [-1, 1)."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit wav: %s" % path)
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def resample_to(x: np.ndarray, sr_in: int = ESC50_RATE,
                sr_out: int = PANNS_RATE) -> np.ndarray:
    """Polyphase resample using the exact rational ratio.

    44100 -> 32000 reduces to 320/441, so this is exact and fast. We resample
    for the model only; the data on disk stays untouched.
    """
    if sr_in == sr_out:
        return x.astype(np.float64)
    f = Fraction(sr_out, sr_in)
    return resample_poly(x, f.numerator, f.denominator).astype(np.float64)


def frame_windows(x: np.ndarray, sr: int, window_s: float = 1.0,
                  hop_s: float = 0.5, pad: bool = True):
    """Split a signal into overlapping windows.

    Returns (windows, rms) with shapes (W, n) and (W,). The RMS is kept so the
    caller can energy-gate or energy-weight, which matters here: median energy
    duty cycle on this dataset is about 0.26, so most windows of most queries
    contain no target at all.
    """
    n = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if n <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    if x.size < n:
        if not pad:
            raise ValueError("signal shorter than one window")
        x = np.concatenate([x, np.zeros(n - x.size)])

    n_win = 1 + (x.size - n) // hop
    tail = x.size - ((n_win - 1) * hop + n)
    if pad and tail > 0:
        x = np.concatenate([x, np.zeros(n - tail)])
        n_win += 1

    idx = np.arange(n)[None, :] + (np.arange(n_win) * hop)[:, None]
    win = x[idx]
    rms = np.sqrt((win ** 2).mean(axis=1))
    return win, rms


def l2norm(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def energy_duty(x: np.ndarray, frame: int = 2048) -> float:
    """Participation ratio of frame energy. Threshold free.

    1.0 means energy is spread evenly, 1/N means it is all in one frame.
    """
    n = (x.size // frame) * frame
    if n == 0:
        return 1.0
    p = (x[:n].reshape(-1, frame) ** 2).mean(axis=1)
    s = p.sum()
    if s <= 0:
        return 1.0
    return float(s ** 2 / (len(p) * (p ** 2).sum()))


# ---------------------------------------------------------------------------
# Checkpoint bootstrap
#
# panns_inference downloads its two data files with os.system('wget ...').
# There is no wget on Windows, so the call fails silently and the package then
# raises FileNotFoundError at import time. We fetch both files ourselves with
# urllib, which works everywhere, and verify the checkpoint against its
# published hash so a truncated download fails loudly instead of quietly
# loading garbage weights.
# ---------------------------------------------------------------------------

PANNS_DATA_DIR = Path(os.environ.get("PANNS_DATA_DIR", str(Path.home()))) / "panns_data"

LABELS_NAME = "class_labels_indices.csv"
LABELS_URL = ("http://storage.googleapis.com/us_audioset/youtube_corpus/"
              "v1/csv/class_labels_indices.csv")

CKPT_NAME = "Cnn14_mAP=0.431.pth"
CKPT_URLS = [
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP%3D0.431.pth",
]
CKPT_SHA256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"


def _download(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "esper-challenge/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print("\r  %s  %5.1f%%" % (dest.name, 100 * done / total),
                      end="", flush=True)
    print()
    shutil.move(str(tmp), str(dest))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_panns_data(verify: bool = False) -> str:
    """Make sure the labels CSV and the CNN14 checkpoint are on disk.

    Returns the checkpoint path. Safe to call repeatedly; it is a no-op once
    both files exist. A freshly downloaded checkpoint is always hash-checked;
    pass verify=True to re-check one that was already there (a few seconds).
    """
    PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    labels = PANNS_DATA_DIR / LABELS_NAME
    if not labels.exists():
        print("fetching AudioSet label index ->", labels)
        _download(LABELS_URL, labels)

    ckpt = PANNS_DATA_DIR / CKPT_NAME
    fresh = False
    if not ckpt.exists():
        print("fetching CNN14 checkpoint (327 MB) ->", ckpt)
        last = None
        for url in CKPT_URLS:
            try:
                _download(url, ckpt)
                fresh = True
                break
            except Exception as exc:          # try the mirror
                last = exc
                print("  failed: %s" % exc)
        if not ckpt.exists():
            raise RuntimeError(
                "could not download %s. Fetch it by hand from "
                "https://zenodo.org/record/3987831 and put it at %s. Last error: %s"
                % (CKPT_NAME, ckpt, last))

    if verify or fresh:
        got = _sha256(ckpt)
        if got != CKPT_SHA256:
            raise RuntimeError(
                "checkpoint hash mismatch at %s\n  expected %s\n  got      %s\n"
                "Delete the file and re-run." % (ckpt, CKPT_SHA256, got))
        print("checkpoint sha256 verified")

    return str(ckpt)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PannsEncoder:
    """Frozen PANNs CNN14. Loads lazily so importing this module is cheap."""

    def __init__(self, checkpoint: str | None = None, device: str | None = None,
                 batch_size: int = 32, source_rate: int = ESC50_RATE):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.source_rate = source_rate
        self._device = device
        self._model = None

    # -- loading ------------------------------------------------------------

    @property
    def device(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        import torch
        # Fetch the label index and checkpoint ourselves. This must happen
        # before the import below, because panns_inference reads the label CSV
        # at import time and raises if it is missing.
        ckpt_path = self.checkpoint or ensure_panns_data()

        from panns_inference.models import Cnn14
        model = Cnn14(sample_rate=PANNS_RATE, window_size=1024,
                      hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                      classes_num=527)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model = model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # -- embedding ----------------------------------------------------------

    def _forward(self, batch_32k: np.ndarray) -> np.ndarray:
        """batch_32k: (B, n) float array already at 32 kHz. Returns (B, 2048)."""
        import torch
        x = torch.from_numpy(np.ascontiguousarray(batch_32k, dtype=np.float32))
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x, None)
        return out["embedding"].detach().cpu().numpy().astype(np.float64)

    def embed(self, waveforms, source_rate: int | None = None) -> np.ndarray:
        """Embed whole clips. waveforms: (N, n) array or list of equal-length 1-D.

        Returns (N, 2048), unnormalised.
        """
        sr = source_rate or self.source_rate
        arr = self._stack(waveforms)
        res = np.stack([resample_to(w, sr, PANNS_RATE) for w in arr])
        outs = []
        for i in range(0, res.shape[0], self.batch_size):
            outs.append(self._forward(res[i:i + self.batch_size]))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, EMBED_DIM))

    def embed_windowed(self, waveforms, window_s: float = 1.0, hop_s: float = 0.5,
                       source_rate: int | None = None):
        """Embed overlapping windows of each clip.

        Returns (emb, rms) with shapes (N, W, 2048) and (N, W). The RMS is
        measured on the resampled signal, which is what the model sees.
        """
        sr = source_rate or self.source_rate
        arr = self._stack(waveforms)
        res = [resample_to(w, sr, PANNS_RATE) for w in arr]

        framed, rmss = [], []
        for w in res:
            fw, fr = frame_windows(w, PANNS_RATE, window_s, hop_s)
            framed.append(fw)
            rmss.append(fr)
        n_win = framed[0].shape[0]
        if any(f.shape[0] != n_win for f in framed):
            raise ValueError("clips of differing length produced differing window counts")

        flat = np.concatenate(framed, axis=0)          # (N*W, n)
        outs = []
        for i in range(0, flat.shape[0], self.batch_size):
            outs.append(self._forward(flat[i:i + self.batch_size]))
        emb = np.concatenate(outs, axis=0).reshape(len(res), n_win, EMBED_DIM)
        return emb, np.stack(rmss)

    @staticmethod
    def _stack(waveforms) -> np.ndarray:
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 2:
            return waveforms
        lens = {np.asarray(w).size for w in waveforms}
        if len(lens) != 1:
            raise ValueError(
                "all waveforms in one call must be the same length; got %s. "
                "Pad or crop first: batching different lengths silently changes "
                "the result." % sorted(lens))
        return np.stack([np.asarray(w, dtype=np.float64).ravel() for w in waveforms])


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Disk cache keyed by a hash of (file list, settings). Cheap and blunt."""

    def __init__(self, root: str = "./cache"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def key(self, tag: str, items, **settings) -> str:
        h = hashlib.sha256()
        h.update(tag.encode())
        for it in items:
            h.update(str(it).encode())
        for k in sorted(settings):
            h.update(("%s=%s" % (k, settings[k])).encode())
        return os.path.join(self.root, "%s_%s.npz" % (tag, h.hexdigest()[:16]))

    def get(self, path: str):
        if os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        return None

    def put(self, path: str, **arrays):
        np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Probe A: clean ceiling. Fold-1 prototypes, clean fold-2/3 queries.
# ---------------------------------------------------------------------------

IN_SCOPE = [
    "dog", "rooster", "pig", "cow", "cat", "hen", "sheep", "frog",
    "door_wood_knock", "mouse_click", "keyboard_typing", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking", "siren", "car_horn", "train",
]


def load_meta(esc50_root: str):
    import csv
    path = os.path.join(esc50_root, "meta", "esc50.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def probe_clean_ceiling(esc50_root: str, encoder: PannsEncoder,
                        window_s: float = 1.0, hop_s: float = 0.5):
    """Nearest-prototype top-1 on clean, unmixed clips.

    This is the ceiling: no mixing, no added noise, no domain gap. If the
    encoder cannot do this, nothing downstream will work. Reports clip-level
    and max-over-window scoring side by side.
    """
    meta = load_meta(esc50_root)
    audio = os.path.join(esc50_root, "audio")
    rows = [r for r in meta if r["category"] in IN_SCOPE]
    lib = [r for r in rows if r["fold"] == "1"]
    qry = [r for r in rows if r["fold"] in ("2", "3")]

    def embed_set(recs):
        waves = [read_wav(os.path.join(audio, r["filename"])) for r in recs]
        n = min(w.size for w in waves)
        waves = np.stack([w[:n] for w in waves])
        clip = encoder.embed(waves)
        win, rms = encoder.embed_windowed(waves, window_s, hop_s)
        return clip, win, rms

    lib_clip, lib_win, _ = embed_set(lib)
    qry_clip, qry_win, _ = embed_set(qry)
    lib_y = np.array([IN_SCOPE.index(r["category"]) for r in lib])
    qry_y = np.array([IN_SCOPE.index(r["category"]) for r in qry])

    # clip-level prototypes
    P = np.stack([l2norm(l2norm(lib_clip)[lib_y == c]).mean(axis=0)
                  for c in range(len(IN_SCOPE))])
    P = l2norm(P)
    acc_clip = (l2norm(qry_clip) @ P.T).argmax(axis=1) == qry_y

    # window-level prototypes, max over query windows
    lw = l2norm(lib_win).reshape(-1, EMBED_DIM)
    lw_y = np.repeat(lib_y, lib_win.shape[1])
    Pw = l2norm(np.stack([lw[lw_y == c].mean(axis=0) for c in range(len(IN_SCOPE))]))
    sims = l2norm(qry_win) @ Pw.T                     # (N, W, C)
    acc_win = sims.max(axis=1).argmax(axis=1) == qry_y

    return {
        "n_library": len(lib), "n_query": len(qry),
        "top1_clip": float(acc_clip.mean()),
        "top1_window_max": float(acc_win.mean()),
        "per_class_clip": {IN_SCOPE[c]: float(acc_clip[qry_y == c].mean())
                           for c in range(len(IN_SCOPE))},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50-root", default="./ESC-50")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--hop-s", type=float, default=0.5)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fetch", action="store_true",
                    help="download and verify the checkpoint, then exit")
    ap.add_argument("--verify", action="store_true",
                    help="re-check the sha256 of an existing checkpoint")
    args = ap.parse_args()

    if args.fetch:
        path = ensure_panns_data(verify=True)
        print("ready:", path)
        return

    enc = PannsEncoder(checkpoint=args.checkpoint, device=args.device)
    if args.verify:
        ensure_panns_data(verify=True)
    if not args.probe:
        print("loaded PANNs CNN14 on %s. Pass --probe to run the clean ceiling."
              % enc.device)
        return

    r = probe_clean_ceiling(args.esc50_root, enc, args.window_s, args.hop_s)
    print("Probe A: clean ceiling, no mixing, no added noise")
    print("  library clips %d   query clips %d" % (r["n_library"], r["n_query"]))
    print("  top-1, clip-level prototypes      %.3f" % r["top1_clip"])
    print("  top-1, max over %.1fs windows      %.3f"
          % (args.window_s, r["top1_window_max"]))
    print("\n  per class (clip-level), worst first")
    for c, a in sorted(r["per_class_clip"].items(), key=lambda kv: kv[1]):
        print("    %-18s %.3f" % (c, a))


if __name__ == "__main__":
    main()