import os
import sys
import json
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram

try:
    import sounddevice as sd
except ImportError:
    print("WARNING: 'sounddevice' not installed. Audio playback disabled.")
    print("  Install with:  pip install sounddevice")
    sd = None

# ==============================
# USER SETTINGS
# ==============================

HDF5_DIR   = "/mnt/SSD3/zf_vocal_development_hdf5"
WAV_DIR    = "/mnt/SSD3/zf_vocal_development/audio"
OUTPUT_DIR = "./development_plots_interactive"

# Leave BIRDS empty to auto-detect from WAV_DIR and pick one at random.
# Set to a list like ["b3g20"] to override.
BIRDS = []

DPH_MIN = 35
DPH_MAX = 90
N_SAMPLES = 15
DPH_RANGE = np.logspace(np.log10(DPH_MIN), np.log10(DPH_MAX), N_SAMPLES).astype(int)[::-1]

# ==============================
# SPECTROGRAM PARAMETERS (Sepp et al.)
# ==============================
SAMPLE_RATE = 32_000
NPERSEG     = 512
STEP_WIDTH  = 128
NOVERLAP    = NPERSEG - STEP_WIDTH   # = 384
LOG_M       = 1e-6

FREQ_MIN = 100
FREQ_MAX = 10_000
MAX_CANDIDATES = 200

SPEC_PARAMS = {
    "nperseg":     NPERSEG,
    "noverlap":    NOVERLAP,
    "step_width":  STEP_WIDTH,
    "log_m":       LOG_M,
    "freq_min_hz": FREQ_MIN,
    "freq_max_hz": FREQ_MAX,
    "window":      "hann",
    "scaling":     "spectrum",
    "normalisation": "rms",
    "colormap":    "viridis",
}

# ---- Viewer display settings ----
DISPLAY_BATCH_SIZE = 10  # <-- spectrograms per page

# ==============================
# Bird Discovery
# ==============================

def discover_birds(wav_dir):
    """List bird IDs from subdirectories of WAV_DIR."""
    if not os.path.isdir(wav_dir):
        print(f"ERROR: WAV_DIR not found: {wav_dir}")
        sys.exit(1)
    birds = sorted([
        d for d in os.listdir(wav_dir)
        if os.path.isdir(os.path.join(wav_dir, d))
    ])
    if not birds:
        print(f"ERROR: No bird folders found in {wav_dir}")
        sys.exit(1)
    return birds


# ==============================
# Utilities
# ==============================

def compute_dph(dob_str, rec_date_str):
    dob = datetime.strptime(dob_str, "%Y-%m-%d")
    rec = datetime.strptime(rec_date_str, "%Y-%m-%d")
    return (rec - dob).days


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio /= np.abs(audio).max()
    if audio.ndim > 1:
        audio = audio[:, 0]
    return sr, audio


def get_candidate_clips(date_group):
    """Returns list of (clip_name, snr) sorted by SNR descending."""
    metadata   = date_group["metadata"]
    candidates = []
    for clip_name in metadata.keys():
        meta = metadata[clip_name]
        if meta.attrs.get("is_song", 0) != 1:
            continue
        if meta.attrs.get("tutor_present", "1") != "0":
            continue
        try:
            snr = float(meta["audio_quality_metrics"]["vad_dependent"].attrs.get("snr_db", -np.inf))
        except Exception:
            snr = -np.inf
        candidates.append((clip_name, snr))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:MAX_CANDIDATES]


def compute_spec(audio, sr):
    """Log-spectrogram per Sepp et al.: x_log = 20*log(m + |X|^2), m=1."""
    audio = audio - np.mean(audio)
    audio = audio / (np.sqrt(np.mean(audio**2)) + 1e-10)

    f, t, Sxx = scipy_spectrogram(audio, fs=sr,
                                   nperseg=NPERSEG,
                                   noverlap=NOVERLAP,
                                   window='hann',
                                   scaling='spectrum')
    Sxx_log = 20 * np.log(LOG_M + Sxx)

    freq_mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    return f[freq_mask], t, Sxx_log[freq_mask]


def save_wav(path, audio, sr):
    wav_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(path, sr, wav_int16)


def save_spec_params(bird_output):
    """Write spectrogram parameters to the bird output folder once."""
    params_path = os.path.join(bird_output, "spectrogram_params.json")
    if not os.path.exists(params_path):
        with open(params_path, "w") as fh:
            json.dump(SPEC_PARAMS, fh, indent=2)
        print(f"  Saved spectrogram params → {params_path}")


# ==============================
# Audio Playback
# ==============================

PLAYBACK_SR = 44_100   # standard rate that ALSA will accept

def _resample(audio, sr_in, sr_out):
    """Simple linear-interpolation resample. Good enough for preview."""
    if sr_in == sr_out:
        return audio
    ratio = sr_out / sr_in
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    idx = np.clip(indices.astype(int), 0, len(audio) - 1)
    return audio[idx]


def play_audio(audio, sr):
    """Non-blocking audio playback at a device-friendly sample rate."""
    if sd is None:
        return
    sd.stop()
    resampled = _resample(audio, sr, PLAYBACK_SR)
    sd.play(resampled, PLAYBACK_SR)


def stop_audio():
    if sd is None:
        return
    sd.stop()


def play_segment(audio, sr, onset, offset):
    """Play a single onset-offset segment."""
    start = int(onset * sr)
    end   = int(offset * sr)
    play_audio(audio[start:end], sr)


# ==============================
# Interactive Batch Viewer
# ==============================

def pick_clip_interactively(specs, bird_id, actual_dph, target_dph):
    """
    Single-column paged viewer.  x-axis = longest clip in the current batch.
    Colour scale fixed across the full candidate set.

    Keys:  [0–9] select  [n/→] next  [p/←] prev  [k] skip DPH  [q] quit
    """
    n_total  = len(specs)
    vmin_all = min(Sxx_log.min() for _, _, _, Sxx_log in specs)
    vmax_all = max(np.percentile(Sxx_log, 99) for _, _, _, Sxx_log in specs)

    batch_start = 0
    result = {"choice": None, "action": "continue"}

    while True:
        batch_end = min(batch_start + DISPLAY_BATCH_SIZE, n_total)
        batch     = specs[batch_start:batch_end]
        n_batch   = len(batch)
        max_dur   = max(t[-1] for _, _, t, _ in batch)

        fig, axes = plt.subplots(n_batch, 1, figsize=(10, 2.8 * n_batch), squeeze=False)

        for i, (clip_name, f, t, Sxx_log) in enumerate(batch):
            ax = axes[i][0]
            ax.imshow(Sxx_log, aspect="auto", origin="lower",
                      extent=[t[0], t[-1], f[0] / 1000, f[-1] / 1000],
                      interpolation="none", cmap="viridis",
                      vmin=vmin_all, vmax=vmax_all)
            ax.set_xlim(0, max_dur)
            ax.set_ylabel("kHz", fontsize=8)
            ax.set_title(f"[{batch_start + i}]  {clip_name}  ({t[-1]:.2f} s)",
                         fontsize=9, loc="left", pad=3)
            ax.tick_params(left=True, bottom=False, labelbottom=False, labelsize=7)

        axes[-1][0].tick_params(bottom=True, labelbottom=True)
        axes[-1][0].set_xlabel("Time (s)", fontsize=8)

        n_batches = int(np.ceil(n_total / DISPLAY_BATCH_SIZE))
        cur_batch = batch_start // DISPLAY_BATCH_SIZE + 1
        fig.suptitle(
            f"{bird_id}  ·  DPH {actual_dph} (target {target_dph})"
            f"   Batch {cur_batch}/{n_batches}  —  x-axis: {max_dur:.2f} s\n"
            "[0–9] select  [n/→] next batch  [p/←] prev batch  [k] skip DPH  [q] quit",
            fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        def on_key(event):
            nonlocal batch_start
            key = event.key
            if key in [str(i) for i in range(10)]:
                idx = batch_start + int(key)
                if idx < n_total:
                    result["choice"] = idx
                    result["action"] = "selected"
                    plt.close(fig)
            elif key in ("n", "right"):
                if batch_start + DISPLAY_BATCH_SIZE < n_total:
                    batch_start += DISPLAY_BATCH_SIZE
                plt.close(fig)
            elif key in ("p", "left"):
                batch_start = max(0, batch_start - DISPLAY_BATCH_SIZE)
                plt.close(fig)
            elif key == "k":
                result["action"] = "skip"
                plt.close(fig)
            elif key == "q":
                result["action"] = "quit"
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if result["action"] in ("selected", "skip", "quit"):
            break

    return result


# ==============================
# Segmentation UI (with playback)
# ==============================

def segment_clip(clip_name, f, t, Sxx_log, vmin, vmax, audio, sr):
    """
    Show a single spectrogram and let the user click onset/offset pairs.
    Plays the segment audio each time a pair is completed.

    Click sequence:   onset₁ → offset₁ → onset₂ → offset₂ → …
    Onsets  = odd  clicks (1st, 3rd, …) — green dashed line
    Offsets = even clicks (2nd, 4th, …) — red   dashed line

    Keys:
        u / backspace   : undo last click
        Enter / d       : confirm segments and close
        Escape          : discard all segments and close
        a               : play full clip
        space           : play last completed segment
        x               : stop audio
    """
    segments  = []   # list of {"onset": float, "offset": float}
    clicks    = []   # raw time values as user clicks

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.imshow(Sxx_log, aspect="auto", origin="lower",
              extent=[t[0], t[-1], f[0] / 1000, f[-1] / 1000],
              interpolation="none", cmap="viridis",
              vmin=vmin, vmax=vmax)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("kHz", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(
        f"Segmenting: {clip_name}\n"
        "Click: odd = onset (green), even = offset (red)  |  "
        "[u/⌫] undo  [Enter/d] confirm  [Esc] discard\n"
        "[a] play clip  [space] play last seg  [x] stop audio",
        fontsize=8)

    lines = []   # vertical line artists

    def redraw_lines():
        for ln in lines:
            ln.remove()
        lines.clear()
        for k, tx in enumerate(clicks):
            color  = "#44ff88" if k % 2 == 0 else "#ff4444"
            ln = ax.axvline(tx, color=color, linestyle="--", linewidth=1.2)
            lines.append(ln)
        # shade completed segments
        for patch in list(ax.patches):
            patch.remove()
        for seg in segments:
            ax.axvspan(seg["onset"], seg["offset"],
                       alpha=0.15, color="#88aaff", zorder=0)
        fig.canvas.draw()

    result = {"segments": [], "confirmed": False}

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        tx = event.xdata
        clicks.append(tx)
        # if we just completed an onset-offset pair, record it and play it
        if len(clicks) % 2 == 0:
            onset  = clicks[-2]
            offset = clicks[-1]
            if onset > offset:
                onset, offset = offset, onset
            segments.append({"onset": round(onset, 5), "offset": round(offset, 5)})
            play_segment(audio, sr, onset, offset)
        redraw_lines()

    def on_key(event):
        key = event.key
        if key in ("u", "backspace"):
            if clicks:
                clicks.pop()
                if segments and len(clicks) < len(segments) * 2:
                    segments.pop()
                redraw_lines()
        elif key in ("enter", "d"):
            result["segments"]  = list(segments)
            result["confirmed"] = True
            stop_audio()
            plt.close(fig)
        elif key == "escape":
            result["segments"]  = []
            result["confirmed"] = False
            stop_audio()
            plt.close(fig)
        elif key == "a":
            play_audio(audio, sr)
        elif key == " ":
            if segments:
                seg = segments[-1]
                play_segment(audio, sr, seg["onset"], seg["offset"])
        elif key == "x":
            stop_audio()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)
    plt.tight_layout()
    plt.show()

    return result["segments"] if result["confirmed"] else []


# ==============================
# Final Figure: Segmented Portions Only
# ==============================

def build_segment_spec(audio, sr, segments):
    """
    Concatenate the segmented audio portions, compute a single
    spectrogram of the result, and return (f, t, Sxx_log, seg_boundaries).

    seg_boundaries is a list of time values in the concatenated signal
    where each original segment begins, useful for drawing dividers.
    """
    if not segments:
        return None

    pieces = []
    boundaries = [0.0]
    running_t = 0.0

    for seg in segments:
        start = int(seg["onset"] * sr)
        end   = int(seg["offset"] * sr)
        chunk = audio[start:end]
        pieces.append(chunk)
        running_t += len(chunk) / sr
        boundaries.append(running_t)

    concat = np.concatenate(pieces)
    if len(concat) < NPERSEG:
        # Too short to spectrogram — pad with silence
        concat = np.pad(concat, (0, NPERSEG - len(concat)))

    f, t, Sxx_log = compute_spec(concat, sr)
    return f, t, Sxx_log, boundaries


def save_final_figure(bird_id, selected_specs, selected_dph, selected_segs, output_dir):
    """
    Build the development figure showing ONLY the segmented portions
    (concatenated per DPH row) with boundary markers.
    Earliest DPH (lowest) at top, latest at bottom.
    All rows share an x-axis scaled to the longest clip.
    Saved as a high-resolution PDF with rasterized spectrograms.
    """
    if not selected_specs:
        return

    from matplotlib.backends.backend_pdf import PdfPages

    # Sort by DPH ascending (earliest at top)
    order = sorted(range(len(selected_dph)), key=lambda i: selected_dph[i])
    specs_sorted = [selected_specs[i] for i in order]
    dph_sorted   = [selected_dph[i]   for i in order]
    segs_sorted  = [selected_segs[i]  for i in order]

    n = len(specs_sorted)

    # Find the longest concatenated clip duration across all rows
    max_dur = max(t[-1] for (_, t, _, _) in specs_sorted)

    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), squeeze=False)

    for row, (ax_row, (f, t, Sxx_log, boundaries), dph_val, seg_list) in enumerate(
            zip(axes, specs_sorted, dph_sorted, segs_sorted)):
        ax = ax_row[0]

        vmin = Sxx_log.min()
        vmax = np.percentile(Sxx_log, 99)

        # Rasterize the spectrogram so the PDF stays a reasonable size
        ax.imshow(Sxx_log, aspect="auto", origin="lower",
                  extent=[t[0], t[-1], f[0] / 1000, f[-1] / 1000],
                  interpolation="none", cmap="viridis",
                  vmin=vmin, vmax=vmax,
                  rasterized=True)

        # Lock x-axis to the longest clip so nothing gets stretched
        ax.set_xlim(0, max_dur)

        # Draw segment boundary dividers (skip first=0 and last=end)
        for b in boundaries[1:-1]:
            ax.axvline(b, color="white", linewidth=0.6, linestyle=":", alpha=0.7)

        # Build subtitle with onset-offset times
        seg_str = "  ".join(
            f"[{s['onset']:.3f}–{s['offset']:.3f}s]" for s in seg_list
        )
        ax.set_ylabel("kHz", fontsize=8)
        ax.set_title(f"DPH {dph_val}   {seg_str}",
                     fontsize=7, loc="left", pad=2, family="monospace")
        ax.tick_params(left=True, bottom=False, labelbottom=False, labelsize=7)

    axes[-1][0].tick_params(bottom=True, labelbottom=True)
    axes[-1][0].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(f"{bird_id} — Segmented Vocal Development Profile",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    final_path = os.path.join(output_dir, f"{bird_id}_development_figure.pdf")
    with PdfPages(final_path) as pdf:
        pdf.savefig(fig, dpi=600)
    plt.close()
    print(f"\nSaved final figure (segments only): {final_path}")


# ==============================
# Main
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Bird selection ---
if BIRDS:
    birds_to_process = BIRDS
else:
    available = discover_birds(WAV_DIR)
    pick = random.choice(available)
    birds_to_process = [pick]
    print(f"Randomly selected bird: {pick}  (from {len(available)} available)")

quit_all = False

for bird_id in birds_to_process:
    if quit_all:
        break

    print(f"\n========== {bird_id} ==========")

    bird_output = os.path.join(OUTPUT_DIR, bird_id)
    os.makedirs(bird_output, exist_ok=True)
    save_spec_params(bird_output)

    selected_specs = []   # (f, t, Sxx_log, boundaries)  — segmented concat
    selected_dph   = []
    selected_segs  = []   # raw segment dicts per DPH

    hdf5_path = os.path.join(HDF5_DIR, f"{bird_id}.h5")
    if not os.path.isfile(hdf5_path):
        print(f"  HDF5 not found: {hdf5_path}, skipping.")
        continue

    with h5py.File(hdf5_path, "r") as hf:
        dob = hf.attrs["date_of_birth"]

        date_dph = {
            d: compute_dph(dob, d)
            for d in hf.keys()
            if not d.startswith("_")
        }

        for target_dph in DPH_RANGE:
            if quit_all:
                break

            print(f"\n--- Target DPH {target_dph} ---")

            valid_dates = [(d, dph) for d, dph in date_dph.items() if 30 <= dph <= 100]
            if not valid_dates:
                continue

            closest_date, actual_dph = min(valid_dates, key=lambda x: abs(x[1] - target_dph))
            print(f"Using date {closest_date} (DPH={actual_dph})")

            candidates = get_candidate_clips(hf[closest_date])
            if not candidates:
                print("No tutor-free song clips.")
                continue

            specs  = []
            audios = []
            srs    = []

            for clip_name, snr in candidates:
                wav_path = os.path.join(WAV_DIR, bird_id, closest_date, f"{clip_name}.wav")
                if not os.path.isfile(wav_path):
                    print(f"  WAV not found, skipping: {clip_name}")
                    continue
                sr, audio = load_wav(wav_path)
                f, t, Sxx_log = compute_spec(audio, sr)
                specs.append((clip_name, f, t, Sxx_log))
                audios.append(audio)
                srs.append(sr)

            if not specs:
                print("No WAVs found for candidates.")
                continue

            # Sort shortest → longest
            order  = sorted(range(len(specs)), key=lambda i: specs[i][2][-1])
            specs  = [specs[i]  for i in order]
            audios = [audios[i] for i in order]
            srs    = [srs[i]    for i in order]

            # Candidate-set colour range
            vmin_set = min(Sxx_log.min() for _, _, _, Sxx_log in specs)
            vmax_set = max(np.percentile(Sxx_log, 99) for _, _, _, Sxx_log in specs)

            result = pick_clip_interactively(specs, bird_id, actual_dph, target_dph)

            if result["action"] == "quit":
                print("Quitting.")
                quit_all = True
                break

            elif result["action"] == "skip":
                print(f"Skipped DPH {target_dph}.")
                continue

            elif result["action"] == "selected":
                idx = result["choice"]
                clip_name, f, t, Sxx_log = specs[idx]

                # ---- Segmentation (with audio playback) ----
                segments = segment_clip(clip_name, f, t, Sxx_log,
                                        vmin_set, vmax_set,
                                        audios[idx], srs[idx])

                # ---- Save full WAV (for reference) ----
                base_path = os.path.join(bird_output, f"DPH_{target_dph}")
                save_wav(f"{base_path}.wav", audios[idx], srs[idx])
                print(f"Saved DPH_{target_dph}.wav  (clip: {clip_name}, dur: {t[-1]:.2f}s)")

                # ---- Save individual segment WAVs ----
                for si, seg in enumerate(segments):
                    start = int(seg["onset"] * srs[idx])
                    end   = int(seg["offset"] * srs[idx])
                    seg_audio = audios[idx][start:end]
                    seg_path = f"{base_path}_seg{si}.wav"
                    save_wav(seg_path, seg_audio, srs[idx])
                    print(f"  Saved segment {si}: {seg['onset']:.3f}–{seg['offset']:.3f}s → {seg_path}")

                # ---- Save segment metadata ----
                seg_data = {
                    "clip_name":   clip_name,
                    "dph":         int(actual_dph),
                    "target_dph":  int(target_dph),
                    "duration_s":  float(t[-1]),
                    "segments":    segments,
                }
                with open(f"{base_path}_segments.json", "w") as fh:
                    json.dump(seg_data, fh, indent=2)
                print(f"Saved DPH_{target_dph}_segments.json  ({len(segments)} segment(s))")

                # ---- Build concatenated segment spectrogram for final figure ----
                seg_spec = build_segment_spec(audios[idx], srs[idx], segments)
                if seg_spec is not None:
                    selected_specs.append(seg_spec)
                    selected_dph.append(target_dph)
                    selected_segs.append(segments)

    # ---- Final development figure (segmented portions only) ----
    save_final_figure(bird_id, selected_specs, selected_dph, selected_segs, OUTPUT_DIR)

print("\nDone.")