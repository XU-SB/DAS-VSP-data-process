"""
Phase 0 — DAS-VSP diagnostic script
Tasks 0.1 (data QC), 0.2 (effective wavelet), 0.3 (bandwidth + f-k audit),
0.4 (manifest).

Usage:
    python phase0_diagnostics.py --data_dir /path/to/data --pilot_dir /path/to/FWI

All outputs go to ./phase0_outputs/.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, fftconvolve

OUT = "phase0_outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit or pass via --data_dir / --pilot_dir)
# ─────────────────────────────────────────────────────────────────────────────
N_TRACES  = 883
N_SAMPLES = 5000
USE_SAMPLES = 2000
DT        = 0.001          # s
CH_SPACING = 1.0           # m  (ASSUMED — verify with Dr. Kimura)

# Ormsby-bandwidth of the data (actual, post-decon)
F_LOW_TAPER  = 8.0         # Hz  ramp-on start
F_LOW_PASS   = 10.0        # Hz  ramp-on end
F_HIGH_PASS  = 70.0        # Hz  ramp-off start
F_HIGH_TAPER = 80.0        # Hz  ramp-off end


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def amp_spectrum(x, dt):
    """One-sided amplitude spectrum, returns (freqs, amplitudes)."""
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], d=dt)
    return f, np.abs(X)


def cosine_taper_bandpass(nt, dt, f1, f2, f3, f4):
    """
    Cosine-ramped band-pass in the frequency domain.
    f1-f2 : low-end ramp (0 -> 1);  f3-f4 : high-end ramp (1 -> 0).
    Returns filter array of length nt//2+1 (rfft frequencies).
    """
    f = np.fft.rfftfreq(nt, d=dt)
    H = np.zeros_like(f)
    for i, fi in enumerate(f):
        if fi < f1:
            H[i] = 0.0
        elif fi < f2:
            H[i] = 0.5 * (1 - np.cos(np.pi * (fi - f1) / (f2 - f1)))
        elif fi <= f3:
            H[i] = 1.0
        elif fi <= f4:
            H[i] = 0.5 * (1 + np.cos(np.pi * (fi - f3) / (f4 - f3)))
        else:
            H[i] = 0.0
    return H


def apply_bp(data, dt, f1, f2, f3, f4):
    """Apply cosine-tapered bandpass to last axis of data."""
    H = cosine_taper_bandpass(data.shape[-1], dt, f1, f2, f3, f4)
    return np.fft.irfft(np.fft.rfft(data, axis=-1) * H, n=data.shape[-1], axis=-1)


def save_fig(name):
    p = os.path.join(OUT, name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0.1 — load + QC
# ─────────────────────────────────────────────────────────────────────────────

def task01_load(data_path):
    print("\n=== Task 0.1: Load & QC ===")
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: data file not found: {data_path}\n"
                 "  Set --data_dir to the directory containing "
                 "stack.run3.decon.geo_ch15_median.bin")

    raw = np.fromfile(data_path, dtype=np.float32)
    expected = N_TRACES * N_SAMPLES
    if raw.size != expected:
        sys.exit(f"ERROR: expected {expected} floats ({N_TRACES}×{N_SAMPLES}), "
                 f"got {raw.size}")

    data = raw.reshape(N_TRACES, N_SAMPLES)[:, :USE_SAMPLES]
    nt = data.shape[1]

    print(f"  n_channels  = {N_TRACES}")
    print(f"  nt (trimmed)= {nt}")
    print(f"  dt          = {DT} s")
    print(f"  ch_spacing  = {CH_SPACING} m  (ASSUMED)")
    print(f"  data range  = [{data.min():.4g}, {data.max():.4g}]")
    print(f"  Inferred total depth coverage: {(N_TRACES-1)*CH_SPACING:.0f} m")
    print(f"  Inferred record length: {(nt-1)*DT:.3f} s")

    t  = np.arange(nt) * DT
    ch = np.arange(N_TRACES)
    vmax = np.percentile(np.abs(data), 98)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data.T, aspect="auto", cmap="seismic",
                   origin="upper", vmin=-vmax, vmax=vmax,
                   extent=[ch[0], ch[-1], t[-1], t[0]])
    ax.set_xlabel("Channel (depth index)")
    ax.set_ylabel("Time (s)")
    ax.set_title("DAS-VSP shot gather  stack.run3.decon.geo_ch15_median\n"
                 "(trimmed to 2000 samples)")
    plt.colorbar(im, ax=ax, label="Amplitude")
    save_fig("task01_gather.png")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0.2 — effective wavelet extraction
# ─────────────────────────────────────────────────────────────────────────────

def _pick_first_breaks_simple(data, dt, vel_apparent=2000.0, z_src=0.0,
                               ch_spacing=CH_SPACING):
    """
    Approximate first-break picks by energy onset on each trace.
    Returns sample indices (float).
    """
    nt = data.shape[1]
    env = np.abs(data)                  # simple envelope proxy
    picks = []
    for tr in env:
        # scan forward; pick when energy > threshold
        threshold = 0.1 * tr.max()
        idx = np.argmax(tr > threshold)
        picks.append(float(idx))
    return np.array(picks)


def task02_wavelet(data, pilot_path, dt=DT):
    print("\n=== Task 0.2: Effective-wavelet extraction ===")

    nt = data.shape[1]
    nch = data.shape[0]

    # ── 1. Approximate first-break picks ──────────────────────────────────
    picks = _pick_first_breaks_simple(data, dt)
    print(f"  First-break range: sample {picks.min():.0f} – {picks.max():.0f}  "
          f"({picks.min()*dt:.3f}s – {picks.max()*dt:.3f}s)")

    # ── 2. Corridor mute: keep ±corridor samples around direct arrival ────
    corridor = 60   # samples (±0.06 s)
    muted = np.zeros_like(data)
    for i, p in enumerate(picks):
        lo = max(0, int(p) - corridor)
        hi = min(nt, int(p) + corridor)
        muted[i, lo:hi] = data[i, lo:hi]

    # ── 3. Align traces to t=0 of direct arrival, stack to get wavelet ───
    aligned = np.zeros((nch, 2 * corridor))
    for i, p in enumerate(picks):
        lo = max(0, int(p) - corridor)
        hi = min(nt, int(p) + corridor)
        seg = data[i, lo:hi]
        pad_l = max(0, corridor - int(p))
        pad_r = 2 * corridor - len(seg) - pad_l
        aligned[i] = np.pad(seg, (pad_l, max(0, pad_r)))[:2*corridor]

    wavelet = aligned.mean(axis=0)
    wavelet /= (np.abs(wavelet).max() + 1e-12)

    out_wav = os.path.join(OUT, "effective_wavelet.npy")
    np.save(out_wav, wavelet.astype(np.float32))
    print(f"  Effective wavelet saved: {out_wav}  (length={len(wavelet)} samples)")

    # ── 4. Load raw pilot for comparison ─────────────────────────────────
    if not os.path.exists(pilot_path):
        print(f"  WARNING: pilot not found at {pilot_path}; skipping comparison")
        pilot = None
    else:
        pilot_raw = np.fromfile(pilot_path, dtype=np.float32)
        pilot = pilot_raw[:len(wavelet)].astype(np.float64)
        pilot /= (np.abs(pilot).max() + 1e-12)

    # ── 5. Phase analysis ────────────────────────────────────────────────
    W = np.fft.rfft(wavelet)
    f_wav = np.fft.rfftfreq(len(wavelet), d=dt)
    phase_wav = np.angle(W)

    # Zero-phase criterion: median |phase| in the passband
    band = (f_wav >= 10) & (f_wav <= 70)
    med_phase = np.median(np.abs(phase_wav[band]))
    phase_label = "near-zero-phase" if med_phase < 0.5 else "mixed-phase"
    print(f"  Median |phase| in 10–70 Hz: {np.degrees(med_phase):.1f}°  → {phase_label}")

    # ── 6. Bandwidth check ───────────────────────────────────────────────
    f_w, A_w = amp_spectrum(wavelet, dt)
    A_db = 20 * np.log10(A_w / (A_w.max() + 1e-30) + 1e-30)
    # find -20 dB cutoffs
    peak = A_db.max()
    cutoff = peak - 20
    above = f_w[A_db > cutoff]
    bw_lo = above[0] if len(above) else 0
    bw_hi = above[-1] if len(above) else 0
    print(f"  Effective wavelet -20 dB bandwidth: {bw_lo:.1f} – {bw_hi:.1f} Hz")

    # ── 7. Plots ─────────────────────────────────────────────────────────
    t_wav = (np.arange(len(wavelet)) - corridor) * dt * 1000  # ms

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time domain
    axes[0, 0].plot(t_wav, wavelet, label="Effective wavelet")
    if pilot is not None:
        tp = np.arange(len(pilot)) * dt * 1000
        axes[0, 0].plot(tp[:len(t_wav)], pilot[:len(t_wav)],
                        alpha=0.6, label="Raw pilot")
    axes[0, 0].set_xlabel("Time (ms)"); axes[0, 0].set_ylabel("Norm. amplitude")
    axes[0, 0].set_title("Time domain"); axes[0, 0].legend(); axes[0, 0].grid(True)

    # Amplitude spectrum
    axes[0, 1].plot(f_wav, 20*np.log10(A_w / (A_w.max()+1e-30) + 1e-30),
                    label="Effective wavelet")
    if pilot is not None:
        fp, Ap = amp_spectrum(pilot[:len(wavelet)], dt)
        axes[0, 1].plot(fp, 20*np.log10(Ap/(Ap.max()+1e-30)+1e-30),
                        alpha=0.6, label="Raw pilot")
    axes[0, 1].axvline(10, color='g', ls='--', lw=0.8, label="10 Hz")
    axes[0, 1].axvline(70, color='r', ls='--', lw=0.8, label="70 Hz")
    axes[0, 1].set_xlim(0, 150); axes[0, 1].set_ylim(-40, 2)
    axes[0, 1].set_xlabel("Frequency (Hz)"); axes[0, 1].set_ylabel("dB")
    axes[0, 1].set_title("Amplitude spectrum (dB)"); axes[0, 1].legend(); axes[0, 1].grid(True)

    # Phase spectrum (wavelet)
    axes[1, 0].plot(f_wav, np.degrees(phase_wav), label="Effective wavelet")
    axes[1, 0].axvline(10, color='g', ls='--', lw=0.8)
    axes[1, 0].axvline(70, color='r', ls='--', lw=0.8)
    axes[1, 0].set_xlim(0, 150); axes[1, 0].set_ylim(-200, 200)
    axes[1, 0].set_xlabel("Frequency (Hz)"); axes[1, 0].set_ylabel("Phase (°)")
    axes[1, 0].set_title(f"Phase spectrum  ({phase_label})"); axes[1, 0].grid(True)

    # Aligned corridor stack
    vmax = np.percentile(np.abs(aligned), 95)
    axes[1, 1].imshow(aligned.T, aspect="auto", cmap="seismic",
                      vmin=-vmax, vmax=vmax)
    axes[1, 1].set_xlabel("Channel"); axes[1, 1].set_ylabel("Sample (re: first break)")
    axes[1, 1].set_title("Corridor-aligned direct arrivals")

    plt.suptitle("Task 0.2 — Effective wavelet vs raw pilot", fontsize=12)
    plt.tight_layout()
    save_fig("task02_wavelet.png")

    return wavelet, picks


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0.3 — bandwidth + f-k separation audit
# ─────────────────────────────────────────────────────────────────────────────

def task03_fk_audit(data, picks, dt=DT, ch_spacing=CH_SPACING):
    print("\n=== Task 0.3: Bandwidth + f-k separation audit ===")

    nt   = data.shape[1]
    nch  = data.shape[0]
    t    = np.arange(nt) * dt
    ch   = np.arange(nch) * ch_spacing

    # ── 1. Spectral roll-off of raw gather ──────────────────────────────
    mean_trace = data.mean(axis=0)
    f_all, A_all = amp_spectrum(mean_trace, dt)
    i70 = np.argmin(np.abs(f_all - 70))
    ipeak = np.argmax(A_all)
    print(f"  Stack spectrum peak at {f_all[ipeak]:.1f} Hz")
    A_at_70 = A_all[i70] / (A_all[ipeak] + 1e-30)
    print(f"  Relative amplitude at 70 Hz vs peak: {20*np.log10(A_at_70+1e-30):.1f} dB")

    # ── 2. Direct-arrival moveout sign ──────────────────────────────────
    # Fit a line through first-break picks: slope = dt/dchannel
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(np.arange(nch), picks * dt, 1)
    slope_s_per_ch = coeffs[0]   # s per channel
    slope_ms_per_m = slope_s_per_ch / ch_spacing * 1000
    print(f"  Direct-arrival moveout: {slope_ms_per_m:.3f} ms/m  "
          f"({'POSITIVE = downgoing ✓' if slope_ms_per_m > 0 else 'NEGATIVE = upgoing?'})")
    # Downgoing = traveltime increases with channel depth → slope > 0 → depth_positive_down=True
    correct_sign = slope_ms_per_m > 0
    print(f"  Implied fk_keep_direction(depth_positive_down="
          f"{'True' if correct_sign else 'False'}) for upgoing separation")

    # ── 3. F-K transform ────────────────────────────────────────────────
    FK = np.fft.fftshift(np.fft.fft2(data))
    f_fk = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    k_fk = np.fft.fftshift(np.fft.fftfreq(nch, d=ch_spacing))

    # Apparent velocity fans
    # Downgoing: f/k > 0  (f and k same sign); upgoing: f/k < 0
    F2D, K2D = np.meshgrid(f_fk, k_fk, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        Vapp = np.where(np.abs(K2D) > 1e-6, F2D / K2D, np.sign(F2D) * 1e9)

    # Upgoing mask: Vapp < 0 (or f>0, k<0 quadrant)
    mask_up = (Vapp < 0) & (np.abs(f_fk[:, None]) > 0)

    FK_up   = FK * mask_up
    FK_down = FK * ~mask_up

    data_up   = np.real(np.fft.ifft2(np.fft.ifftshift(FK_up)))
    data_down = np.real(np.fft.ifft2(np.fft.ifftshift(FK_down)))

    # Check: is the direct arrival suppressed in upgoing?
    # Compare energy in corridor around first breaks
    energy_full_direct  = 0.0
    energy_up_direct    = 0.0
    for i, p in enumerate(picks):
        lo = max(0, int(p) - 30); hi = min(nt, int(p) + 30)
        energy_full_direct += np.sum(data[i, lo:hi]**2)
        energy_up_direct   += np.sum(data_up[i, lo:hi]**2)

    suppression_dB = -10 * np.log10(energy_up_direct / (energy_full_direct + 1e-30) + 1e-30)
    print(f"  Direct-arrival energy suppression in upgoing field: {suppression_dB:.1f} dB")
    if suppression_dB < 10:
        print("  WARNING: direct arrival NOT well suppressed — "
              "check depth_positive_down sign")

    # ── 4. Plots ─────────────────────────────────────────────────────────
    vmax_d = np.percentile(np.abs(data), 98)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    def pgather(ax, d, title):
        ax.imshow(d.T, aspect="auto", cmap="seismic",
                  origin="upper", vmin=-vmax_d, vmax=vmax_d,
                  extent=[0, nch-1, t[-1], t[0]])
        ax.set_xlabel("Channel"); ax.set_ylabel("Time (s)")
        ax.set_title(title)

    pgather(axes[0], data,      "Full (with direct)")
    pgather(axes[1], data_down, "Downgoing (f-k)")
    pgather(axes[2], data_up,   "Upgoing (f-k)")

    # F-K power plot
    FK_db = 20 * np.log10(np.abs(FK).T + 1e-30)
    FK_db -= FK_db.max()
    axes[3].imshow(FK_db[::-1], aspect="auto", cmap="hot",
                   vmin=-40, vmax=0,
                   extent=[k_fk[0], k_fk[-1], f_fk[0], f_fk[-1]])
    axes[3].set_xlabel("Wavenumber (1/m)"); axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_ylim(-150, 150); axes[3].set_xlim(-0.5, 0.5)
    axes[3].set_title("F-K power (dB)")

    plt.suptitle("Task 0.3 — Bandwidth + f-k audit", fontsize=12)
    plt.tight_layout()
    save_fig("task03_fk_audit.png")

    # ── 5. Spectrum plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, d in [("Full", data), ("Upgoing", data_up), ("Downgoing", data_down)]:
        f_, A_ = amp_spectrum(d.mean(axis=0), dt)
        ax.plot(f_, 20*np.log10(A_/(A_.max()+1e-30)+1e-30), label=label)
    ax.axvline(10, color='g', ls='--', lw=0.8, label="10 Hz"); ax.axvline(70, color='r', ls='--', lw=0.8, label="70 Hz")
    ax.set_xlim(0, 150); ax.set_ylim(-50, 2)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("dB")
    ax.set_title("Spectra: full vs f-k separated fields"); ax.legend(); ax.grid(True)
    save_fig("task03_spectra.png")

    print(f"  correct depth_positive_down for upgoing separation: {correct_sign}")
    return data_up, data_down, correct_sign


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0.4 — manifest
# ─────────────────────────────────────────────────────────────────────────────

def task04_manifest(wavelet, correct_sign, data_dir, pilot_path):
    print("\n=== Task 0.4: Manifest ===")

    manifest = {
        "observable_type": (
            "PENDING_DR_KIMURA — mixed: (DAS strain-rate response) / "
            "(geophone velocity response); source signature removed by decon. "
            "NOT clean pressure, NOT clean v_z, NOT clean strain-rate."
        ),
        "decon_type": "PENDING_DR_KIMURA — cross-correlation or spectral division?",
        "reference_trace": "geo_ch15 — physical location PENDING_DR_KIMURA",
        "stack_definition": "PENDING_DR_KIMURA — shots? time windows?",
        "gauge_length_L_m": "UNKNOWN — REQUIRED from Dr. Kimura",
        "channel_spacing_m": CH_SPACING,
        "channel_spacing_note": "ASSUMED 1 m — verify with Dr. Kimura",
        "n_channels": N_TRACES,
        "nt_original": N_SAMPLES,
        "nt_trimmed": USE_SAMPLES,
        "dt_s": DT,
        "record_length_s": (USE_SAMPLES - 1) * DT,
        "bandwidth_Hz": {
            "low_ramp_start": F_LOW_TAPER,
            "low_ramp_end": F_LOW_PASS,
            "high_ramp_start": F_HIGH_PASS,
            "high_ramp_end": F_HIGH_TAPER,
            "filter_type": "Ormsby (cosine taper)",
        },
        "source_geometry": {
            "x_m": 182.3,
            "z_m": 1.0,
            "note": "Assumes outcome (A): near-surface effective source. "
                    "PENDING_DR_KIMURA confirmation."
        },
        "receiver_geometry": {
            "x_m": 200.0,
            "z_m_range": [1.0, 883.0],
            "n_receivers": 883,
            "layout": "vertical well, 1 m spacing"
        },
        "grid": {
            "dx_m": 1.0, "dz_m": 1.0,
            "nx": 551, "nz": 901,
            "x_range_m": [0, 550],
            "z_range_m": [0, 900],
            "model_padding": [100, 100]
        },
        "effective_wavelet_path": os.path.join(OUT, "effective_wavelet.npy"),
        "effective_wavelet_length_samples": len(wavelet),
        "raw_pilot_path": pilot_path,
        "fk_separation": {
            "correct_depth_positive_down": correct_sign,
            "method": "2D FFT with f/k sign mask (no hard velocity bounds)"
        },
        "known_bugs_to_fix": [
            "Brick-wall 10-150 Hz -> replace with cosine-tapered 8/10/70/80 Hz",
            "Double downgoing removal (align+median AND f-k upgoing) -> collapse to one",
            "First-pick index convention inconsistency (trace-1 vs no offset)",
            "FWI scale factor computed but not applied -> fold into wavelet or data",
            "Dead code in RTM: forward(vp2d_true) and gaussian_filter(60) unused"
        ],
        "open_questions_for_dr_kimura": [
            "Is decon a cross-correlation or spectral division?",
            "Which trace is geo_ch15 (geophone #15 or co-located DAS channel)? Physical location?",
            "What is the stack over (shots / time windows)?",
            "What is the gauge length L? (REQUIRED for Phase 3 DAS operator)",
            "Confirm channel spacing is 1 m"
        ]
    }

    out_path = os.path.join(OUT, "manifest.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Manifest saved: {out_path}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="./Data",
                        help="Directory containing stack.run3.decon.geo_ch15_median.bin")
    parser.add_argument("--pilot_dir", default="../FWI",
                        help="Directory containing source_pilot_run3_raw.bin")
    args = parser.parse_args()

    data_path  = os.path.join(args.data_dir,  "stack.run3.decon.geo_ch15_median.bin")
    pilot_path = os.path.join(args.pilot_dir, "source_pilot_run3_raw.bin")

    data               = task01_load(data_path)
    wavelet, picks     = task02_wavelet(data, pilot_path)
    data_up, data_down, correct_sign = task03_fk_audit(data, picks)
    manifest           = task04_manifest(wavelet, correct_sign, args.data_dir, pilot_path)

    print("\n=== Phase 0 complete ===")
    print(f"Outputs in ./{OUT}/")
    print("\nCRITICAL BLOCKERS before Phase 1:")
    for q in manifest["open_questions_for_dr_kimura"]:
        print(f"  • {q}")


if __name__ == "__main__":
    main()
