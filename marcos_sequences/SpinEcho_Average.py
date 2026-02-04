#!/usr/bin/env python3
"""
spin_echo_average.py

Standalone script to acquire N spin-echo repeats at one LO frequency,
align & phase-correct each shot, compute coherent and magnitude averages,
and save standard diagnostic plots.

How to use:
- Edit the SETTINGS block below to match your hardware / experiment parameters.
- Run: python spin_echo_average.py
- Outputs are saved in out_dir (PNG + a .npz with raw buffers and metadata).

Notes:
- This script expects the same Experiment class / API used in SpinEcho_Sweep.py.
  If your Experiment import path differs, edit the import below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import argparse
import json
from experiment import Experiment   # same import used in SpinEcho_Sweep.py

# -------------------------
# SETTINGS - EDIT THESE
# -------------------------
OUT_DIR = "averages_out"         # where plots + data are written
OUT_PREFIX = "avg_run1"          # prefix for files inside OUT_DIR
LO_FREQ_MHZ = 2.113             # choose the resonant LO you found
RX_T_US = 0.30                   # RX time used when creating Experiment (seconds in µs)
VOLTS_PER_UNIT = 0.00845         # scale factor if you use same as SpinEcho_Sweep.py
N_REPS = 64                      # number of repeats to average (1,2,4,... recommended)
LPF_CUT_HZ = 5000.0              # optional LPF applied in postprocessing (None to disable)
BUTTER_ORDER = 4
# Pulse params: match what you used when you identified the resonance.
PULSE90_US = 300.0
PULSE180_US = 600.0
TAU_US = 600.0
BLANK2_US = 100.0
ACQ_US = 1500.0

# Save raw buffers (npz). Set to True to keep them for offline analysis.
SAVE_RAW = True

# -------------------------
# Utilities
# -------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def find_echo_offset(ref_mag, mag, max_shift_samples=None):
    if max_shift_samples is None:
        max_shift = min(len(ref_mag)//2, 2000)
    else:
        max_shift = int(max_shift_samples)
    corr = sig.correlate(mag, ref_mag, mode='full')
    lags = np.arange(-len(ref_mag)+1, len(mag))
    center_idx = np.argmax(np.abs(corr))
    shift = lags[center_idx]
    shift = int(np.clip(shift, -max_shift, max_shift))
    return shift

def apply_shift_and_pad(x, shift):
    L = len(x)
    y = np.zeros_like(x)
    if shift >= 0:
        if shift < L:
            y[shift:] = x[:L-shift]
    else:
        s = -shift
        if s < L:
            y[:L-s] = x[s:]
    return y

def per_echo_phase_correct(ref, trace, center_sample=None, phase_fit_len=64):
    if center_sample is None:
        center_sample = np.argmax(np.abs(ref))
    L = len(ref)
    i0 = max(0, center_sample - phase_fit_len//2)
    i1 = min(L, center_sample + phase_fit_len//2)
    r_seg = ref[i0:i1]
    t_seg = trace[i0:i1]
    if np.sum(np.abs(t_seg)**2) < 1e-20:
        return 1.0+0j
    z = np.sum(np.conj(t_seg) * r_seg)
    ph = np.angle(z)
    return np.exp(1j * ph)

# -------------------------
# Acquisition wrapper
# -------------------------
def acquire_raw_echo(
    expt,
    lo_freq_MHz=LO_FREQ_MHZ,
    rf90_amplitude=0.9,
    rf180_amplitude=0.9,
    pulse90_us=PULSE90_US,
    pulse180_us=PULSE180_US,
    tau_us=TAU_US,
    blank2_us=BLANK2_US,
    acq_us=ACQ_US,
    volts_per_unit=VOLTS_PER_UNIT
):
    """
    Configure the Experiment for a single spin echo and return (complex_buffer, Ts_us, Fs).
    """
    expt.set_lo_freq(lo_freq_MHz)
    expt._seq = None

    t0 = 0.0
    t90_end    = t0 + pulse90_us
    t180_start = t90_end + tau_us
    t180_end   = t180_start + pulse180_us

    rx_start   = t180_end + max(blank2_us, 0.0)
    rx_times   = np.array([rx_start, rx_start + acq_us], dtype=float)
    rx_gate    = np.array([1, 0], dtype=int)

    tx_times = np.array([t0, t90_end, t180_start, t180_end], dtype=float)
    tx_amps  = np.array([rf90_amplitude, 0.0, rf180_amplitude, 0.0], dtype=float)

    seq = {
        "tx0":    (tx_times, tx_amps + 0j),
        "rx0_en": (rx_times, rx_gate),
    }
    expt.add_flodict(seq, append=False)
    rxd, msgs = expt.run()

    data = np.asarray(rxd.get("rx0", np.array([], dtype=complex)))
    if data.size == 0:
        raise RuntimeError("No RX0 data returned from device.")
    Ts_us = float(expt.get_rx_ts()[0])
    Ts_s = Ts_us * 1e-6
    Fs = 1.0 / Ts_s
    if volts_per_unit is not None:
        data = data * float(volts_per_unit)
    return data, Ts_us, Fs

# -------------------------
# Averaging main
# -------------------------
def run_average(
    out_dir=OUT_DIR,
    out_prefix=OUT_PREFIX,
    lo_freq_MHz=LO_FREQ_MHZ,
    rx_t_us=RX_T_US,
    n_reps=N_REPS,
    lpf_cut_hz=LPF_CUT_HZ,
    butter_order=BUTTER_ORDER,
    save_raw=SAVE_RAW
):
    ensure_dir(out_dir)
    out_prefix_full = os.path.join(out_dir, out_prefix)

    # Create Experiment. If your Experiment requires different args, edit here.
    expt = Experiment(lo_freq=lo_freq_MHz, rx_t=rx_t_us, init_gpa=False, auto_leds=True, flush_old_rx=True)

    print(f"Collecting {n_reps} echoes at {lo_freq_MHz:.6f} MHz ...")
    buff_list = []
    Ts_us = None
    Fs = None

    # Acquire n_reps
    for k in range(n_reps):
        data, Ts_us, Fs = acquire_raw_echo(expt, lo_freq_MHz=lo_freq_MHz)
        buff_list.append(np.asarray(data))
        print(f"  Acquired {k+1}/{n_reps}  (N_samples={data.size})")

    L = buff_list[0].size
    raw = np.vstack([np.pad(b, (0, max(0, L - b.size)), mode='constant')[:L] for b in buff_list])

    # Optional LPF on each shot (same used in SpinEcho_Sweep.py for consistency)
    Iq = raw.copy()
    if lpf_cut_hz is not None:
        wn = min(lpf_cut_hz / (Fs/2.0), 0.999)
        b, a = sig.butter(butter_order, wn, btype='low')
        padlen = max(3*(max(len(b), len(a))), 0)
        for k in range(n_reps):
            # zero-phase filter
            Iq[k,:] = sig.filtfilt(b, a, raw[k,:], padlen=padlen)

    # Reference for alignment (use first shot)
    ref = Iq[0,:].copy()
    ref_mag = np.abs(ref)

    # Align + phase correct + accumulate
    acc_complex = np.zeros(L, dtype=complex)
    acc_mag_incoherent = np.zeros(L, dtype=float)  # store magnitude-average (incoherent)
    aligned_shots = np.zeros((n_reps, L), dtype=complex)

    for k in range(n_reps):
        shot = Iq[k,:].copy()
        # alignment on magnitude
        shift = find_echo_offset(ref_mag, np.abs(shot))
        shot = apply_shift_and_pad(shot, shift)
        # phase correction to reference
        ph = per_echo_phase_correct(ref, shot, center_sample=None, phase_fit_len=64)
        shot_ph = shot * ph
        aligned_shots[k,:] = shot_ph
        acc_complex += shot_ph
        acc_mag_incoherent += np.abs(shot_ph)

    avg_complex = acc_complex / float(n_reps)
    avg_mag_coherent = np.abs(avg_complex)
    avg_mag_incoherent = acc_mag_incoherent / float(n_reps)

    t = np.arange(L) * Ts_us  # µs axis

    # Compute simple SNR metric using echo window logic (same formula as SpinEcho_Sweep.py)
    t_echo_rx_us = 0.5 * PULSE90_US + TAU_US - BLANK2_US
    echo_window_us = 150.0
    i0 = int(max(0, (t_echo_rx_us - echo_window_us)/Ts_us))
    i1 = int(min(L-1, (t_echo_rx_us + echo_window_us)/Ts_us))

    # Peak and noise estimates for coherent average
    echo_peak_coh = float(np.max(avg_mag_coherent[i0:i1])) if i1>i0 else float(np.max(avg_mag_coherent))
    # Pre-echo noise window
    j1 = max(0, i0 - (i1 - i0))
    j0 = max(0, j1 - (i1 - i0))
    if j1 > j0:
        noise_rms_coh = float(np.sqrt(np.mean(avg_mag_coherent[j0:j1]**2)))
    else:
        noise_rms_coh = float(np.sqrt(np.mean(avg_mag_coherent**2)))
    snr_coh = echo_peak_coh / (noise_rms_coh + 1e-12)

    # Peak and SNR for incoherent (magnitude) average
    echo_peak_incoh = float(np.max(avg_mag_incoherent[i0:i1])) if i1>i0 else float(np.max(avg_mag_incoherent))
    if j1 > j0:
        noise_rms_incoh = float(np.sqrt(np.mean(avg_mag_incoherent[j0:j1]**2)))
    else:
        noise_rms_incoh = float(np.sqrt(np.mean(avg_mag_incoherent**2)))
    snr_incoh = echo_peak_incoh / (noise_rms_incoh + 1e-12)

    # Save results + summary
    meta = dict(
        lo_freq_MHz=lo_freq_MHz,
        Ts_us=Ts_us,
        Fs=Fs,
        n_reps=n_reps,
        lpf_cut_hz=lpf_cut_hz,
        pulse90_us=PULSE90_US,
        pulse180_us=PULSE180_US,
        tau_us=TAU_US,
        blank2_us=BLANK2_US,
        acq_us=ACQ_US,
        volts_per_unit=VOLTS_PER_UNIT,
        echo_center_rx_us=t_echo_rx_us,
        i0=i0, i1=i1
    )

    # ---------- Plots ----------
    ensure_dir(out_dir)

    plt.figure(figsize=(10,4))
    plt.plot(t, avg_complex.real, label="Real")
    plt.plot(t, avg_complex.imag, label="Imag", alpha=0.7)
    plt.title(f"Averaged Complex (N={n_reps}) - {lo_freq_MHz:.6f} MHz")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix_full}_complex_avg.png", dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(t, avg_mag_coherent, label="Coherent avg (|<I+jQ>|)")
    plt.plot(t, avg_mag_incoherent, label="Incoherent avg (⟨|I+jQ|⟩)", alpha=0.8)
    plt.title(f"Averaged Magnitude (N={n_reps})  | SNR_coh≈{snr_coh:.2f}  SNR_incoh≈{snr_incoh:.2f}")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix_full}_mag_avg.png", dpi=150); plt.close()

    # zoom on echo
    plt.figure(figsize=(8,4))
    tz = t[i0:i1]
    plt.plot(tz, avg_mag_coherent[i0:i1], label="Coherent")
    plt.plot(tz, avg_mag_incoherent[i0:i1], label="Incoherent", alpha=0.8)
    plt.title("Averaged Echo (zoom)")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix_full}_mag_avg_zoom.png", dpi=150); plt.close()

    # SNR vs N (build incremental averages)
    Ns = np.arange(1, n_reps+1)
    snrs_coh = []
    snrs_incoh = []
    for m in Ns:
        avgm_coh = np.abs(np.mean(aligned_shots[:m, :], axis=0))
        avgm_incoh = np.mean(np.abs(aligned_shots[:m, :]), axis=0)
        peak_coh = np.max(avgm_coh[i0:i1])
        peak_incoh = np.max(avgm_incoh[i0:i1])
        if j1>j0:
            nrm_coh = np.sqrt(np.mean(avgm_coh[j0:j1]**2))
            nrm_incoh = np.sqrt(np.mean(avgm_incoh[j0:j1]**2))
        else:
            nrm_coh = np.sqrt(np.mean(avgm_coh**2))
            nrm_incoh = np.sqrt(np.mean(avgm_incoh**2))
        snrs_coh.append(peak_coh/(nrm_coh+1e-12))
        snrs_incoh.append(peak_incoh/(nrm_incoh+1e-12))

    plt.figure(figsize=(8,4))
    plt.plot(Ns, snrs_coh, 'o-', label="Coherent")
    plt.plot(Ns, snrs_incoh, 's--', label="Incoherent")
    plt.plot(Ns, np.sqrt(Ns)*snrs_coh[0]/np.sqrt(1), '--', alpha=0.4, label="√N ref (scaled)")
    plt.xlabel("Number of averages N"); plt.ylabel("Estimated SNR_peak")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix_full}_snr_vs_N.png", dpi=150); plt.close()

    # PSD (ASD) before and after averaging - use first shot and coherent avg
    nper = min(8192, L)
    f_raw, Pxx_raw = sig.welch(raw[0,:].real, fs=Fs, nperseg=nper)
    # combine I & Q if complex; here signal stored complex as complex numbers
    # Use magnitude ASD as quick diagnostic (consistent with SpinEcho_Sweep.py)
    f_raw, PxxI = sig.welch(raw[0,:].real, fs=Fs, nperseg=nper)
    _, PxxQ = sig.welch(raw[0,:].imag, fs=Fs, nperseg=nper)
    ASD_raw = np.sqrt(PxxI + PxxQ)

    f_avg, PxxIavg = sig.welch(avg_complex.real, fs=Fs, nperseg=nper)
    _, PxxQavg = sig.welch(avg_complex.imag, fs=Fs, nperseg=nper)
    ASD_avg = np.sqrt(PxxIavg + PxxQavg)

    plt.figure(figsize=(10,4))
    plt.semilogy(f_raw/1e3, ASD_raw, label="Single-shot (first)")
    plt.semilogy(f_avg/1e3, ASD_avg, label="Coherent avg")
    if lpf_cut_hz is not None:
        plt.axvline(lpf_cut_hz/1e3, ls="--", alpha=0.6)
    plt.title("Amplitude spectral density (V/√Hz)")
    plt.xlabel("Frequency (kHz)"); plt.ylabel("ASD (V/√Hz)")
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix_full}_asd.png", dpi=150); plt.close()

    # Save raw buffers + metadata for offline reprocessing
    if save_raw:
        np.savez_compressed(
            f"{out_prefix_full}_raw.npz",
            raw=raw,
            aligned_shots=aligned_shots,
            avg_complex=avg_complex,
            avg_mag_coherent=avg_mag_coherent,
            avg_mag_incoherent=avg_mag_incoherent,
            t=t,
            meta=meta
        )
    # Save summary JSON
    summary = dict(
        snr_coherent=snr_coh,
        snr_incoherent=snr_incoh,
        echo_peak_coherent=echo_peak_coh,
        echo_peak_incoherent=echo_peak_incoh,
        noise_rms_coherent=noise_rms_coh,
        noise_rms_incoherent=noise_rms_incoh,
        meta=meta
    )
    with open(f"{out_prefix_full}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Done. Outputs written to:", out_dir)
    print(f"  Coherent SNR ≈ {snr_coh:.2f}  | Incoherent SNR ≈ {snr_incoh:.2f}")
    try:
        expt.close_server()
    except Exception:
        pass

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--out_prefix", default=OUT_PREFIX)
    parser.add_argument("--lo", type=float, default=LO_FREQ_MHZ, help="LO frequency MHz")
    parser.add_argument("--n", type=int, default=N_REPS)
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()

    run_average(out_dir=args.out_dir, out_prefix=args.out_prefix, lo_freq_MHz=args.lo, n_reps=args.n, save_raw=args.save_raw)
