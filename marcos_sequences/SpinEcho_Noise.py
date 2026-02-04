#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from experiment import Experiment
import scipy.signal as sig


def _compute_enbw_IIR(b, a, Fs, worN=1 << 16):
    """
    Effective Noise Bandwidth (ENBW) of an IIR in Hz:
      ENBW = (Fs/N) * sum(|H[k]|^2) / |H[0]|^2
    """
    w, h = sig.freqz(b, a, worN=worN, fs=Fs)
    # Avoid divide-by-zero if DC gain ~ 0 (shouldn't happen for LPF)
    h0 = np.abs(h[0])
    if h0 == 0:
        return (Fs / len(w)) * np.sum(np.abs(h) ** 2)
    return (Fs / len(w)) * np.sum(np.abs(h) ** 2) / (h0 ** 2)


def acquire_spin_echo(
    lo_freq_MHz=2.118,      # center at your Larmor
    rf90_amplitude=0.0,      # TX OFF for noise
    rf180_amplitude=0.0,     # TX OFF for noise
    pulse90_us=200.0,
    pulse180_us=600.0,
    tau_us=400.0,
    blank2_us=200.0,         # start RX after pulses
    acq_us=30_000.0,         # long capture to stabilize RMS
    rx_t_us=0.60,            # sample period (µs) -> Fs = 2.5 MS/s
    volts_per_unit=0.00846,  # your calibrated scale
    out_prefix="se",

    # ---- noise-analysis knobs ----
    target_bw_hz=10_000.0,   # LPF to this bandwidth (Δf target). Set None to skip LPF path
    notch_50Hz=False,        # optionally notch 50 Hz mains (and its 150/250 Hz harmonics)
    lna_gain_dB=38.0,
    lna_nf_dB=1.0,
    R_ohm=50.0,
    T_K=300.0,

    # ---- plotting knobs ----
    plot_unmix_to_rf=False,  # optional; mostly educational
):
    # --- build and run the experiment -------------------------------------------------
    expt = Experiment(
        lo_freq=lo_freq_MHz,
        rx_t=rx_t_us,
        init_gpa=False,
        auto_leds=True,
        flush_old_rx=True,   # drop any stale samples
    )

    # TX sequence (amplitudes are 0.0 so it's silent; keeps timing realistic)
    t0 = 0.0
    t90_end = t0 + pulse90_us
    t180_start = t90_end + tau_us
    t180_end = t180_start + pulse180_us

    tx_times = np.array([t0, t90_end, t180_start, t180_end], dtype=float)
    tx_amps = np.array([rf90_amplitude, 0.0, rf180_amplitude, 0.0], dtype=float)

    # RX enable: start after 180° pulse ring-down and acquire for acq_us
    rx_start = t180_end + max(blank2_us, 0.0)
    rx_times = np.array([rx_start, rx_start + acq_us], dtype=float)
    rx_gate = np.array([1, 0], dtype=int)

    seq = {
        "tx0": (tx_times, tx_amps + 0j),
        "rx0_en": (rx_times, rx_gate),
        "rx1_en": (rx_times, rx_gate),
    }

    expt.add_flodict(seq)

    try:
        rxd, msgs = expt.run()
    except Exception as e:
        print(f"Experiment run failed: {e}")
        return np.array([], dtype=complex)

    # Pull data
    data = np.asarray(rxd.get("rx0", np.array([], dtype=complex)))
    if data.size == 0:
        print("No RX0 data returned.")
        return data

    # Timing
    Ts_us = float(expt.get_rx_ts()[0])  # µs/sample on RX0
    Ts_s = Ts_us * 1e-6
    Fs = 1.0 / Ts_s                    # samples per second

    # Apply volts scaling
    if volts_per_unit is not None:
        data = data.astype(np.complex128) * float(volts_per_unit)
    else:
        data = data.astype(np.complex128)

    # Time axis for plots
    t_us = np.arange(data.size, dtype=np.float64) * Ts_us  # µs

    # ------------------------------- raw plots --------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t_us, data.real, label="Real")
    plt.plot(t_us, data.imag, label="Imag", alpha=0.7)
    plt.title("RX0 waveform (Noise capture)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_complex.png", dpi=150)

    plt.figure(figsize=(10, 4))
    plt.plot(t_us, data.real, label="Real")
    plt.title("RX0 Real waveform (Noise capture)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_Real.png", dpi=150)

    mag = np.abs(data)
    plt.figure(figsize=(10, 4))
    plt.plot(t_us, mag)
    plt.title("Noise Magnitude")
    plt.xlabel("Time (µs)")
    plt.ylabel("Voltage (V)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_mag.png", dpi=150)

    # Optional: "unmix" back to RF (not needed for noise, educational only)
    if plot_unmix_to_rf:
        freq_offset_Hz = lo_freq_MHz * 1e6
        t_s = np.arange(data.size, dtype=np.float64) * Ts_s
        data_raw = data * np.exp(1j * 2 * np.pi * freq_offset_Hz * t_s)
        plt.figure(figsize=(10, 4))
        plt.plot(t_us, data_raw.real, label="Real (unmixed)")
        plt.plot(t_us, data_raw.imag, label="Imag (unmixed)", alpha=0.7)
        plt.legend(); plt.title("Digitally shifted back to RF (optional)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"{out_prefix}_raw.png", dpi=150)

    # --------------------------- NOISE ANALYSIS ----------------------------
    # DC removal (per channel)
    I = data.real - np.mean(data.real)
    Q = data.imag - np.mean(data.imag)

    # Optional: mains notch (50 Hz and a couple harmonics)
    if notch_50Hz:
        for f0 in (50.0, 150.0, 250.0):
            try:
                bz, az = sig.iirnotch(w0=f0, Q=30, fs=Fs)
                I = sig.filtfilt(bz, az, I)
                Q = sig.filtfilt(bz, az, Q)
            except Exception:
                pass  # if Fs is low or edge case, skip gracefully

    # 1) RAW wideband complex RMS (complex stream -> Δf_raw = Fs)
    vrms_raw = np.sqrt(np.mean(I ** 2 + Q ** 2))

    # 2) Optional: band-limit to chosen Δf target via Butterworth LPF and compute ENBW
    have_lpf = target_bw_hz is not None
    enbw = None
    vrms_lpf = None
    data_lp = None

    if have_lpf:
        wn = min(target_bw_hz / (Fs / 2.0), 0.999)
        b, a = sig.butter(4, wn)
        I_lp = sig.filtfilt(b, a, I)
        Q_lp = sig.filtfilt(b, a, Q)
        data_lp = I_lp + 1j * Q_lp
        vrms_lpf = np.sqrt(np.mean(I_lp ** 2 + Q_lp ** 2))
        enbw = _compute_enbw_IIR(b, a, Fs)

        # ---- PLOTS FOR PROCESSED (LPF) NOISE ----
        plt.figure(figsize=(10, 4))
        plt.plot(t_us, data_lp.real, label=f"Real (LPF ~{target_bw_hz/1e3:.1f} kHz)")
        plt.plot(t_us, data_lp.imag, label=f"Imag (LPF ~{target_bw_hz/1e3:.1f} kHz)", alpha=0.7)
        plt.title("RX0 waveform (band-limited noise)")
        plt.xlabel("Time (µs)")
        plt.ylabel("Voltage (V)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{out_prefix}_complex_lpf_{int(target_bw_hz/1e3)}k.png", dpi=150)

        plt.figure(figsize=(10, 4))
        plt.plot(t_us, np.abs(data_lp))
        plt.title(f"Noise magnitude (LPF ~{target_bw_hz/1e3:.1f} kHz)")
        plt.xlabel("Time (µs)")
        plt.ylabel("Voltage (V)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"{out_prefix}_mag_lpf_{int(target_bw_hz/1e3)}k.png", dpi=150)

    # 3) Thermal predictions (referenced to output)
    kB = 1.380649e-23
    Gv = 10 ** (lna_gain_dB / 20.0)     # voltage gain
    F = 10 ** (lna_nf_dB / 10.0)        # noise factor
    e_out = Gv * np.sqrt(F) * np.sqrt(4 * kB * T_K * R_ohm)  # V/√Hz @ output

    # Baselines for reference
    vth_50_real = e_out * np.sqrt(Fs / 2.0)  # if stream were real
    vth_50_cplx = e_out * np.sqrt(Fs)        # complex stream baseline
    vth_target = e_out * np.sqrt(enbw) if have_lpf else None

    # 4) Spectral densities
    en_meas_raw = vrms_raw / np.sqrt(Fs)                 # V/√Hz
    en_meas_lpf = (vrms_lpf / np.sqrt(enbw)) if have_lpf else None

    # 5) Spectrum (Welch): raw (and LPF if present)
    nper = min(8192, len(I)) if len(I) >= 256 else max(64, len(I) // 2)
    f_raw, PxxI_raw = sig.welch(I, fs=Fs, nperseg=nper)
    _, PxxQ_raw = sig.welch(Q, fs=Fs, nperseg=nper)
    ASD_raw = np.sqrt(PxxI_raw + PxxQ_raw)  # V/√Hz

    plt.figure(figsize=(10, 4))
    plt.semilogy(f_raw / 1e3, ASD_raw, label="Raw")
    xlim_max_kHz = Fs / 2 / 1e3
    if have_lpf:
        f_lp, PxxI_lp = sig.welch(I_lp, fs=Fs, nperseg=nper)
        _, PxxQ_lp = sig.welch(Q_lp, fs=Fs, nperseg=nper)
        ASD_lp = np.sqrt(PxxI_lp + PxxQ_lp)
        plt.semilogy(f_lp / 1e3, ASD_lp, label=f"LPF ~{target_bw_hz/1e3:.1f} kHz")
        plt.axvline(target_bw_hz / 1e3, ls="--", alpha=0.6)
        xlim_max_kHz = min(xlim_max_kHz, max(10 * (target_bw_hz / 1e3), 50))

    plt.title("Amplitude spectral density (V/√Hz)")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("ASD (V/√Hz)")
    plt.xlim(0, xlim_max_kHz)
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}_asd_raw_vs_lpf.png", dpi=150)

    # --- print summary
    print("==== 50 Ω Noise Summary ====")
    print(f"Fs = {Fs/1e6:.3f} MS/s, N = {data.size}, T_acq ≈ {data.size*Ts_us/1000.0:.2f} ms")
    print(f"Wideband complex RMS (Δf≈Fs): {vrms_raw*1e6:.2f} µV")
    print(f"Predicted 50Ω baseline (real, Δf≈Fs/2):  {vth_50_real*1e6:.2f} µV")
    print(f"Predicted 50Ω baseline (complex, Δf≈Fs): {vth_50_cplx*1e6:.2f} µV")
    print(
        f"Measured density (raw):  {en_meas_raw*1e9:.1f} nV/√Hz"
        f"   | Thermal density (pred): {e_out*1e9:.1f} nV/√Hz"
        f"   → ratio {en_meas_raw/e_out:.2f}×"
    )

    if have_lpf:
        print(f"\nLPF target cutoff: {target_bw_hz/1e3:.1f} kHz  | ENBW ≈ {enbw/1e3:.2f} kHz")
        print(f"Band-limited RMS (Δf≈ENBW): {vrms_lpf*1e6:.2f} µV")
        print(f"Predicted baseline for ENBW: {vth_target*1e6:.2f} µV")
        print(
            f"Measured density (LPF): {en_meas_lpf*1e9:.1f} nV/√Hz"
            f" → ratio {en_meas_lpf/e_out:.2f}×"
        )

    print("\nTip: compare raw RMS to the complex baseline (Δf≈Fs),")
    print("and compare LPF RMS to the ENBW baseline (not just the cutoff).")
    print(f"Requested acq: {acq_us} µs")
    print(f"Returned samples: {data.size}, actual RX length: {data.size*Ts_us:.2f} µs")

    return data


if __name__ == "__main__":
    acquire_spin_echo()