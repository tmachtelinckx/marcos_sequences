import numpy as np
import os
import time
import matplotlib.pyplot as plt
from experiment import Experiment
import scipy.signal as sig

# -------------------------------
# Helpers
# -------------------------------
def _compute_enbw(b, a, Fs, passes=2, two_sided=True):
    """
    Effective Noise Bandwidth (ENBW) for a digital IIR filter.
    If applied with filtfilt (zero-phase), set passes=2 (forward+backward).
    ENBW = ∫ |H(f)|^{2*passes} df (two-sided if two_sided=True).
    """
    w, H = sig.freqz(b, a, worN=16384)      # normalized rad/sample, 0..pi
    Hpow  = np.abs(H)**(2*passes)
    enbw_one_sided = (Fs/(2*np.pi)) * np.trapz(Hpow, w)
    return 2.0 * enbw_one_sided if two_sided else enbw_one_sided


# -------------------------------
# Single run (spin echo) with LPF, ENBW, plots & metrics
# -------------------------------
def acquire_spin_echo_reusable(
    expt,
    lo_freq_MHz=2.1075,
    rf90_amplitude=0.2,
    rf180_amplitude=0.2,
    pulse90_us=300.0,
    pulse180_us=600.0,
    tau_us=600.0,
    blank2_us=100.0,
    acq_us=1500.0,
    volts_per_unit=0.00845,
    out_prefix="se",
    # Analysis knobs
    lpf_cut_hz=None,         # e.g., 3600 for coil BW ~ 7.4 kHz (ENBW ~ 2.05*cutoff)
    butter_order=4,
    lna_gain_dB=38,
    lna_nf_dB=1.0,
    R_ohm=50.0,
    T_K=300.0,
    echo_window_us=150.0     # half-width around echo center for metrics
):
    # Configure LO
    expt.set_lo_freq(lo_freq_MHz)
    expt._seq = None  # clear any previous sequence

    # --- TX/RX sequence (Hahn spin echo) ---
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
        # "rx1_en": (rx_times, rx_gate),  # keep disabled to avoid FIFO pressure
    }
    expt.add_flodict(seq, append=False)
    rxd, msgs = expt.run()

    # --- Pull data ---
    data = np.asarray(rxd.get("rx0", np.array([], dtype=complex)))
    if data.size == 0:
        print("No RX0 data returned.")
        return {}

    Ts_us = float(expt.get_rx_ts()[0])   # µs/sample
    Ts_s  = Ts_us * 1e-6
    Fs    = 1.0 / Ts_s

    if volts_per_unit is not None:
        data = data * float(volts_per_unit)

    t = np.arange(data.size, dtype=float) * Ts_us  # µs axis

    # --- RAW plots ---
    plt.figure(figsize=(10,4))
    plt.plot(t, data.real, label="Real")
    plt.plot(t, data.imag, label="Imag", alpha=0.7)
    plt.title(f"RX0 waveform (Spin Echo) - {lo_freq_MHz:.6f} MHz")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_complex.png", dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(t, data.real, label="Real")
    plt.title(f"RX0 Real waveform (Spin Echo) - {lo_freq_MHz:.6f} MHz")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_Real.png", dpi=150); plt.close()

    mag = np.abs(data)
    plt.figure(figsize=(10,4))
    plt.plot(t, mag)
    plt.title(f"Spin Echo Magnitude (Envelope) - {lo_freq_MHz:.6f} MHz")
    plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_mag.png", dpi=150); plt.close()

    # Optional RF view (digital unmix)
    freq_offset_Hz = lo_freq_MHz * 1e6
    t_s = np.arange(data.size) * Ts_s
    data_raw = data * np.exp(1j * 2 * np.pi * freq_offset_Hz * t_s)
    plt.figure(figsize=(10,4))
    plt.plot(t, data_raw.real, label="Real (unmixed)")
    plt.plot(t, data_raw.imag, label="Imag (unmixed)", alpha=0.7)
    plt.legend(); plt.title(f"Digitally shifted back to RF - {lo_freq_MHz:.6f} MHz")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{out_prefix}_raw.png", dpi=150); plt.close()

    # --- Noise analysis ---
    I = data.real - np.mean(data.real)
    Q = data.imag - np.mean(data.imag)

    # Raw wideband (complex stream ⇒ Δf ≈ Fs)
    vrms_raw = np.sqrt(np.mean(I**2 + Q**2))

    # LPF (optional)
    enbw_hz = None
    I_lp, Q_lp = I, Q
    if lpf_cut_hz is not None:
        wn = min(lpf_cut_hz/(Fs/2.0), 0.999)
        b, a = sig.butter(butter_order, wn, btype='low')
        # zero-phase (filtfilt) → power response |H|^4
        padlen = max(3*(max(len(b), len(a))), 0)
        I_lp = sig.filtfilt(b, a, I, padlen=padlen)
        Q_lp = sig.filtfilt(b, a, Q, padlen=padlen)
        enbw_hz = _compute_enbw(b, a, Fs, passes=2, two_sided=True)

        # Filtered plots
        data_lp = I_lp + 1j*Q_lp
        plt.figure(figsize=(10,4))
        plt.plot(t, data_lp.real, label=f"Real (LPF ~{lpf_cut_hz/1e3:.1f} kHz)")
        plt.plot(t, data_lp.imag, label=f"Imag (LPF ~{lpf_cut_hz/1e3:.1f} kHz)", alpha=0.7)
        plt.title("Band-limited I/Q")
        plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"{out_prefix}_complex_lpf.png", dpi=150); plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(t, np.abs(data_lp))
        plt.title("Band-limited magnitude")
        plt.xlabel("Time (µs)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"{out_prefix}_mag_lpf.png", dpi=150); plt.close()

    # RMS after LPF (or same as raw if no LPF)
    vrms_lpf = np.sqrt(np.mean(I_lp**2 + Q_lp**2))

    # Thermal prediction
    kB = 1.380649e-23
    Gv = 10**(lna_gain_dB/20.0)
    F  = 10**(lna_nf_dB/10.0)
    e_out = Gv * np.sqrt(F) * np.sqrt(4*kB*T_K*R_ohm)  # V/√Hz

    bw_for_pred = Fs if enbw_hz is None else enbw_hz
    vth_pred = e_out * np.sqrt(bw_for_pred)

    # ASD (raw vs LPF)
    nper = min(8192, len(I))
    f_raw, PxxI_raw = sig.welch(I, fs=Fs, nperseg=nper)
    _,     PxxQ_raw = sig.welch(Q, fs=Fs, nperseg=nper)
    ASD_raw = np.sqrt(PxxI_raw + PxxQ_raw)

    plt.figure(figsize=(10,4))
    plt.semilogy(f_raw/1e3, ASD_raw, label="Raw")
    if enbw_hz is not None:
        f_lp, PxxI_lp = sig.welch(I_lp, fs=Fs, nperseg=nper)
        _,   PxxQ_lp  = sig.welch(Q_lp, fs=Fs, nperseg=nper)
        ASD_lp = np.sqrt(PxxI_lp + PxxQ_lp)
        plt.semilogy(f_lp/1e3, ASD_lp, label="LPF")
        plt.axvline(lpf_cut_hz/1e3, ls="--", alpha=0.6)
    plt.title("Amplitude spectral density (V/√Hz)")
    plt.xlabel("Frequency (kHz)"); plt.ylabel("ASD (V/√Hz)")
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_prefix}_asd.png", dpi=150); plt.close()

    # --- Echo metrics (band-limited stream) ---
    # echo center relative to RX start: t_echo,RX = 0.5*t90 + tau - tblank
    t_echo_rx_us = 0.5*pulse90_us + tau_us - blank2_us

    # choose a window around echo center
    i0 = int(max(0, (t_echo_rx_us - echo_window_us)/Ts_us))
    i1 = int(min(len(mag)-1, (t_echo_rx_us + echo_window_us)/Ts_us))
    mag_lp = np.abs(I_lp + 1j*Q_lp)

    if i1 > i0:
        echo_peak   = float(np.max(mag_lp[i0:i1]))
        echo_energy = float(np.sum(mag_lp[i0:i1]**2) * Ts_s)
        # noise from a pre-echo window (same size) if possible
        j1 = max(0, i0 - (i1 - i0))
        j0 = max(0, j1 - (i1 - i0))
        noise_rms_pre = float(np.sqrt(np.mean((mag_lp[j0:j1]**2)))) if j1 > j0 else float(np.sqrt(np.mean(mag_lp**2)))
    else:
        echo_peak   = float(np.max(mag_lp))
        echo_energy = float(np.sum(mag_lp**2) * Ts_s)
        noise_rms_pre = float(np.sqrt(np.mean(mag_lp**2)))

    snr_peak = float(echo_peak / (noise_rms_pre + 1e-12))

    # --- Densities & ratios (for summaries) ---
    dens_raw = vrms_raw / np.sqrt(Fs)             # V/√Hz
    dens_lpf = vrms_lpf / np.sqrt(bw_for_pred)    # V/√Hz (uses ENBW if LPF, else Fs)
    ratio_rms  = (vrms_lpf / vth_pred) if vth_pred > 0 else np.nan
    ratio_dens = (dens_lpf / e_out)   if e_out    > 0 else np.nan

    # Print quick console summary
    print("==== Spin-Echo Summary ====")
    print(f"Fs = {Fs/1e6:.3f} MS/s, N = {data.size}, T_acq ≈ {data.size*Ts_us/1000:.2f} ms")
    print(f"Raw complex RMS (Δf≈Fs): {vrms_raw*1e6:.2f} µV")
    if enbw_hz is None:
        print(f"Predicted thermal RMS (complex, Δf≈Fs): {vth_pred*1e6:.2f} µV")
    else:
        print(f"LPF cutoff (nominal): {lpf_cut_hz/1e3:.2f} kHz  | ENBW ≈ {enbw_hz/1e3:.2f} kHz")
        print(f"Band-limited RMS (Δf≈ENBW): {vrms_lpf*1e6:.2f} µV  | Predicted: {vth_pred*1e6:.2f} µV")
        print(f"Noise density (meas/pred): {dens_lpf*1e9:.1f}/{e_out*1e9:.1f} nV/√Hz → {ratio_dens:.2f}×")
    print(f"Echo center (from RX start): {t_echo_rx_us:.1f} µs | Peak: {echo_peak:.3g} V | SNR_peak≈{snr_peak:.1f}")
    print(f"Echo energy (|·|^2·dt): {echo_energy:.3e} V^2·s")

    return {
        "Fs": Fs,
        "Ts_us": Ts_us,
        "N": data.size,
        "vrms_raw": vrms_raw,
        "vrms_lpf": vrms_lpf,
        "enbw_hz": bw_for_pred,   # ENBW if LPF else Fs
        "vth_pred": vth_pred,
        "e_out": e_out,
        "dens_raw": dens_raw,
        "dens_lpf": dens_lpf,
        "ratio_rms": ratio_rms,
        "ratio_dens": ratio_dens,
        "echo_peak": echo_peak,
        "echo_energy": echo_energy,
        "snr_peak": snr_peak,
        "t_echo_rx_us": t_echo_rx_us
    }


# -------------------------------
# Frequency sweep (creates per-frequency folders, plots & summary.txt)
# -------------------------------
def sweep_spin_echo(
    freq_start=2.08,
    freq_stop=2.13,
    freq_step=0.0025,
    results_dir="sweep_results",
    lpf_cut_hz=3600.0,     # ~ coil BW/2.05 if Q_loaded ~ 280 at 2.1 MHz
    rx_t_us=0.25,          # Fs ≈ 4.0 MS/s → change as needed
    # default pulses per your matrix (you'll tweak when calling acquire directly if needed)
    pulse90_us=300.0,
    pulse180_us=600.0,
    tau_us=600.0,
    blank2_us=100.0,
    acq_us=1500.0,
    rf90_amp=0.2,
    rf180_amp=0.2
):
    os.makedirs(results_dir, exist_ok=True)
    expt = Experiment(
        lo_freq=freq_start,
        rx_t=rx_t_us,
        init_gpa=False,
        auto_leds=True,
        flush_old_rx=True
    )

    results = []
    freqs = np.arange(freq_start, freq_stop + 0.5*freq_step, freq_step)

    try:
        for i, f in enumerate(freqs):
            freq_folder = os.path.join(results_dir, f"freq_{f:.6f}MHz")
            os.makedirs(freq_folder, exist_ok=True)
            prefix = os.path.join(freq_folder, f"se_{f:.6f}MHz")

            print(f"\nRunning spin echo {i+1}/{len(freqs)} at {f:.6f} MHz ...")

            metrics = acquire_spin_echo_reusable(
                expt,
                lo_freq_MHz=f,
                rf90_amplitude=rf90_amp,
                rf180_amplitude=rf180_amp,
                pulse90_us=pulse90_us,
                pulse180_us=pulse180_us,
                tau_us=tau_us,
                blank2_us=blank2_us,
                acq_us=acq_us,
                out_prefix=prefix,
                lpf_cut_hz=lpf_cut_hz
            )
            if not metrics:
                print("⚠️ No data returned for this frequency.")
                continue

            # Append to consolidated results (CSV later)
            results.append([
                f,
                metrics["echo_peak"],
                metrics["snr_peak"],
                metrics["vrms_lpf"],
                metrics["enbw_hz"],
                metrics["vth_pred"],
                metrics["dens_lpf"],
                metrics["e_out"],
                metrics["ratio_rms"],
                metrics["ratio_dens"]
            ])

            # Per-frequency summary.txt
            meas_density_nV = metrics["dens_lpf"] * 1e9
            pred_density_nV = metrics["e_out"]   * 1e9
            ratio_rms  = metrics["ratio_rms"]
            ratio_dens = metrics["ratio_dens"]

            summary_path = os.path.join(freq_folder, "summary.txt")
            with open(summary_path, "w") as fh:
                fh.write(f"Frequency (LO): {f:.6f} MHz\n")
                fh.write(f"Fs: {metrics['Fs']/1e6:.3f} MS/s, N: {metrics['N']}, "
                         f"T_acq ≈ {metrics['N']*metrics['Ts_us']/1000:.2f} ms\n")
                if metrics["enbw_hz"] == metrics["Fs"]:
                    fh.write("Band: RAW (complex), Δf ≈ Fs\n")
                else:
                    fh.write(f"LPF nominal cutoff: ~{lpf_cut_hz/1e3:.2f} kHz | "
                             f"ENBW ≈ {metrics['enbw_hz']/1e3:.2f} kHz\n")

                fh.write("\nNoise metrics:\n")
                fh.write(f"  Raw RMS (Δf≈Fs):      {metrics['vrms_raw']*1e6:8.2f} µV\n")
                fh.write(f"  Band-limited RMS:      {metrics['vrms_lpf']*1e6:8.2f} µV\n")
                fh.write(f"  Measured density:      {meas_density_nV:8.1f} nV/√Hz\n")
                fh.write(f"  Thermal density (pred):{pred_density_nV:8.1f} nV/√Hz\n")
                fh.write(f"  RMS ratio (meas/pred): {ratio_rms:6.2f}×\n")
                fh.write(f"  Density ratio:         {ratio_dens:6.2f}×\n")

                fh.write("\nEcho metrics (band-limited):\n")
                fh.write(f"  Echo center (RX start): {metrics['t_echo_rx_us']:.1f} µs\n")
                fh.write(f"  Peak magnitude:         {metrics['echo_peak']:.6g} V\n")
                fh.write(f"  Echo energy (|·|^2·dt): {metrics['echo_energy']:.3e} V^2·s\n")
                fh.write(f"  SNR_peak ≈ {metrics['snr_peak']:.1f}\n")

                fh.write("\nFiles saved in this folder:\n")
                fh.write("  se_complex.png, se_Real.png, se_mag.png, se_raw.png\n")
                fh.write("  se_complex_lpf.png, se_mag_lpf.png, se_asd.png\n")

            time.sleep(1.0)

    finally:
        try:
            expt.close_server()
        except Exception:
            pass

    # Save consolidated CSV and a quick SNR plot across frequencies
    if results:
        results = np.array(results, dtype=float)
        csv_path = os.path.join(results_dir, "frequency_sweep_metrics.csv")
        np.savetxt(
            csv_path,
            results,
            delimiter=",",
            header=("Frequency_MHz,PeakEcho_V,SNR_peak,VRMS_LPF_V,ENBW_Hz,"
                    "PredThermal_V,MeasDensity_V_per_rtHz,PredDensity_V_per_rtHz,"
                    "Ratio_RMS,Ratio_Density"),
            comments=""
        )

        # Rank by SNR
        order = np.argsort(results[:,2])[::-1]
        best = results[order]
        print("\nTop by SNR_peak:")
        for k in range(min(10, len(best))):
            print(f"{k+1:2d}. {best[k,0]:.6f} MHz | SNR={best[k,2]:.1f} | Peak={best[k,1]:.3g} V")

        # Plot SNR vs frequency
        plt.figure(figsize=(10,6))
        plt.plot(results[:,0], results[:,2], 'o-')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('SNR_peak')
        plt.grid(True)
        plt.title('Spin-Echo SNR vs LO frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "sweep_snr.png"), dpi=150)
        plt.close()
    else:
        print("\n❌ No valid data collected.")


# -------------------------------
# Example runner
# -------------------------------
if __name__ == "__main__":
    # You can edit these to match one of your test cases (P1/P2/P3) and TX amplitude set
    sweep_spin_echo(
        freq_start=2.1,
        freq_stop=2.13,
        freq_step=0.0002,
        results_dir="sweep_results",
        lpf_cut_hz=5000.0,     # ~Δf_coil/2.05 if Δf_coil≈7.4 kHz
        rx_t_us=0.30,          # Fs ≈ 3.33 MS/s (adjust as needed)
        pulse90_us=150.0,      # pick 150/300/500 per your matrix
        pulse180_us=300.0,     # = 2*t90
        tau_us=300.0,          # = t180
        blank2_us=100.0,       # fixed ring-down blanking
        acq_us=1500.0,         # 1.5 ms RX
        rf90_amp=0.7,         # set 0.01 / 0.20 / 0.80 V equivalent DAC as needed
        rf180_amp=0.7
    )