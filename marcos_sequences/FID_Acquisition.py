#!/usr/bin/env python3
"""
Acquire a simple pulse-and-acquire data set and plot:

1. Complex waveform vs time (arb. volts).
2. Magnitude (FID envelope) vs time.
3. Power spectrum (|FFT|) vs frequency to check for an FID peak.
4. (Optional) frequency-tracked, demodulated FID.

Notes
-----
* Absolute volts: pass `volts_per_unit` if you have a calibration
  (see comments below). Otherwise plots are in arbitrary volts.
* Timebase is taken from Experiment.get_rx_ts() so it always matches HW.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # remove if you want interactive windows
import matplotlib.pyplot as plt
from experiment import Experiment
import scipy.signal as sig

def acquire_fid(
    lo_freq_MHz = 2.1075,
    rf_amplitude = 0.011,
    pulse_us = 50.0,
    blank_us = 0,          # give the coil/LNA a little more decay headroom
    acq_us = 100,          # total RX window length
    rx_t_us = 0.03,      
    volts_per_unit = 0.0169,   # scale factor to convert Experiment units -> volts
    raw_adc = True,
    out_prefix = "fid"
):
    # --- build and run the experiment -------------------------------------------------
    expt = Experiment(
        lo_freq = lo_freq_MHz,
        rx_t    = rx_t_us,         # µs request; HW will quantize
        init_gpa=False,
        auto_leds=True,
        flush_old_rx=True
    )

    # TX pulse: start at 0, end at pulse_us
    tx_times = np.array([0.0, pulse_us], dtype=float)
    tx_amps  = np.array([rf_amplitude, 0.0], dtype=float)

    # RX enable: starts after blank_us, lasts acq_us
    rx_start = pulse_us + blank_us
    rx_times = np.array([rx_start, rx_start + acq_us], dtype=float)
    rx_gate  = np.array([1, 0], dtype=int)

    seq = {
        "tx0":   (tx_times, tx_amps + 0j),  # complex expected
        "rx0_en": (rx_times, rx_gate),
        "rx1_en": (rx_times, rx_gate),
    }

    expt.add_flodict(seq)
    rxd, msgs = expt.run()

    Ts_hw_us = float(expt.get_rx_ts()[0])      # what HW actually used (per sample)
    N = len(rxd["rx0"])
    T_span_us = N * Ts_hw_us

    print(f"Requested acq_us: {acq_us}")
    print(f"N samples: {N}")
    print(f"HW Ts: {Ts_hw_us:.6f} µs  -> span = {T_span_us:.3f} µs")


    # Pull data
    data = np.asarray(rxd.get("rx0", np.array([], dtype=complex)))
    if data.size == 0:
        print("No RX0 data returned.")
        return data

    print("Before scale:", np.min(data.real), np.max(data.real))
    if volts_per_unit is not None:
        data = data * volts_per_unit
    print("After  scale:", np.min(data.real), np.max(data.real))

    # ======== START: complex processing block ========

    # Time bases
    Ts_us = float(expt.get_rx_ts()[0])           # µs
    Ts_s  = Ts_us * 1e-6                         # s
    t     = np.arange(len(data)) * Ts_s
    t_us  = t * 1e6
    fs    = 1.0 / Ts_s

    # Keep baseline raw-real plot (sanity check)
    plt.figure(figsize=(10,4))
    plt.plot(t_us, data.real)
    plt.title("RAW real voltage (baseline)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Volts")
    plt.minorticks_on(); plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("00_raw_real_baseline.png")

    # Work COMPLEX from here on
    z = data.astype(np.complex128)
    z = z - np.mean(z)   # remove DC

    # 1) Complex waveform (real & imag)
    plt.figure(figsize=(10,4))
    plt.plot(t_us, z.real, label='Real')
    plt.plot(t_us, z.imag, label='Imag', alpha=0.7)
    plt.title("Complex FID (Volts)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Volts")
    plt.legend()
    plt.minorticks_on(); plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("01_complex_time.png")

    # 2) Magnitude envelope (no demod yet)
    env = np.abs(z)
    plt.figure(figsize=(10,4))
    plt.plot(t_us, env)
    plt.title("|FID| envelope (no residual demod)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Volts")
    plt.minorticks_on(); plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("02_env_nodemod.png")

    # 3) Spectrum to find the residual frequency (complex FFT)
    Z = np.fft.fft(z)
    f = np.fft.fftfreq(len(z), Ts_s)   # Hz
    idx = np.argmax(np.abs(Z))
    f_res = f[idx]
    print(f"Residual baseband peak (complex) = {f_res/1e3:.1f} kHz")

    # 4) Demod by the residual to center the FID at DC
    lo   = np.exp(-1j * 2*np.pi * f_res * t)
    z_bb = z * lo
    env_bb = np.abs(z_bb)

    plt.figure(figsize=(10,4))
    plt.plot(t_us, env_bb)
    plt.title("|FID| envelope (after complex residual demod)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Volts")
    plt.minorticks_on(); plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("03_env_demod.png")

    # 5) (Optional) instantaneous frequency from phase – useful to see drift
    phi = np.unwrap(np.angle(z_bb))
    inst_freq = np.gradient(phi) / (2*np.pi*Ts_s)   # Hz

    plt.figure(figsize=(10,4))
    plt.plot(t_us, inst_freq/1e3)
    plt.title("Instantaneous frequency after demod (kHz)")
    plt.xlabel("Time (µs)")
    plt.ylabel("kHz")
    plt.minorticks_on(); plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig("04_inst_freq.png")

# ======== END: complex processing block ========

        
if __name__ == "__main__":
    acquire_fid()