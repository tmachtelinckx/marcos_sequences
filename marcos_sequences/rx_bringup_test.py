#!/usr/bin/env python3
"""
Red Pitaya RX Test Script
Test the receive functionality using an external signal generator
"""

from experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt

def simple_rx_test():
    """Simple RX test - just capture data for a specified time"""
    
    print("=== Simple RX Test ===")
    print("Connect your signal generator to RX0 and/or RX1 inputs")
    print("Recommended signal: 1-10 MHz, -10 to 0 dBm amplitude")
    input("Press Enter when signal generator is connected and configured...")
    
    # Configuration
    lo_freq = 2.1075  # MHz - local oscillator frequency
    rx_time = 1000  # microseconds - how long to capture
    rx_sample_period = 1  # microseconds - sampling period
    
    print(f"LO Frequency: {lo_freq} MHz")
    print(f"RX Duration: {rx_time} μs")
    print(f"RX Sample Period: {rx_sample_period} μs")
    
    # Create experiment
    expt = Experiment(
        lo_freq=lo_freq,
        rx_t=rx_sample_period,
        init_gpa=False,
        auto_leds=False,  # Turn off LED scanning for cleaner test
        flush_old_rx=True
    )
    
    # Define RX sequence - just enable RX for specified time
    sequence_dict = {
        'rx0_en': (np.array([0, rx_time]), np.array([1, 0])),  # Enable RX0
        'rx1_en': (np.array([0, rx_time]), np.array([1, 0])),  # Enable RX1
    }
    
    # Add sequence and run
    try:
        expt.add_flodict(sequence_dict)
        print("Running RX test...")
        rxd, msgs = expt.run()
        
        print("✓ RX test completed!")
        
        # Analyze received data
        analyze_rx_data(rxd, rx_sample_period, lo_freq, detailed=True)
        
    except Exception as e:
        print(f"✗ Error in RX test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        expt.close_server(only_if_sim=True)

def frequency_sweep_rx_test():
    """Test RX with different LO frequencies to find your signal"""
    
    print("=== Frequency Sweep RX Test ===")
    print("This will test different LO frequencies to help locate your signal")
    
    # Get signal generator frequency from user
    try:
        sig_freq = float(input("Enter your signal generator frequency (MHz): "))
    except:
        sig_freq = 5.0
        print(f"Using default: {sig_freq} MHz")
    
    # Test LO frequencies around the signal frequency
    lo_frequencies = np.array([sig_freq - 1, sig_freq - 0.5, sig_freq, sig_freq + 0.5, sig_freq + 1])
    rx_time = 500  # shorter captures for sweep
    rx_sample_period = 5
    
    results = []
    
    for lo_freq in lo_frequencies:
        print(f"\nTesting LO = {lo_freq:.1f} MHz...")
        
        expt = Experiment(
            lo_freq=lo_freq,
            rx_t=rx_sample_period,
            init_gpa=False,
            auto_leds=False,
            flush_old_rx=True
        )
        
        sequence_dict = {
            'rx0_en': (np.array([0, rx_time]), np.array([1, 0])),
            'rx1_en': (np.array([0, rx_time]), np.array([1, 0])),
        }
        
        try:
            expt.add_flodict(sequence_dict)
            rxd, msgs = expt.run()
            
            # Calculate signal strength
            if 'rx0' in rxd:
                signal_power = np.mean(np.abs(rxd['rx0'])**2)
                results.append((lo_freq, signal_power))
                print(f"  RX0 signal power: {signal_power:.2e}")
            else:
                results.append((lo_freq, 0))
                print("  No RX0 data received")
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append((lo_freq, 0))
        
        finally:
            expt.close_server(only_if_sim=True)
    
    # Plot results
    if results:
        freqs, powers = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, powers, 'bo-')
        plt.xlabel('LO Frequency (MHz)')
        plt.ylabel('Received Signal Power')
        plt.title('RX Signal Power vs LO Frequency')
        plt.grid(True)
        plt.savefig("sweeptest.png")
        plt.close()
        
        # Find best frequency
        best_idx = np.argmax(powers)
        print(f"\nBest LO frequency: {freqs[best_idx]:.1f} MHz (power: {powers[best_idx]:.2e})")

def detailed_rx_analysis():
    """Detailed RX test with signal analysis and plotting"""
    
    print("=== Detailed RX Analysis ===")
    
    # Get parameters from user
    try:
        lo_freq = float(input("Enter LO frequency (MHz) [2.0]: ") or "2.0")
        rx_time = float(input("Enter capture time (μs) [2000]: ") or "2000")
        rx_period = float(input("Enter sample period (μs) [2]: ") or "2")
    except:
        lo_freq, rx_time, rx_period = 5.0, 2000, 2
    
    print(f"Configuration: LO={lo_freq} MHz, Time={rx_time} μs, Period={rx_period} μs")
    
    expt = Experiment(
        lo_freq=lo_freq,
        rx_t=rx_period,
        init_gpa=False,
        auto_leds=False,
        flush_old_rx=True
    )
    
    sequence_dict = {
        'rx0_en': (np.array([0, rx_time]), np.array([1, 0])),
        'rx1_en': (np.array([0, rx_time]), np.array([1, 0])),
    }
    
    try:
        expt.add_flodict(sequence_dict)
        print("Capturing data...")
        rxd, msgs = expt.run()
        
        if rxd:
            print("✓ Data captured successfully!")
            analyze_rx_data(rxd, rx_period, lo_freq, detailed=True)
        else:
            print("✗ No data received")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        expt.close_server(only_if_sim=True)

def analyze_rx_data(rxd, sample_period_us, lo_freq, detailed=False):
    """Analyze and plot received data"""
    
    print("\n--- RX Data Analysis ---")
    
    for ch in ['rx0', 'rx1']:
        if ch in rxd:
            data = rxd[ch]
            print(f"\n{ch.upper()} Channel:")
            print(f"  Samples: {len(data)}")
            print(f"  Data type: {type(data[0]) if len(data) > 0 else 'No data'}")
            
            if len(data) > 0:
                # Basic statistics
                magnitude = np.abs(data)
                phase = np.angle(data)
                
                print(f"  Magnitude - Mean: {np.mean(magnitude):.3e}, Max: {np.max(magnitude):.3e}")
                print(f"  Phase - Range: {np.min(phase):.2f} to {np.max(phase):.2f} rad")
                
                # Check for signal presence
                noise_floor = np.median(magnitude)
                signal_threshold = 3 * noise_floor
                signal_detected = np.max(magnitude) > signal_threshold
                
                print(f"  Noise floor estimate: {noise_floor:.3e}")
                print(f"  Signal detected: {'Yes' if signal_detected else 'No'}")
                
                if detailed and len(data) > 10:
                    plot_rx_data(data, sample_period_us, lo_freq, ch)
        else:
            print(f"\n{ch.upper()} Channel: No data")

def plot_rx_data(data, sample_period_us, lo_freq, channel_name):
    """Create detailed plots of RX data"""
    
    # Time axis
    t = np.arange(len(data)) * sample_period_us
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{channel_name.upper()} Data Analysis (LO = {lo_freq} MHz)')
    
    # Time domain - magnitude
    axes[0,0].plot(t, np.abs(data))
    axes[0,0].set_xlabel('Time (μs)')
    axes[0,0].set_ylabel('Magnitude')
    axes[0,0].set_title('Signal Magnitude vs Time')
    axes[0,0].grid(True)
    
    # Time domain - I/Q
    axes[0,1].plot(t, data.real, label='I (Real)')
    axes[0,1].plot(t, data.imag, label='Q (Imag)')
    axes[0,1].set_xlabel('Time (μs)')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].set_title('I/Q Components vs Time')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Frequency domain
    if len(data) > 1:
        sample_rate = 1e6 / sample_period_us  # Hz
        freqs = np.fft.fftfreq(len(data), 1/sample_rate) / 1e6 + lo_freq # MHz
        fft_data = np.fft.fft(data)
        
        # Only plot positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_data[:len(fft_data)//2]
        
        axes[1,0].plot(pos_freqs, 20*np.log10(np.abs(pos_fft) + 1e-12))
        axes[1,0].set_xlabel('Frequency (MHz)')
        axes[1,0].set_ylabel('Power (dB)')
        axes[1,0].set_title('Frequency Spectrum')
        axes[1,0].grid(True)
    
    # I/Q constellation
    axes[1,1].scatter(data.real, data.imag, alpha=0.5, s=1)
    axes[1,1].set_xlabel('I (Real)')
    axes[1,1].set_ylabel('Q (Imaginary)')
    axes[1,1].set_title('I/Q Constellation')
    axes[1,1].grid(True)
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("Rx_data")
    plt.close()

if __name__ == "__main__":
    print("Red Pitaya RX Test Suite")
    print("=" * 30)
    print("1. Simple RX Test")
    print("2. Frequency Sweep Test") 
    print("3. Detailed RX Analysis")
    
    choice = input("\nChoose test (1-3): ")
    
    if choice == "1":
        simple_rx_test()
    elif choice == "2":
        frequency_sweep_rx_test()
    elif choice == "3":
        detailed_rx_analysis()
    else:
        print("Invalid choice, running simple test...")
        simple_rx_test()
