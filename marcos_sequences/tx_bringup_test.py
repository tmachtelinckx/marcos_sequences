#!/usr/bin/env python3
"""
Simple test script for Red Pitaya without GPA
Generates a continuous sine wave on TX outputs
"""

from experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt

def simple_tx_test():
    """Generate a simple continuous sine wave for oscilloscope testing"""
    
    # Create experiment with basic settings
    # lo_freq: carrier frequency in MHz
    # rx_t: not critical for TX-only test
    expt = Experiment(
        lo_freq=0.1,  # 1 MHz carrier frequency
        rx_t=10,      # RX sampling period (not used in this test)
        init_gpa=False,  # Don't initialize GPA
        auto_leds=True,  # Keep LED scanning for visual feedback
        flush_old_rx=True  # Clear any old data
    )
    
    # Define a simple test sequence
    # Times in microseconds, amplitudes from -1 to +1
    test_duration = 2000  # 1000 microseconds = 1 ms
    
    # Create a simple step function for testing
    # Start with 0, then go to 0.5 amplitude, then back to 0
    tx_times = np.array([0, 10, test_duration-10, test_duration])  # us
    tx_amplitudes = np.array([0, 0.2, 1.0, 0])  # amplitude
    
    # Alternative: create a more complex waveform
    # Uncomment the lines below for a ramped signal
    # tx_times = np.linspace(0, test_duration, 100)
    # tx_amplitudes = 0.5 * np.sin(2 * np.pi * 0.01 * tx_times)  # 10 kHz modulation
    
    # Define the sequence dictionary
    sequence_dict = {
        'tx0': (tx_times, tx_amplitudes + 0j),  # TX0 output (complex format)
        'tx1': (tx_times, tx_amplitudes + 0j),  # TX1 output (same signal)
        'tx_gate': (np.array([0, test_duration]), np.array([1, 0])),  # Enable TX
    }
    
    # Add the sequence to the experiment
    expt.add_flodict(sequence_dict)
    
    # Plot the sequence for verification
    print("Plotting sequence...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot TX signals
    axes[0].step(tx_times, tx_amplitudes, where='post', label='TX amplitude')
    axes[0].set_ylabel('TX Amplitude')
    axes[0].set_title('TX Output Signal')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot TX gate
    gate_times = np.array([0, test_duration])
    gate_values = np.array([1, 0])
    axes[1].step(gate_times, gate_values, where='post', label='TX Gate', color='red')
    axes[1].set_ylabel('TX Gate')
    axes[1].set_xlabel('Time (μs)')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("simple_tx_test_plot.png")
    plt.close()

    
    # Run the experiment
    print("Running experiment...")
    print(f"Expected TX frequency: {expt._lo_freqs[0]:.3f} MHz (carrier)")
    print(f"TX duration: {test_duration} μs")
    print("Check your oscilloscope for the output signal!")
    
    try:
        rxd, msgs = expt.run()
        print("Experiment completed successfully!")
        
        # Print any messages from the server
        if msgs:
            print("Server messages:", msgs)
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        print("Check your Red Pitaya connection and configuration.")
    
    finally:
        # Close the server connection if it's a simulation
        expt.close_server(only_if_sim=True)

def continuous_sine_test():
    """Generate a longer continuous sine wave"""
    
    expt = Experiment(
        lo_freq=1.0,  # 500 kHz carrier
        rx_t=10,
        init_gpa=False,
        auto_leds=True,
        flush_old_rx=True
    )
    
    # Create a longer, continuous signal
    duration = 10000  # 10 ms
    num_points = 1000
    
    tx_times = np.linspace(0, duration, num_points)
    # Create a modulated signal: carrier + low frequency modulation
    modulation_freq = 0.001  # 1 kHz modulation
    tx_amplitudes = 0.8 * np.sin(2 * np.pi * modulation_freq * tx_times)
    
    sequence_dict = {
        'tx0': (tx_times, tx_amplitudes + 0j),
        'tx1': (tx_times, tx_amplitudes + 0j),
        'tx_gate': (np.array([0, duration]), np.array([1, 0])),
    }
    
    expt.add_flodict(sequence_dict)

    # ▶️ Plot the TX waveform
    plt.figure(figsize=(10, 4))
    plt.plot(tx_times, tx_amplitudes, label="TX Amplitude")
    plt.title("Transmitted Signal: Modulated Sine Wave")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("simple_sinusoid_test_plot.png")
    plt.close()
    
    print("Running continuous sine wave test...")
    print(f"Carrier frequency: {expt._lo_freqs[0]:.3f} MHz")
    print(f"Modulation frequency: 1 kHz")
    print(f"Duration: {duration/1000:.1f} ms")
    
    try:
        rxd, msgs = expt.run()
        print("Continuous sine test completed!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        expt.close_server(only_if_sim=True)

if __name__ == "__main__":
    print("Red Pitaya TX Test Script")
    print("=" * 30)
    
    # Choose which test to run
    test_choice = input("Choose test (1=simple step, 2=continuous sine): ")
    
    if test_choice == "1":
        simple_tx_test()
    elif test_choice == "2":
        continuous_sine_test()
    else:
        print("Running simple test by default...")
        simple_tx_test()
