#!/usr/bin/env python3
"""
Real-time minimal output for Nucleo: read IMU lines from serial and print CSV:
timestamp, ax, ay, az, gx, gy, gz, vel_x, vel_y, vel_z, is_moving

This file is a compact adaptation of analysis-try-uart.py focusing only on the requested outputs.
"""
from dataclasses import dataclass
import time
import sys
# imufusion removed: show raw sensor data only
import numpy as np
import serial

# Serial configuration (adjust if needed)
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 921600

# Parameters (match analysis-try-uart.py)
sample_rate = 100  # nominal, used for Offset initialisation
motion_threshold = 10.0  # m/s^2: same threshold used in analysis script

def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        return ser
    except Exception as e:
        print(f"Failed to open serial {port}: {e}", file=sys.stderr)
        raise


def main():
    ser = open_serial(SERIAL_PORT, BAUDRATE)
    running = True
    from collections import deque
    import threading
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    data_deque = deque(maxlen=int(sample_rate * 15))  # store up to ~15s to be safe
    lock = threading.Lock()

    prev_time = None
    accel_unit = 'm/s^2'  # detected unit for accel (displayed on plot)

    # background thread: read serial, process and append samples
    def reader():
        nonlocal prev_time, running, accel_unit
        try:
            while running:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    s = raw.decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue
                if not s:
                    continue
                parts = s.split(',')
                if len(parts) < 7:
                    continue
                try:
                    vals = [float(p) for p in parts[:7]]
                except ValueError:
                    continue

                tstamp = vals[0]
                # input format: timestamp, gx, gy, gz, ax, ay, az
                gx, gy, gz = vals[1], vals[2], vals[3]
                ax, ay, az = vals[4], vals[5], vals[6]

                # compute dt from timestamps (kept for possible future use)
                if prev_time is None:
                    dt = 0.0
                else:
                    dt = tstamp - prev_time
                    if dt < 0:
                        dt = 0.0
                prev_time = tstamp

                # work with raw accel/gyro (no AHRS, no integration)
                gyro = np.array([gx, gy, gz], dtype=float)
                accel = np.array([ax, ay, az], dtype=float)

                # Detect acc unit for label only
                with lock:
                    accel_unit = 'm/s^2' if np.max(np.abs(accel)) > 4.0 else 'g'

                sample = (tstamp, accel.copy(), gyro.copy())
                with lock:
                    data_deque.append(sample)
        except Exception as e:
            print(f"Reader thread error: {e}", file=sys.stderr)
        finally:
            running = False

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    # Setup matplotlib figure with 2 stacked subplots (accel, gyro)
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # time display text (absolute timestamp of latest sample)
    time_text = fig.text(0.01, 0.98, '', va='top')

    # lines: accel (r,g,b), gyro (r,g,b)
    accel_lines = [axes[0].plot([], [], color=c, label=l)[0] for l, c in zip(('ax', 'ay', 'az'), ('r', 'g', 'b'))]
    gyro_lines = [axes[1].plot([], [], color=c, label=l)[0] for l, c in zip(('gx', 'gy', 'gz'), ('r', 'g', 'b'))]

    axes[0].set_ylabel('acc (m/s^2)')
    axes[1].set_ylabel('gyro (deg/s)')
    axes[1].set_xlabel('time (s)')

    for ax in axes:
        ax.grid(True)

    axes[0].legend()
    axes[1].legend()

    window = 10.0  # seconds to show
    # adaptive y-limits state (smoothed to avoid flicker)
    accel_ylim = [-20.0, 20.0]
    gyro_ylim = [-200.0, 200.0]
    ylim_alpha = 0.2  # smoothing factor: 0..1 (larger = faster adaptation)

    def update_plot(frame):
        with lock:
            if not data_deque:
                return accel_lines + gyro_lines
            data = list(data_deque)
            current_unit = accel_unit

        latest_t = data[-1][0]
        # update displayed time text
        time_text.set_text(f"t = {latest_t:.3f} s")
        t0 = latest_t - window

        # filter for last `window` seconds
        filtered = [d for d in data if d[0] >= t0]
        if not filtered:
            return accel_lines + gyro_lines

        times = np.array([d[0] - latest_t for d in filtered])  # negative to 0

        acc = np.vstack([d[1] for d in filtered])  # N x 3
        gyr = np.vstack([d[2] for d in filtered])

        # update accel (set data only; avoid autoscale to prevent flicker)
        for i, line in enumerate(accel_lines):
            line.set_data(times, acc[:, i])
            line.set_linewidth(1.5)

        # update gyro (set data only)
        for i, line in enumerate(gyro_lines):
            line.set_data(times, gyr[:, i])
            line.set_linewidth(1.5)

        # dynamic y-limits: compute target range from visible data then smooth
        # accel target
        if acc.size > 0:
            a_min = float(np.min(acc[:, 0:3]))
            a_max = float(np.max(acc[:, 0:3]))
            # padding (10% of span or small absolute)
            span = max(1e-3, a_max - a_min)
            pad = max(0.1 * span, 0.1 if current_unit == 'g' else 1.0)
            target_a_min = a_min - pad
            target_a_max = a_max + pad
        else:
            # fallback depending on unit
            if current_unit == 'g':
                target_a_min, target_a_max = -2.0, 2.0
            else:
                target_a_min, target_a_max = -20.0, 20.0

        # gyro target
        if gyr.size > 0:
            g_min = float(np.min(gyr[:, 0:3]))
            g_max = float(np.max(gyr[:, 0:3]))
            g_span = max(1e-3, g_max - g_min)
            g_pad = max(0.1 * g_span, 5.0)
            target_g_min = g_min - g_pad
            target_g_max = g_max + g_pad
        else:
            target_g_min, target_g_max = -200.0, 200.0

        # smooth current limits toward targets
        accel_ylim[0] = accel_ylim[0] + ylim_alpha * (target_a_min - accel_ylim[0])
        accel_ylim[1] = accel_ylim[1] + ylim_alpha * (target_a_max - accel_ylim[1])
        gyro_ylim[0] = gyro_ylim[0] + ylim_alpha * (target_g_min - gyro_ylim[0])
        gyro_ylim[1] = gyro_ylim[1] + ylim_alpha * (target_g_max - gyro_ylim[1])

        axes[0].set_ylim(accel_ylim[0], accel_ylim[1])
        axes[1].set_ylim(gyro_ylim[0], gyro_ylim[1])

        # fix xlim to show last `window` seconds
        for ax in axes:
            ax.set_xlim(-window, 0)

        # update accel unit label if changed
        axes[0].set_ylabel(f'acc ({current_unit})')

        return accel_lines + gyro_lines

    ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    try:
        # Keep the UI alive until window is closed or KeyboardInterrupt
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        try:
            reader_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
