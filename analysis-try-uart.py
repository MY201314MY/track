from dataclasses import dataclass

import threading
import time
import imufusion
import matplotlib.pyplot as pyplot
import numpy
from matplotlib import animation
from scipy.interpolate import interp1d
import serial
from collections import deque

# Serial configuration
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 921600

# Algorithm parameters
sample_rate = 100  # nominal sample rate (Hz)
# Motion detection threshold (magnitude of earth-frame acceleration)
# Increase this if you see drift while stationary. Default was 3.0; raised to 6.0.
motion_threshold = 10.0  # m/s^2
margin_seconds = 0.1  # 100 ms margin
margin_samples = int(margin_seconds * sample_rate)

# Shared buffers (raw)
lock = threading.Lock()
timestamps = []
gyro_raw = []
accel_raw = []

# Processed buffers
euler_list = []
internal_states_list = []
acceleration_list = []
is_moving_list = []
velocity_list = []
position_list = []

# Moving-period tracking
@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1

is_moving_periods = []
current_moving_start = -1

# AHRS / offset instances (shared)
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()
ahrs.settings = imufusion.Settings(
    imufusion.CONVENTION_NWU,
    0.5,  # gain
    0,  # gyroscope range (0 = no limit)
    0,  # acceleration rejection (0 = disabled)
    0,  # magnetic rejection
    5 * sample_rate,  # rejection timeout = 5 seconds
)

# Serial reader thread
stop_event = threading.Event()

def serial_reader():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except Exception as e:
        print(f"Failed to open serial port {SERIAL_PORT}: {e}")
        stop_event.set()
        return

    print(f"Serial port {SERIAL_PORT} opened at {BAUDRATE} bps")
    try:
        while not stop_event.is_set():
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
            with lock:
                timestamps.append(vals[0])
                gyro_raw.append(vals[1:4])
                accel_raw.append(vals[4:7])
    except Exception as e:
        print(f"Serial reader error: {e}")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        stop_event.set()

# Start serial reader thread
t = threading.Thread(target=serial_reader, daemon=True)
t.start()

processed_count = 0

# Helper: recompute positions from a start index using current velocity_list
def recompute_positions_from(start_idx):
    # assumes velocity_list is already corrected up to end
    if start_idx <= 0:
        prev_pos = numpy.array([0.0, 0.0, 0.0])
        prev_time = timestamps[0] if len(timestamps) > 0 else 0.0
        i0 = 0
    else:
        prev_pos = position_list[start_idx - 1].copy()
        prev_time = timestamps[start_idx - 1]
        i0 = start_idx

    for i in range(i0, len(timestamps)):
        dt = timestamps[i] - prev_time
        if dt < 0:
            dt = 0.0
        vel = velocity_list[i]
        new_pos = prev_pos + dt * vel
        if i < len(position_list):
            position_list[i] = new_pos
        else:
            position_list.append(new_pos)
        prev_pos = new_pos
        prev_time = timestamps[i]

# Main processing: process newly arrived samples incrementally

def process_new_samples():
    global processed_count, current_moving_start
    with lock:
        n_total = len(timestamps)
    while processed_count < n_total:
        with lock:
            t = timestamps[processed_count]
            gyro = numpy.array(gyro_raw[processed_count], dtype=float)
            accel = numpy.array(accel_raw[processed_count], dtype=float)
            # compute dt purely from received timestamps
            if processed_count == 0:
                dt = 0.0
            else:
                dt = timestamps[processed_count] - timestamps[processed_count - 1]
                if dt < 0.0:
                    # Received non-monotonic timestamp, clamp to zero (no integration)
                    dt = 0.0

        # apply offset and AHRS update
        gyro_cal = offset.update(gyro)
        ahrs.update_no_magnetometer(gyro_cal, accel, dt)

        euler = ahrs.quaternion.to_euler()
        ahrs_internal_states = ahrs.internal_states
        internal_states = numpy.array(
            [
                ahrs_internal_states.acceleration_error,
                ahrs_internal_states.accelerometer_ignored,
                ahrs_internal_states.acceleration_recovery_trigger,
            ]
        )
        acc_earth = 9.81 * ahrs.earth_acceleration  # m/s^2

        # determine motion using magnitude (simple, uses instantaneous acc)
        moving_now = float(numpy.sqrt(acc_earth.dot(acc_earth))) > motion_threshold

        # append processed values
        euler_list.append(euler)
        internal_states_list.append(internal_states)
        acceleration_list.append(acc_earth)
        is_moving_list.append(moving_now)

        # apply leading margin smoothing (based on recent samples)
        i = processed_count
        start_check = max(0, i - margin_samples + 1)
        smoothed_moving = any(is_moving_list[start_check - processed_count : len(is_moving_list)]) if False else None
        # simpler: check last margin entries in acceleration list
        smoothed_moving = False
        lookback = min(len(acceleration_list), margin_samples)
        for j in range(-lookback, 0):
            if numpy.linalg.norm(acceleration_list[j]) > motion_threshold:
                smoothed_moving = True
                break

        # compute velocity and position
        if len(velocity_list) == 0:
            prev_vel = numpy.array([0.0, 0.0, 0.0])
        else:
            prev_vel = velocity_list[-1]

        if smoothed_moving:
            new_vel = prev_vel + dt * acc_earth
        else:
            new_vel = prev_vel.copy()

        velocity_list.append(new_vel)

        if len(position_list) == 0:
            prev_pos = numpy.array([0.0, 0.0, 0.0])
        else:
            prev_pos = position_list[-1]
        new_pos = prev_pos + dt * new_vel
        position_list.append(new_pos)

        # detect moving period boundaries using smoothed flag
        prev_moving = is_moving_list[-2] if len(is_moving_list) >= 2 else False
        if not prev_moving and smoothed_moving:
            # start
            current_moving_start = processed_count
        if prev_moving and not smoothed_moving and current_moving_start != -1:
            # stop — perform drift correction on the period [start, processed_count]
            start_idx = current_moving_start
            stop_idx = processed_count
            # linear drift from velocity[start] to velocity[stop]
            v_start = velocity_list[start_idx - processed_count + len(velocity_list) - 1] if False else velocity_list[start_idx]
            v_stop = velocity_list[stop_idx]
            # create linear correction for each axis and subtract
            t0 = timestamps[start_idx]
            t1 = timestamps[stop_idx]
            if t1 - t0 != 0:
                for axis in range(3):
                    x = [v_start[axis], v_stop[axis]]
                    tseg = [t0, t1]
                    t_new = timestamps[start_idx : stop_idx + 1]
                    drift = interp1d(tseg, x)(t_new)
                    # apply correction
                    for k, idx_global in enumerate(range(start_idx, stop_idx + 1)):
                        velocity_list[idx_global] = velocity_list[idx_global].copy()
                        velocity_list[idx_global][axis] -= drift[k]
                # recompute positions from start_idx onward
                recompute_positions_from(start_idx)
            is_moving_periods.append(IsMovingPeriod(start_index=start_idx, stop_index=stop_idx))
            current_moving_start = -1

        processed_count += 1

# Setup plotting (similar to original)
pyplot.ion()
fig = pyplot.figure(figsize=(12, 6))
ax_traj = fig.add_subplot(1, 2, 1)
ax_rel = fig.add_subplot(1, 2, 2)

traj_line, = ax_traj.plot([], [], linewidth=2, label='Trajectory')
start_sc = ax_traj.scatter([], [], marker='o', s=100, label='Start')
end_sc = ax_traj.scatter([], [], marker='x', s=100, label='End')
live_sc = ax_traj.scatter([], [], c='magenta', s=80, label='Live')
past_sc = ax_traj.scatter([], [], c='cyan', s=60, label='T-5s')
ax_traj.set_title('XY Trajectory')
ax_traj.set_xlabel('X (m)')
ax_traj.set_ylabel('Y (m)')
ax_traj.axis('equal')
ax_traj.grid()
ax_traj.legend()

line_anim, = ax_rel.plot([], [], lw=2, label='Relative trajectory')
start_rel = ax_rel.scatter([], [], c='green', s=50, label='Window start')
end_rel = ax_rel.scatter([], [], c='red', s=50, label='Now')
ax_rel.set_title('Last 5s relative trajectory (aligned)')
ax_rel.set_xlabel('X (m)')
ax_rel.set_ylabel('Y (m)')
ax_rel.grid()
ax_rel.legend()

time_text = ax_rel.text(0.02, 0.95, '', transform=ax_rel.transAxes, fontsize=10, va='top')
status_text = ax_rel.text(0.02, 0.88, '', transform=ax_rel.transAxes, fontsize=10, va='top')

last_angle = None
angle_alpha = 0.2
window_duration = 5.0
window_samples = int(window_duration * sample_rate)
anim_step = max(1, int(sample_rate / 20))

# helper rotate_align

def rotate_align(points, rotation=None):
    if len(points) < 1:
        return points.copy()
    pts = points - points[0]
    if rotation is None:
        return pts[:, :2].copy()
    c = numpy.cos(rotation)
    s = numpy.sin(rotation)
    R = numpy.array([[c, -s], [s, c]])
    pts2 = pts[:, :2].dot(R.T)
    return pts2

# animation init

def init_anim():
    traj_line.set_data([], [])
    start_sc.set_offsets(numpy.empty((0, 2)))
    end_sc.set_offsets(numpy.empty((0, 2)))
    live_sc.set_offsets(numpy.empty((0, 2)))
    past_sc.set_offsets(numpy.empty((0, 2)))
    line_anim.set_data([], [])
    start_rel.set_offsets(numpy.empty((0, 2)))
    end_rel.set_offsets(numpy.empty((0, 2)))
    time_text.set_text('')
    status_text.set_text('')
    return (traj_line, start_sc, end_sc, live_sc, past_sc, line_anim, start_rel, end_rel, time_text, status_text)

# animation update: process new samples then update plots

def update(frame):
    global last_angle
    # process any new samples
    process_new_samples()

    if len(position_list) == 0:
        return init_anim()

    pos = numpy.array(position_list)

    # use real time to show only the last minute in the main trajectory
    if len(timestamps) > 0:
        t_now = timestamps[-1]
        main_window_duration = 60.0  # seconds to show on main trajectory
        main_start_time = t_now - main_window_duration
        times_arr = numpy.array(timestamps)
        main_start_idx = int(numpy.searchsorted(times_arr, main_start_time, side='left'))
        main_pos = pos[main_start_idx:]
        if main_pos.shape[0] == 0:
            main_pos = pos[-1:].copy()
    else:
        t_now = 0.0
        main_pos = pos

    # update main trajectory with only recent data
    try:
        traj_line.set_data(main_pos[:, 0], main_pos[:, 1])
        start_sc.set_offsets([[main_pos[0, 0], main_pos[0, 1]]])
        end_sc.set_offsets([[main_pos[-1, 0], main_pos[-1, 1]]])

        # update trajectory axes dynamically to fit the recent minute
        min_x, max_x = float(numpy.min(main_pos[:, 0])), float(numpy.max(main_pos[:, 0]))
        min_y, max_y = float(numpy.min(main_pos[:, 1])), float(numpy.max(main_pos[:, 1]))
        pad_x = max(0.5, 0.05 * (max_x - min_x + 1e-6))
        pad_y = max(0.5, 0.05 * (max_y - min_y + 1e-6))
        ax_traj.set_xlim(min_x - pad_x, max_x + pad_x)
        ax_traj.set_ylim(min_y - pad_y, max_y + pad_y)
        ax_traj.set_aspect('equal', adjustable='box')
    except Exception:
        # fallback: show full trajectory if something goes wrong
        traj_line.set_data(pos[:, 0], pos[:, 1])
        start_sc.set_offsets([[pos[0, 0], pos[0, 1]]])
        end_sc.set_offsets([[pos[-1, 0], pos[-1, 1]]])

    # live and past points
    # determine past index by real time (5s before latest sample)
    if len(timestamps) > 0:
        t_now = timestamps[-1]
        start_time = t_now - window_duration
        times_arr = numpy.array(timestamps)
        # find first index >= start_time
        start_idx = int(numpy.searchsorted(times_arr, start_time, side='left'))
        prev_idx = max(0, start_idx)
    else:
        prev_idx = 0
        start_idx = max(0, len(pos) - window_samples)

    live_sc.set_offsets([[pos[-1, 0], pos[-1, 1]]])
    past_sc.set_offsets([[pos[prev_idx, 0], pos[prev_idx, 1]]])

    # relative window based on timestamps (last window_duration seconds)
    pts = pos[start_idx:, :2]
    if len(pts) == 0:
        line_anim.set_data([], [])
        start_rel.set_offsets(numpy.empty((0, 2)))
        end_rel.set_offsets(numpy.empty((0, 2)))
    else:
        # determine motion direction similar to original
        if len(pts) >= 2:
            subset_len = max(2, int(len(pts) * 0.2))
            pts_sub = pts[-subset_len:]
            disp = pts_sub[-1] - pts_sub[0]
            if numpy.linalg.norm(disp) > 1e-3:
                raw_angle = numpy.arctan2(disp[1], disp[0])
            else:
                disp_full = pts[-1] - pts[0]
                if numpy.linalg.norm(disp_full) > 1e-3:
                    raw_angle = numpy.arctan2(disp_full[1], disp_full[0])
                else:
                    pts_rel = pts - pts[0]
                    cov = numpy.cov(pts_rel.T)
                    evals, evecs = numpy.linalg.eigh(cov)
                    principal = evecs[:, numpy.argmax(evals)]
                    if last_angle is not None:
                        last_dir = numpy.array([numpy.cos(last_angle), numpy.sin(last_angle)])
                        if numpy.dot(principal, last_dir) < 0:
                            principal = -principal
                    raw_angle = numpy.arctan2(principal[1], principal[0])
        else:
            raw_angle = last_angle if last_angle is not None else 0.0

        rotation_needed = (numpy.pi / 2.0) - raw_angle
        if last_angle is None:
            smoothed_rotation = rotation_needed
        else:
            diff = rotation_needed - last_angle
            diff = (diff + numpy.pi) % (2 * numpy.pi) - numpy.pi
            smoothed_rotation = last_angle + angle_alpha * diff
        last_angle = smoothed_rotation

        pts_aligned = rotate_align(pts, rotation=smoothed_rotation)

        # do not subtract mean_x here; rotate_align already makes start at (0,0)

        xs, ys = pts_aligned[:, 0], pts_aligned[:, 1]
        line_anim.set_data(xs, ys)
        start_rel.set_offsets([[xs[0], ys[0]]])
        end_rel.set_offsets([[xs[-1], ys[-1]]])

        # autoscale relative axes based on windowed data (tight dynamic fit)
        try:
            min_x = float(numpy.min(xs))
            max_x = float(numpy.max(xs))
            min_y = float(numpy.min(ys))
            max_y = float(numpy.max(ys))
            pad_x = max(0.2, 0.1 * (max_x - min_x + 1e-6))
            pad_y = max(0.2, 0.1 * (max_y - min_y + 1e-6))
            ax_rel.set_xlim(min_x - pad_x, max_x + pad_x)
            ax_rel.set_ylim(min_y - pad_y, max_y + pad_y)
            ax_rel.set_aspect('equal', adjustable='box')
        except Exception:
            # fallback to a small symmetric window if numeric fails
            lim = 3.0
            ax_rel.set_xlim(-lim, lim)
            ax_rel.set_ylim(-lim, lim)

    # update time and motion status
    t_now = timestamps[-1] if len(timestamps) > 0 else 0.0
    moving_now = any(is_moving_list[max(0, len(is_moving_list) - margin_samples):])
    status_str = 'Moving' if moving_now else 'Stopped'
    time_text.set_text(f"t = {t_now:.2f} s")
    status_text.set_text(status_str)
    status_text.set_color('green' if moving_now else 'gray')

    return (traj_line, start_sc, end_sc, live_sc, past_sc, line_anim, start_rel, end_rel, time_text, status_text)

# run animation
frames = range(10**9)  # large finite iterator to avoid unbounded caching
ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init_anim,
    blit=False,
    interval=50,
    cache_frame_data=False,
    save_count=1000,
)

try:
    # use blocking show to ensure the animation runs until the window is closed
    pyplot.show(block=True)
finally:
    stop_event.set()
    t.join(timeout=1.0)

# When the window closes, print final error
if len(position_list) > 0:
    final_err = numpy.sqrt(position_list[-1].dot(position_list[-1]))
    print(f"Final error: {final_err:.3f} m")
else:
    print("No position computed.")
