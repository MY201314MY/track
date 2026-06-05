from dataclasses import dataclass

import imufusion
import matplotlib.pyplot as pyplot
import numpy
from matplotlib import animation
from scipy.interpolate import interp1d

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = numpy.genfromtxt("walk.csv", delimiter=",", skip_header=1)

sample_rate = 400  # 400 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
# Instantiate AHRS algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(
    imufusion.CONVENTION_NWU,
    0.5,  # gain
    2000,  # gyroscope range
    10,  # acceleration rejection
    0,  # magnetic rejection
    5 * sample_rate,  # rejection timeout = 5 seconds
)

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array(
        [
            ahrs_internal_states.acceleration_error,
            ahrs_internal_states.accelerometer_ignored,
            ahrs_internal_states.acceleration_recovery_trigger,
        ]
    )

    acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s

# Identify moving periods
is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

margin = int(0.1 * sample_rate)  # 100 ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index : (index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin) : index])  # add trailing margin

# Calculate velocity (includes integral drift)
velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# Find start and stop indices of each moving period
is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])


@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1


is_moving_periods = []
is_moving_period = IsMovingPeriod()

for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
        if is_moving_diff[index] == 1:
            is_moving_period.start_index = index

    elif is_moving_period.stop_index == -1:
        if is_moving_diff[index] == -1:
            is_moving_period.stop_index = index
            is_moving_periods.append(is_moving_period)
            is_moving_period = IsMovingPeriod()

# Remove integral drift from velocity
velocity_drift = numpy.zeros((len(timestamp), 3))

for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]

    t_new = timestamp[start_index : (stop_index + 1)]

    velocity_drift[start_index : (stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index : (stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index : (stop_index + 1), 2] = interp1d(t, z)(t_new)

velocity = velocity - velocity_drift


# Calculate position
position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]


# Print error as distance between start and final positions
print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")

# ==========================
# XY轨迹图
# ==========================

pyplot.figure(figsize=(8, 8))

pyplot.plot(
    position[:, 0],
    position[:, 1],
    linewidth=2,
    label="Trajectory"
)

# 起点
pyplot.scatter(
    position[0, 0],
    position[0, 1],
    marker="o",
    s=100,
    label="Start"
)

# 终点
pyplot.scatter(
    position[-1, 0],
    position[-1, 1],
    marker="x",
    s=100,
    label="End"
)

pyplot.xlabel("X Position (m)")
pyplot.ylabel("Y Position (m)")
pyplot.title("2D Trajectory")
pyplot.axis("equal")      # 保持比例
pyplot.grid(True)
pyplot.legend()

# 在第一幅图上添加一个“实时点”，显示第二幅图（窗口当前时刻）对应的全局坐标
traj_ax = pyplot.gca()
live_point = traj_ax.scatter(position[0, 0], position[0, 1], c='magenta', s=80, label='Live')
traj_ax.legend()

# 动态显示最近5秒的相对轨迹（对齐到窗口起始朝向），便于识别左转/右转的局部模式
window_duration = 5.0  # seconds
window_samples = int(window_duration * sample_rate)
anim_step = max(1, int(sample_rate / 20))  # 每帧跨越的样本数，约20 FPS

fig_anim, ax_anim = pyplot.subplots(figsize=(5, 5))
line_anim, = ax_anim.plot([], [], lw=2, label='Relative trajectory')
start_scatter = ax_anim.scatter([], [], c='green', s=50, label='Window start')
end_scatter = ax_anim.scatter([], [], c='red', s=50, label='Now')

ax_anim.set_title('Last 5s relative trajectory (aligned)')
ax_anim.set_xlabel('X (m)')
ax_anim.set_ylabel('Y (m)')
ax_anim.set_xlim(-5, 5)
ax_anim.set_ylim(-5, 5)
ax_anim.grid()
ax_anim.legend()

# add time and motion-status text (显示在图像左上角)
time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=10, va='top')
status_text = ax_anim.text(0.02, 0.88, '', transform=ax_anim.transAxes, fontsize=10, va='top')

# 将窗口内点云平移到起点并以起始运动方向对齐到x轴
# 改进：允许传入平滑后的角度，避免帧间突然旋转
last_angle = None
angle_alpha = 0.2  # 平滑因子 (0..1)，值越小旋转越平滑

def rotate_align(points, rotation=None):
    # rotation: 旋转角（弧度），将 points (以 points[0] 为原点) 旋转 rotation，使终点朝向 +Y
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

def init_anim():
    line_anim.set_data([], [])
    start_scatter.set_offsets(numpy.empty((0, 2)))
    end_scatter.set_offsets(numpy.empty((0, 2)))
    # 初始化第一幅图的实时点为空
    live_point.set_offsets(numpy.empty((0, 2)))
    time_text.set_text('')
    status_text.set_text('')
    return line_anim, start_scatter, end_scatter, live_point, time_text, status_text

def update_anim(frame):
    global last_angle
    idx = min(frame * anim_step, len(timestamp) - 1)
    start_idx = max(0, idx - window_samples + 1)
    pts = position[start_idx: idx + 1, :2]
    if len(pts) == 0:
        line_anim.set_data([], [])
        return line_anim, start_scatter, end_scatter

    # 用窗口起点->终点向量计算整体朝向，目标为 +Y (pi/2)
    if len(pts) >= 2:
        vec = pts[-1] - pts[0]
        raw_angle = numpy.arctan2(vec[1], vec[0])
    else:
        raw_angle = last_angle if last_angle is not None else 0.0

    # 需要旋转的角度，使当前向量指向 +Y
    rotation_needed = (numpy.pi / 2.0) - raw_angle

    # 平滑旋转（处理周期跳变）
    if last_angle is None:
        smoothed_rotation = rotation_needed
    else:
        diff = rotation_needed - last_angle
        diff = (diff + numpy.pi) % (2 * numpy.pi) - numpy.pi
        smoothed_rotation = last_angle + angle_alpha * diff

    last_angle = smoothed_rotation

    pts_aligned = rotate_align(pts, rotation=smoothed_rotation)
    xs, ys = pts_aligned[:, 0], pts_aligned[:, 1]

    line_anim.set_data(xs, ys)
    # 起点在对齐后的坐标应为 (0,0)
    start_scatter.set_offsets([[xs[0], ys[0]]])
    end_scatter.set_offsets([[xs[-1], ys[-1]]])

    # update time and motion status text, and print to console
    t = float(timestamp[idx])
    moving = bool(is_moving[idx])
    status_str = 'Moving' if moving else 'Stopped'
    time_text.set_text(f"t = {t:.2f} s")
    status_text.set_text(status_str)
    status_text.set_color('green' if moving else 'gray')

    # 更新第一幅图上的实时点（使用绝对轨迹坐标）
    live_point.set_offsets([[position[idx, 0], position[idx, 1]]])

    return line_anim, start_scatter, end_scatter, live_point, time_text, status_text

frames = int(numpy.ceil(len(timestamp) / anim_step))
ani = animation.FuncAnimation(fig_anim, update_anim, frames=frames, init_func=init_anim, blit=True, interval=1000/20)

# 如果你希望将每一帧导出为图像以供CNN训练，可以在这里保存帧（可选）
# 例如在 update_anim 中保存到文件或将图像渲染到 numpy 数组

pyplot.show()
