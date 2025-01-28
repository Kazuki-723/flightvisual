import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# CSVファイルの読み込み（最初の500行に限定）
csv_file = 'data/KSE/flightdata.csv'
data = pd.read_csv(csv_file, header=None, nrows=500)

# サンプリング周波数
sampling_frequency = 7  # 例として100Hzと仮定

# 移動平均フィルタの設定
window_size = 5

# データの分割
acc_data = data.iloc[:, 0:3].values
gyro_data = data.iloc[:, 3:6].values
mag_data = data.iloc[:, 6:9].values

# 移動平均フィルタの適用
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

acc_data_filtered = np.apply_along_axis(moving_average, 0, acc_data, window_size)
gyro_data_filtered = np.apply_along_axis(moving_average, 0, gyro_data, window_size)
mag_data_filtered = np.apply_along_axis(moving_average, 0, mag_data, window_size)

# 時間ステップの計算
time_steps = np.arange(len(gyro_data_filtered)) / sampling_frequency

# 初期クォータニオン
q = R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

# クォータニオンの計算
def update_quaternion(q, omega, dt):
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 0:
        theta = omega_mag * dt
        q_delta = R.from_rotvec((omega / omega_mag) * theta).as_quat()
        q = R.from_quat(q) * R.from_quat(q_delta)
        q = q.as_quat()
    return q

quaternions = [q]
for i in range(1, len(time_steps)):
    dt = time_steps[i] - time_steps[i-1]
    omega = gyro_data_filtered[i-1]
    q = update_quaternion(q, omega, dt)
    quaternions.append(q)

quaternions = np.array(quaternions)

# クォータニオンからロール、ピッチ、ヨーの計算
euler_angles = np.zeros((len(quaternions), 3))
for i, quat in enumerate(quaternions):
    rot = R.from_quat(quat)
    euler_angles[i] = rot.as_euler('xyz', degrees=True)

roll = euler_angles[:, 0]
pitch = euler_angles[:, 1]
yaw = euler_angles[:, 2]

# 結果のプロット
fig, ax = plt.subplots(3, 1, figsize=(10, 10))

ax[0].plot(time_steps, roll, label='Roll')
ax[0].set_title('Roll')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Angle [degrees]')
ax[0].legend()
ax[0].grid()

ax[1].plot(time_steps, pitch, label='Pitch')
ax[1].set_title('Pitch')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Angle [degrees]')
ax[1].legend()
ax[1].grid()

ax[2].plot(time_steps, yaw, label='Yaw')
ax[2].set_title('Yaw')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel('Angle [degrees]')
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.show()
