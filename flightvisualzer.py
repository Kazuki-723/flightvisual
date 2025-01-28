import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# CSVファイルの読み込み
csv_file = 'data/KSE/flightdata.csv'
data = pd.read_csv(csv_file, header=None, nrows=500)

# サンプリング周波数
sampling_frequency = 8

# データの分割
acc_data = data.iloc[:, 0:3].values
gyro_data = data.iloc[:, 3:6].values
mag_data = data.iloc[:, 6:9].values

# 時間ステップの計算
time_steps = np.arange(len(acc_data)) / sampling_frequency

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
    omega = gyro_data[i-1]
    q = update_quaternion(q, omega, dt)
    quaternions.append(q)

quaternions = np.array(quaternions)

# 位置と速度の計算
positions = np.zeros((len(time_steps), 3))
velocities = np.zeros((len(time_steps), 3))
for i in range(1, len(time_steps)):
    dt = time_steps[i] - time_steps[i-1]
    acc = acc_data[i-1]
    rot = R.from_quat(quaternions[i-1])
    acc = rot.apply(acc)
    velocities[i] = velocities[i-1] + acc * dt
    positions[i] = positions[i-1] + velocities[i] * dt

# 結果のプロット
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(time_steps, positions[:, 0], label='X')
ax[0].plot(time_steps, positions[:, 1], label='Y')
ax[0].plot(time_steps, positions[:, 2], label='Z')
ax[0].set_title('Positions')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Position [m]')
ax[0].legend()
ax[0].grid()

ax[1].plot(time_steps, velocities[:, 0], label='X')
ax[1].plot(time_steps, velocities[:, 1], label='Y')
ax[1].plot(time_steps, velocities[:, 2], label='Z')
ax[1].set_title('Velocities')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Velocity [m/s]')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()
