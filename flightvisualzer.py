import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# CSVファイルの読み込み（最初の300行に限定）
csv_file = 'data/KSE/flightdata.csv'
data = pd.read_csv(csv_file, header=None, nrows=300)

# サンプリング周波数
sampling_frequency = 7  # 例として7Hzと仮定

# データの分割
acc_data = data.iloc[:, 0:3].values
gyro_data = data.iloc[:, 3:6].values
mag_data = data.iloc[:, 6:9].values

# 地磁気データをスケール補正
mag_data = mag_data / 16

# ジャイロスコープデータを度/秒（°/s）からラジアン/秒（rad/s）に変換
gyro_data = np.deg2rad(gyro_data)

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

# 重力の影響を取り除いた加速度データの計算
g = np.array([0, 0, 9.8])  # 重力加速度
corrected_acc_data = np.zeros_like(acc_data)
for i in range(len(time_steps)):
    rot = R.from_quat(quaternions[i])
    acc_corrected = acc_data[i] - rot.apply(g)
    corrected_acc_data[i] = acc_corrected

# 位置と速度の計算
positions = np.zeros((len(time_steps), 3))
velocities = np.zeros((len(time_steps), 3))
for i in range(1, len(time_steps)):
    dt = time_steps[i] - time_steps[i-1]
    acc = corrected_acc_data[i-1]
    rot = R.from_quat(quaternions[i-1])
    acc_world = rot.apply(acc)
    velocities[i] = velocities[i-1] + acc_world * dt
    positions[i] = positions[i-1] + velocities[i] * dt

# 結果のプロット
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory')
ax.set_title('3D Trajectory')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
plt.show()
