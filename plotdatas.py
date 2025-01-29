import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# CSVファイルの読み込み（最初の500行に限定）
csv_file = 'data/KSE/flightdata.csv'
data = pd.read_csv(csv_file, header=None, nrows=300)

# サンプリング周波数
sampling_frequency = 7  # 例として100Hzと仮定

# データの分割
acc_data = data.iloc[:, 0:3].values
gyro_data = data.iloc[:, 3:6].values
mag_data = data.iloc[:, 6:9].values

#magはどうやら16で割ればいいらしい
mag_data = mag_data / 16

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
g = np.array([0, 0, -9.8])  # 重力加速度
corrected_acc_data = np.zeros_like(acc_data)
for i in range(len(time_steps)):
    rot = R.from_quat(quaternions[i])
    acc_corrected = acc_data[i] - rot.apply(g)
    corrected_acc_data[i] = acc_corrected

# 修正された加速度データのプロット
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(time_steps, corrected_acc_data[:, 0], label='Acc X (corrected)')
plt.plot(time_steps, corrected_acc_data[:, 1], label='Acc Y (corrected)')
plt.plot(time_steps, corrected_acc_data[:, 2], label='Acc Z (corrected)')
plt.title('Corrected Acceleration Data')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.legend()
plt.grid()

# 角速度データのプロット
plt.subplot(4, 1, 2)
plt.plot(time_steps, gyro_data[:, 0], label='Gyro X')
plt.plot(time_steps, gyro_data[:, 1], label='Gyro Y')
plt.plot(time_steps, gyro_data[:, 2], label='Gyro Z')
plt.title('Gyroscope Data')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()
plt.grid()

# 地磁気データのプロット
plt.subplot(4, 1, 3)
plt.plot(time_steps, mag_data[:, 0], label='Mag X')
plt.plot(time_steps, mag_data[:, 1], label='Mag Y')
plt.plot(time_steps, mag_data[:, 2], label='Mag Z')
plt.title('Magnetometer Data')
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field [µT]')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time_steps, np.sqrt(mag_data[:, 0] ** 2 + mag_data[:, 1] ** 2 + mag_data[:, 2] ** 2), label = "Mag_norm")
plt.title('Magnetometer Data')
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field Norm [µT]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
