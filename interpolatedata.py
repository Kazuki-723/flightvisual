import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# CSVファイルの読み込み
csv_file = 'data/KSE/newflight.csv'
data = pd.read_csv(csv_file, header=None)

# サンプリング周波数
original_sampling_frequency = 1/0.135  # 元のサンプリング周波数
target_sampling_frequency = 50  # 補完後のサンプリング周波数
gravity = np.array([0.0,0.0,0.0])

# データの分割
acc_data = data.iloc[:, 0:3].values
gyro_data = data.iloc[:, 3:6].values
mag_data = data.iloc[:, 6:9].values

#magはどうやら16で割ればいいらしい
mag_data = mag_data / 16

#rad/sへの補正
gyro_data = np.deg2rad(gyro_data)

#重力項の計算
for i in range(5):
    gravity += acc_data[i]

gravity = gravity / 5

# 時間ステップの計算
time_steps = np.arange(len(acc_data)) / original_sampling_frequency

# 新しい時間ステップの計算
new_time_steps = np.arange(time_steps[0], time_steps[-1], 1/target_sampling_frequency)

# 補完関数を作成
interpolate_acc = interp1d(time_steps, acc_data, axis=0, kind='cubic', fill_value='extrapolate')
interpolate_gyro = interp1d(time_steps, gyro_data, axis=0, kind='cubic', fill_value='extrapolate')
interpolate_mag = interp1d(time_steps, mag_data, axis=0, kind='cubic', fill_value='extrapolate')

# 新しい時間ステップに対して補完
new_acc_data = interpolate_acc(new_time_steps)
new_gyro_data = interpolate_gyro(new_time_steps)
new_mag_data = interpolate_mag(new_time_steps)

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
for i in range(1, len(new_time_steps)):
    dt = new_time_steps[i] - new_time_steps[i-1]
    omega = new_gyro_data[i-1]
    q = update_quaternion(q, omega, dt)
    quaternions.append(q)

quaternions = np.array(quaternions)

# 重力の影響を取り除いた加速度データの計算
g = gravity
g = np.array([0.0,0.0,0.0]) #重力差分を消すとき
corrected_acc_data = np.zeros_like(new_acc_data)
for i in range(len(new_time_steps)):
    rot = R.from_quat(quaternions[i])
    acc_corrected = new_acc_data[i] - rot.apply(g)
    corrected_acc_data[i] = acc_corrected

# 修正された加速度データのプロット
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(new_time_steps, corrected_acc_data[:, 0], label='Acc X')
plt.plot(new_time_steps, corrected_acc_data[:, 1], label='Acc Y')
plt.plot(new_time_steps, corrected_acc_data[:, 2], label='Acc Z')
plt.title('Corrected Acceleration Data')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.legend()
plt.grid()

# 角速度データのプロット
plt.subplot(4, 1, 2)
plt.plot(new_time_steps, new_gyro_data[:, 0], label='Gyro X')
plt.plot(new_time_steps, new_gyro_data[:, 1], label='Gyro Y')
plt.plot(new_time_steps, new_gyro_data[:, 2], label='Gyro Z')
plt.title('Gyroscope Data')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()
plt.grid()

# 地磁気データのプロット
plt.subplot(4, 1, 3)
plt.plot(new_time_steps, new_mag_data[:, 0], label='Mag X')
plt.plot(new_time_steps, new_mag_data[:, 1], label='Mag Y')
plt.plot(new_time_steps, new_mag_data[:, 2], label='Mag Z')
plt.title('Magnetometer Data')
plt.xlabel('Time [s]')
plt.ylabel('Magnetic Field [µT]')
plt.legend()
plt.grid()

#ノルムプロット
plt.subplot(4, 1, 4)
plt.plot(new_time_steps, np.sqrt(corrected_acc_data[:, 0] ** 2 + corrected_acc_data[:, 1] ** 2 + corrected_acc_data[:, 2] ** 2), label = "Acc_norm")
#plt.plot(new_time_steps, np.sqrt(new_mag_data[:, 0] ** 2 + new_mag_data[:, 1] ** 2 + new_mag_data[:, 2] ** 2), label = "Mag_norm")
plt.title('Accelation Data')
#plt.title('Magnetometer Data')
plt.xlabel('Time [s]')
plt.ylabel('Accelation Norm[m/s^2]')
#plt.ylabel('Magnetic Field Norm [µT]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

new_data = np.concatenate([new_acc_data, new_gyro_data, new_mag_data],1)
df = pd.DataFrame(new_data)

df.to_csv('interpolated_sensor_data.csv', index=False)