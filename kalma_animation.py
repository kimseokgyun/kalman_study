#! /usr/bin/env python3

import time
import serial
import sys

import threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from threading import Timer
import datetime as dt
import numpy as np
import scipy as sp
from scipy.stats import norm 
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from matplotlib.animation import FuncAnimation


# 초기 설정
velocity = 1.0  # m/s
reverse_velocity = -1.0 #m/s
dt = 0.1  # time step
time_duration = 10  # time duration
num_samples = int(time_duration / dt)
num_samples_half = int(num_samples / 2)
# 시간, 잡음 생성
std_dev = 0.1 # 표준편차
# std_dev2 = 5.0
noise = np.random.normal(0,std_dev, num_samples_half)
noise = np.random.normal(0,std_dev, num_samples_half)
velocity_data1 = velocity + noise
velocity_data2 = reverse_velocity + noise 
velocity_data = np.concatenate([velocity_data1, velocity_data2])


# 각 시간 단계에서의 위치를 저장할 리스트
initial_position =0.0
positions = [initial_position]
# 각 dt 당 위치를 계산
for i in range(1, num_samples):
    # 이전 위치에 속도 * 시간 간격(dt)을 더해 다음 위치를 계산
    next_position = positions[-1] + velocity * dt
    positions.append(next_position)


# Kalman 필터 초기값
x_0 = np.array([0, 1.0])  # 초기 위치와 속도
P_0 = np.array([[10,0],[0,10]])
A = np.array([[1, dt], [0, 1]])
H = np.array([[0,1]])
# Process Noise
# Q = np.array([[0.5, 0], [0, 0.5]])
Q = np.array([[0.5*(dt**2),0.5*dt],[0.5*dt,0.5]])
# 
R = np.array([[1]])   



is_paused= True
x_esti = None
P_predic = None
x_pred = None
K_vals = None
K_vals_p = None
def kalman_filter(z_meas, x_esti, P,A,H,R,Q): 
    """Kalman Filter Algorithm."""
    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P , P_pred , x_pred , K

def on_press(event):
    global is_paused
    if event.key == ' ':
        if is_paused:
            is_paused = False  # 재생 상태로 변경
        else:
            is_paused = True  # 일시 정지 상태로 변경
# 애니메이션 업데이트 함수





def main(args=None):

    # 애니메이션을 위한 준비
    fig, axs = plt.subplots(2, 2, figsize=(12, 6)) 
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    n = np.arange(-10, 10, 0.001)
    y_range = np.linspace(0, 0.5, 100)  # y 축 값의 범위 설정, 0부터 0.5까지 100개의 포인트
    x_true_velocity = np.full_like(y_range, velocity)  # x 축 값으로 True 속도 값을 고정
    line_true, = ax1.plot(x_true_velocity, y_range, color='red', label='True Velocity', linestyle='--')  # 세로 직선 그래프
    line_meas, = ax1.plot([], [], color='blue', label='Observe - Velocity')
    line_esti, = ax1.plot([], [], color='green', label='Esti - Velocity')
    line_pred, = ax1.plot([], [], color='orange', label='Predict - Velocity')




    ax1.set_xlim(-10, 10)
    ax1.set_ylim(0, 1.0)
    ax1.grid()
    ax1.legend()
    ax1.set_title('Kalman Filter: Velocity Estimation')
    ax1.set_xlabel('Velocity')
    ax1.set_ylabel('Probability Density')

    # Kalman Gain (K) 값을 시각화하기 위한 서브플롯 설정
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1)  # K 값의 범위에 따라 조정할 수 있습니다.
    ax2.grid()
    ax2.set_title('Kalman Gain (K) over Time')
    ax2.set_xlabel('N)')
    ax2.set_ylabel('Kalman Gain (K)')
    line_k, = ax2.plot([], [], color='purple', label='Kalman Gain (K)')  # K 값을 나타낼 라인 생성
    ax2.legend()


    # 위치(Position) 그래프를 위한 초기 설정
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(0, 0.5)  # K 값의 범위에 따라 조정할 수 있습니다.
    ax3.grid()
    ax3.set_title('Position over Time')
    ax3.set_xlabel('Position [m]')
    ax3.set_ylabel('Probability Densit')
    line_esti_pos, = ax3.plot([],[], color = 'green' , label='esti - Position')
    line_pred_pos, = ax3.plot([], [], color='orange', label='Predict - Position')   
    # line_true_pos, = ax3.plot(positions, y_range, color='red', label='True Velocity', linestyle='--')  # 세로 직선 그래프
    ax3.legend()

    # Kalman Gain (K) 값을 시각화하기 위한 서브플롯 설정
    ax4.set_xlim(0, 1000)
    ax4.set_ylim(0, 1)  # K 값의 범위에 따라 조정할 수 있습니다.
    ax4.grid()
    ax4.set_title('Kalman Gain (K) over Time')
    ax4.set_xlabel('N)')
    ax4.set_ylabel('Kalman Gain (K)')
    line_k_position, = ax4.plot([], [], color='purple', label='Kalman Gain (K)')  # K 값을 나타낼 라인 생성
    ax4.legend()

    def update(frame):
        global x_esti, P, P_predic, x_pred, is_paused , K_vals , K_vals_p
        if not is_paused:  # 스페이스바가 눌려서 애니메이션이 진행 중이라면
            #version1
            z_meas = velocity_data[frame]

            # ##version2################################### 
            # z_meas = np.array([[0,0],[0,0]]) 
            # z_meas[1][1] =velocity_data[frame]
            # # 첫 번째 프레임에서는 x_esti가 None일 수 있으므로, 이 경우를 처리
            # if frame == 0:
            #     # 첫 번째 프레임에서 z_meas[0]에 대한 초기값을 설정할 수 있습니다.
            #     # 예를 들어, 초기 위치를 0으로 가정할 수 있습니다.
            #     z_meas[0][0] = 0
            # elif x_esti is not None:
            #     # 첫 번째 프레임이 아니고 x_esti가 None이 아닐 때만 x_esti[0]을 할당
            #     z_meas[0][0] = x_esti[0]

            ##############################################

            if frame == 0 or x_esti is None:
                x_esti, P = x_0, P_0
                P_predic, x_pred = None, None  # 첫 프레임 또는 x_esti가 정의되지 않았을 경우 초기화
                K_vals=[]
                K_vals_p = []
            else:
                x_esti, P, P_predic, x_pred , K_i = kalman_filter(z_meas, x_esti, P, A, H, R, Q)
                K_vals.append(K_i[1])
                K_vals_p.append(K_i[0])
            print(x_esti)
            line_esti_pos.set_data(n, norm.pdf(n,loc=x_esti[0], scale =np.sqrt(P[0][0])))
            ##version1
            line_meas.set_data(n, norm.pdf(n, loc=z_meas, scale=np.sqrt(R[0])))
            ##version2 
            # line_meas.set_data(n, norm.pdf(n, loc=z_meas[1][1], scale=np.sqrt(R[0])))


            line_esti.set_data(n, norm.pdf(n, loc=x_esti[1], scale=np.sqrt(P[1][1])))
            
            if frame > 0 and P_predic is not None:
                line_pred.set_data(n, norm.pdf(n, loc=x_pred[1], scale=np.sqrt(P_predic[1][1])))
                line_pred_pos.set_data(n, norm.pdf(n,loc=x_pred[0], scale= np.sqrt(P_predic[0][0])))
            else:
                line_pred.set_data([], [])
                line_pred_pos.set_data([],[])
            if K_vals:
                line_k.set_data(np.arange(0, len(K_vals)), K_vals)
                line_k_position.set_data(np.arange(0,len(K_vals_p)),K_vals_p)


        return line_meas, line_esti, line_pred , line_k , line_pred_pos , line_esti_pos, line_k_position

    

    fig.canvas.mpl_connect('key_press_event', on_press)  # 키보드 이벤트 리스너 등록

    ani = FuncAnimation(fig, update, frames=num_samples, blit=True, repeat=False)
 
    plt.show()   

if __name__ == '__main__':

    main()