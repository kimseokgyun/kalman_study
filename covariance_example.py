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


is_paused = True  # 초기 상태는 일시 정지
##what is covariance 
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

    return x_esti, P , P_pred , x_pred





def what_is_kalmanfilter():

    velocity =  1.0 # m/s
    dt = 0.01 #dt
    time_duration = 10 # time
    num_samples = int(time_duration / dt)

    time = np.arange(0, 10, dt)
    z_pos_meas_save = np.zeros(num_samples)
    x_pos_esti_save = np.zeros(num_samples)
    x_vel_esti_save = np.zeros(num_samples)



    std_dev = 0.1 # 표준편차
    std_dev2 = 5.0
    noise = np.random.normal(0,std_dev2, 1000)
    # noise2 = np.random.normal(0,std_dev2, 500)
    # noise = np.concatenate([noise1, noise2])
    velocity_data = velocity + noise
    # x= np.array([[0.],[1.0]])
    p= np.array([[1,0],[0,1]])
    H = np.array([[1, 0]])
    # Prediction Noise ~ process Noise
    Q = np.array([[1, 0],
              [0, 1]])
    # Measurement Noise ~ Observation Noise
    R = np.array([[1]])
    # constant speed model 
    # p_{k} = p_{k-1} + v_{k-1}
    # v_{k} = v{k}  
    A = np.array([[1, dt],
                 [0, 1]])
    

    x_0 = np.array([0, 1.0])  # position and velocity
    P_0 = 5 * np.eye(2) #5 
    x_esti = None
    P_predic = None,None
    x_pred = None
    for i in range(num_samples):
        if i ==500 :
            print('now')
        z_meas = velocity_data[i]
        if i == 0:
            x_esti, P = x_0, P_0
        else:
            x_esti, P ,P_predic,x_pred= kalman_filter(z_meas, x_esti, P,A,H,R,Q)
        z_pos_meas_save[i] = z_meas
        x_pos_esti_save[i] = x_esti[0]
        x_vel_esti_save[i] = x_esti[1]

        print (x_esti)
        print(x_pred)

 
        ##Visual
        plt.figure(figsize=(10, 6))  # 전체 그림 설정
        n = np.arange(-10, 10, 0.001) 
        plt.grid()
        plt.title('position graph')
        plt.xlabel('position')
        plt.ylabel('Density')
        
        plt.plot(n, norm.pdf(n, loc=z_meas, scale=np.sqrt(R[0])),color = 'blue' ,label = 'observe - velocity')
        # plt.plot(n, norm.pdf(n, loc=x_esti[0], scale=np.sqrt(P[0][0])),label = 'esti - position')
        plt.plot(n, norm.pdf(n, loc=x_esti[1], scale=np.sqrt(P[1][1])),color = 'green',label ='esti - velocity')
        
        if i > 1 :
            plt.plot(n, norm.pdf(n, loc=x_pred[1], scale=np.sqrt(P_predic[1][1])),color = 'orange',label = 'predict - velocity')  
            print(x_esti[1])
            print(x_pred[1])         
        plt.axvline(x=1.0, color='red', label='True Velocity')
        plt.axvline(x=z_meas, color ='blue' , linestyle ='--')
        plt.axvline(x=x_esti[1], color ='green' , linestyle ='--')
        if i > 1 : 
            plt.axvline(x=x_pred[1], color ='orange' , linestyle ='--')

        plt.legend() 
        plt.show()



def what_is_covariance1():
    #x_1
    #       [position,
    #       veloicty]
    #p_1
    #covariance matrix [(sigma)^2]
    #       [p_p , p_v]
    #       [v_p , v_v]
    ##sigma = 1.0 
    ##not p_v , v_p
    dt = 1.0 
    x_1= np.array([[0.],[1.0]])
    p_1 =np.array([[1,0],[0,1]]) #우리가 모델링한 covarinace matrix 

    plt.figure(figsize=(10, 6))  # 전체 그림 설정
    plt.subplot(1,2,1)
    n = np.arange(-5, 5, 0.001) 
    plt.grid()
    plt.plot(n, norm.pdf(n, loc=x_1[0][0], scale=np.sqrt(p_1[0][0])))
    plt.title('position graph')
    plt.xlabel('position')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.title('Velocity graph')
    n = np.arange(-5, 5, 0.001) 
    plt.plot(n, norm.pdf(n, loc=x_1[1][0], scale=np.sqrt(p_1[1][1])))
    plt.xlabel('velocity')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    plt.show()




def what_is_covariance2():
    # 평균 및 공분산 행렬 정의
    mean = [0, 0]  # 위치와 속도의 평균
    covariance = [[1, 0],  # pp, pv
                [0, 1]]  # vp, vv
    #속도와 위치는 선형관계에 존재
    #등속 모델의 공분산 행렬 선언
    # covariance2 = [[1*(dt**2),1*dt],[dt*1,1]]
    #sigma =1 , dt = 0.5 
    sigma = 1
    dt = 0.5 
    
    covariance2 = [[(sigma**2)*(dt**2),(sigma**2)*dt],[dt*(sigma**2),(sigma**2)]]
    # covariance2 = [[2,3],[3,7]]
    # 격자 생성
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))

    # 다변량 정규 분포 생성
    rv = multivariate_normal(mean, covariance)

    # 등고선 플롯으로 2D 가우시안 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.subplot(1,2,1)
    contour = plt.contourf(x, y, rv.pdf(pos), levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('2D Gaussian Distribution of Position and Velocity')
    plt.grid(True)
    

    # rv2 = sp.stats.multivariate_normal(mean,covariance2, allow_singular=True)
    rv2 = multivariate_normal(mean, covariance2, allow_singular=True)
    plt.subplot(1,2,2)
    plt.grid(True)
    contou2 = plt.contourf(x, y, rv2.pdf(pos), levels=50, cmap='viridis')
    plt.colorbar(contou2)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('2D Gaussian Distribution of Position and Velocity')
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    plt.show()

def main(args=None):

    # what_is_kalmanfilter()
    # what_is_covariance1()
    what_is_covariance2()


    


    # A = np.array([[1, dt],
    #           [0, 1]])   

if __name__ == '__main__':

    main()