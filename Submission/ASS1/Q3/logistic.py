import pandas as pd
import numpy as np

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


df_x = pd.read_csv("./../Data/logisticX.csv", header=None)
df_y = pd.read_csv("./../Data/logisticY.csv", header=None)

x1 = np.array(df_x[0])
x2 = np.array(df_x[1])

x1 = (x1 - np.mean(x1))/(np.std(x1))
x2 = (x2 - np.mean(x2))/(np.std(x2))

x_data = np.column_stack((np.ones(len(x1)), x1,x2))
y_data = np.array(df_y[0])


def h_theta_x(theta,x):
    return 1/(1+np.exp(-np.dot(theta,x.T)))

def J(theta,x,y):
    m = len(y)
    h = h_theta_x(theta,x)
    return (1/(2*m))*(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h)))
    
    
def grad_J(theta,x,y):
    m = len(x)
    return np.dot(h_theta_x(theta,x) - y, x)/m 

def hess_J(theta,x,y):
    m = len(y)
    ht = h_theta_x(theta,x)
    v = ht*(1-ht)
    v2 = np.column_stack((v,v,v))
    x2 = v2*x
    
    H = np.dot(x2.T,x)
    return H/m


J_theta_old = 0
J_theta = 1e5

threshold=1e-10
theta_initial = np.array([0,0,0])

while abs(J_theta_old - J_theta) > threshold:
    J_theta_old = J_theta
    H_inv = np.linalg.inv(hess_J(theta_initial,x_data,y_data))
    theta_initial = theta_initial - np.dot(H_inv, grad_J(theta_initial,x_data,y_data))
    J_theta = J(theta_initial,x_data,y_data)
    
    print(f"J_theta is {J_theta}| theta is {theta_initial}, Difference is {abs(J_theta_old - J_theta)}")
    


import plotly.graph_objects as go


trace_0 = go.Scatter(x=x1[y_data == 0], y=x2[y_data == 0], mode='markers', marker=dict(
                                        color='blue', symbol='circle',size=10,
                                        line=dict(width=2,
                                        color='DarkSlateGrey')
                                        ), name='Label 0')
trace_1 = go.Scatter(x=x1[y_data == 1], y=x2[y_data == 1], mode='markers', marker=dict(
                                        color='red', symbol='x',size=10,
                                        line=dict(width=2,
                                        color='DarkSlateGrey')
                                        ), name='Label 1')

boundary_x = np.linspace(min(x1), max(x1), 100)
boundary_y = -(theta_initial[0] + theta_initial[1] * boundary_x) / theta_initial[2]

trace_boundary = go.Scatter(x=boundary_x, y=boundary_y, mode='lines', line=dict(color='black'), name='Decision Boundary')

layout = go.Layout(title='Logistic Regression and Decision Boundary', xaxis_title='X1', yaxis_title='X2')
fig = go.Figure(data=[trace_0, trace_1, trace_boundary], layout=layout)

fig.write_html('logistic.html')

# fig.show()


    
    