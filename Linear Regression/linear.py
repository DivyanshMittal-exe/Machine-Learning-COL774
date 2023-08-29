import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)




df_x = pd.read_csv("./../Data/linearX.csv", header=None)
df_y = pd.read_csv("./../Data/linearY.csv", header=None)

x_data_raw = np.array(df_x[0].to_list())
y_data = np.array(df_y[0].to_list())

# Normalise x_data_raw
x_data_original = (x_data_raw - np.mean(x_data_raw))/np.std(x_data_raw)

x_data = np.column_stack((x_data_original, np.ones(len(x_data_original))))


def perform_linear_regression(threshold = 1e-8, eta=0.01):
    J_theta_old = 0
    J_theta_new = 1e5
    
    animation_data = []
    theta_initial = np.zeros(2)
    while abs(J_theta_new-J_theta_old) > threshold:
        
        J_theta = 0
        m = len(x_data)
        for x,y in zip(x_data,y_data):
            J_theta += (y - np.sum(x * theta_initial))**2
        J_theta = J_theta/(2*m)
        
        J_theta_old = J_theta_new
        J_theta_new = J_theta   

        animation_data.append((J_theta, theta_initial[0],theta_initial[1]))

        theta_initial = theta_initial - eta * (1/m) * np.dot((np.dot(x_data, theta_initial) - y_data) , x_data)
        print(f"J_theta: {J_theta}, Difference: {abs(J_theta_new-J_theta_old)}, Parameters:{theta_initial}")
    
    return theta_initial, animation_data
    
theta_initial,animation_data = perform_linear_regression()

#Plot this data and linear regression output

plt.style.use('ggplot')


plt.figure(figsize=(10, 6))
plt.scatter(x_data_original, y_data, label='Data', color='blue', marker='x')
regression = x_data_original * theta_initial[0] + theta_initial[1]
plt.plot(x_data_original, regression, label='Linear Regression fit', color='red')

plt.xlabel("Acidity of wine (Normalised)")
plt.ylabel("Density of wine")
plt.title("Visualising relation between acidity and density of wine ") 
plt.legend()
plt.savefig('Plots/linear_visualised.png', dpi=300, bbox_inches='tight')
# plt.show()


theta_0 = np.linspace(-2, 2, 200)  
theta_1 = np.linspace(-2, 2, 200)  
def get_mesh():
    X, Y = np.meshgrid(theta_0, theta_1)
    Z = np.zeros((len(theta_0),len(theta_1)))

    m = len(x_data)
    for i,t0 in enumerate(theta_0):
        for j,t1 in enumerate(theta_1):
            theta_temp = [t0, t1]
            for x, y in zip(x_data, y_data):
                Z[j, i] += (y - np.sum(x * theta_temp)) ** 2
            Z[j, i] = Z[j, i] / (2 * m)

    return X, Y, Z

X, Y, Z = get_mesh()

animation_data_subset = animation_data[0::20]






fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')


ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='none',label='Loss function')
ax.contour3D(X, Y, Z, 10, cmap=cm.viridis)

ax.view_init(elev=20., azim=110)

frame = ax.scatter([], [], [], marker="o", c="black",alpha=1,s=17,label='Gradient descent')

def update1(i):
    ax.view_init(elev=20., azim=80 + i/5)
    x_frame = []
    y_frame = []
    z_frame = []
    for z,x,y in animation_data_subset[:i]:
        x_frame.append(x)
        y_frame.append(y)
        z_frame.append(z) 
    frame._offsets3d = (x_frame, y_frame, z_frame)
    return frame

plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
ax.set_zlabel('Loss')
plt.title('Loss function for linear regression')
# plt.legend(loc="upper left")

gif = FuncAnimation(fig, update1, frames=100, interval=200)
gif.save('Animation/3d_lost_plot.gif', dpi=90, writer='imagemagick')

# plt.legend()

plt.savefig("Plots/3d_plot_and_cost.png",dpi=500)

# plt.show()



for neta in [0.001, 0.025, 0.1,1.5]:
    theta_initial,animation_data_subset = perform_linear_regression(eta=neta,threshold=1e-10)
    
    # animation_data_subset = animation_data_subset[0::(len(animation_data_subset)//100)]
    
    fig,ax = plt.subplots(figsize=(20, 20))
    CS = ax.contour(X, Y, Z, extend='both',linewidths=3,levels=[0.01,0.1,0.2,0.5,1,2,3,4,5])
    ax.clabel(CS, inline=True, fontsize=10)

    frame = ax.scatter([], [], marker="o", c="black",alpha=1,s=17,label='Gradient descent')

    def update2(i):
        ax.clear()
        CS = ax.contour(X, Y, Z, extend='both',linewidths=3,levels=[0.01,0.1,0.2,0.5,1,2,3,4,5])
        ax.clabel(CS, inline=True, fontsize=10)



        x_frame = []
        y_frame = []
        z_frame = []
        for z,x,y in animation_data_subset[:i]:
            x_frame.append(x)
            y_frame.append(y)
            z_frame.append(z) 

        frame = ax.scatter(x_frame, y_frame, marker="o", c="black",alpha=1,s=17,label='Gradient descent')
        return frame

    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title(f'Loss function for linear regression with eta={neta}')

    gif = FuncAnimation(fig, update2, frames=100, interval=200, repeat_delay=3000)
    gif.save(f'Animation/contour_graph_{neta}.gif', dpi=90, writer='imagemagick')


    plt.savefig(f"Plots/plot_contour_{neta}.png",dpi=500)

    # plt.show()


