
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def get_training_data(count):
    x1 = np.random.normal(loc=3, scale=2, size=count)
    x2 = np.random.normal(loc=-1, scale=2, size=count)
    y = 3 + x1 + 2*x2 + np.random.normal(loc=0, scale=math.sqrt(2), size=count)

    return (x1, x2, y)


x, y, z = get_training_data(10000)

e = z - 3 - x - 2*y


fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=e,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

fig.update_layout(
    scene=dict(
        aspectratio=dict(x=1, y=1, z=0.5),
        aspectmode='manual',
        xaxis_title='X 1',
        yaxis_title='X 2',
        zaxis_title='Y',
    ),
    showlegend=True,
    title='Generated Data Visualised',
)


fig.write_html('stochastic_data_visualise.html')

# fig.show()

# print("Running stoc descent")

# batch_size = 100
# avg_over = 100

# J_theta_old = 0
# J_theta_new = 1e5
# threshold = 0.00001
# eta = 0.001

# theta_initial = np.zeros(3)

# animation_data = []

# global_iterator = 0

# total_data = 1000000

# x1_data, x2_data, y_data = get_training_data(total_data)


# while abs(J_theta_new-J_theta_old) > threshold:

#     J_theta_old = J_theta_new
#     J_theta_new = 0

#     for _ in range(avg_over):
#         J_theta = 0
#         m = batch_size
#         update_direction = np.zeros(3)
#         for i in range(global_iterator, global_iterator + batch_size):

#             x1 = x1_data[i % total_data]
#             x2 = x2_data[i % total_data]
#             y = y_data[i % total_data]

#             h_theta = theta_initial[0] + x1 * \
#                 theta_initial[1] + x2*theta_initial[2]
#             J_theta += (y - h_theta)**2
#             update_direction += (h_theta - y)*np.array([1, x1, x2])

#         J_theta = J_theta/(2*m)
#         global_iterator += batch_size
#         global_iterator %= total_data
#         animation_data.append(
#             (theta_initial[0], theta_initial[1], theta_initial[2], J_theta))

#         J_theta_new += J_theta
#         theta_initial = theta_initial - eta * (1/m) * update_direction

#     J_theta_new /= avg_over
    # print(f"J_theta: {J_theta_new}, Difference: {abs(J_theta_new-J_theta_old)}, Parameters:{theta_initial}")


test_df = pd.read_csv("./../Data/q2test.csv")
X1 = np.array(test_df['X_1'])
X2 = np.array(test_df['X_2'])
Y = np.array(test_df['Y'])

# theta_initial = [3, 1, 2]
# # X1

# err = Y - (theta_initial[0] + theta_initial[1]*X1 + theta_initial[2]*X2)
# err = np.sum(err**2)
# err = err/len(Y)



# theta_0_vals = [t[0] for t in animation_data]
# theta_1_vals = [t[1] for t in animation_data]
# theta_2_vals = [t[2] for t in animation_data]
# J_val = [t[3] for t in animation_data]

# hover_text = [f'Theta 0: {t0}<br>Theta 1: {t1}<br>Theta 2: {t2}<br>Cost: {J:.2f}'
#               for t0, t1, t2, J in zip(theta_0_vals, theta_1_vals, theta_2_vals, J_val)]

# fig = go.Figure(data=go.Scatter3d(
#     x=theta_0_vals, y=theta_1_vals, z=theta_2_vals,
#     marker=dict(
#         size=4,
#         color=J_val,
#         colorscale='Viridis',
#     ),
#     line=dict(
#         color='darkblue',
#         width=2
#     ),
#     hovertemplate=hover_text,
#     name="Stochastic Gradient Descent Path"
# ))

# fig.update_layout(
#     scene=dict(
#         aspectratio=dict(x=1, y=1, z=0.7),
#         aspectmode='manual',
#         xaxis_title='Theta 0',
#         yaxis_title='Theta 1',
#         zaxis_title='Theta 2',
#     ),
#     showlegend=True,
#     title='Stochastic Gradient Descent Path',
# )

# fig.write_html('stochastic_path.html')


batch_sizes = [1, 100, 10000, 1000000]
thresholds = [5e-4, 5e-4, 5e-5, 1e-7]
avg_overs = [1000, 1000, 100, 1]


total_data = 1000000
x1_data, x2_data, y_data = get_training_data(total_data)

x1_data = np.hstack((x1_data, x1_data))
x2_data = np.hstack((x2_data, x2_data))
y_data = np.hstack((y_data, y_data))

fig_datas = []

for avg_over, batch_size, threshold in zip(avg_overs, batch_sizes, thresholds):

    J_theta_old = 0
    J_theta_new = 1e5

    avg_over = avg_over
    eta = 0.001

    theta_initial = np.zeros(3)

    animation_data = []

    global_iterator = 0

    number_of_iterations = 0

    while abs(J_theta_new-J_theta_old) > threshold:
        # print(f"Currently {J_theta_new-J_theta_old} for batch size: {batch_size}, threshold: {threshold}, avg_over: {avg_over}")

        J_theta_old = J_theta_new
        J_theta_new = 0

        for _ in range(avg_over):
            J_theta = 0
            m = batch_size
            update_direction = np.zeros(3)

            x1_curr = x1_data[global_iterator:global_iterator+batch_size]
            x2_curr = x2_data[global_iterator:global_iterator+batch_size]
            y_curr = y_data[global_iterator:global_iterator+batch_size]

            x_curr = np.column_stack((np.ones(batch_size), x1_curr, x2_curr))
            h_theta = np.dot(x_curr, theta_initial)
            J_theta = np.sum((y_curr - h_theta)**2)
            update_direction += np.dot((np.dot(x_curr,
                                       theta_initial) - y_curr), x_curr)

            # for i in range(global_iterator, global_iterator + batch_size):

            #     x1 = x1_data[i%total_data]
            #     x2 = x2_data[i%total_data]
            #     y = y_data[i%total_data]

            #     h_theta = theta_initial[0] + x1*theta_initial[1] + x2*theta_initial[2]
            #     J_theta += (y - h_theta)**2
            #     update_direction += (h_theta - y)*np.array([1,x1,x2])

            J_theta = J_theta/(2*m)
            global_iterator += batch_size
            global_iterator %= total_data
            animation_data.append(
                (theta_initial[0], theta_initial[1], theta_initial[2], J_theta))

            J_theta_new += J_theta
            theta_initial = theta_initial - eta * (1/m) * update_direction

        J_theta_new /= avg_over
        number_of_iterations += 1
        # print(f"{batch_size}| J_theta: {J_theta_new}, Difference: {abs(J_theta_new-J_theta_old)}, Parameters:{theta_initial}")

    print(f"Theta learned: {theta_initial} for batch size: {batch_size}, threshold: {threshold}, avg_over: {avg_over} in {number_of_iterations} iterations")

    err = Y - (theta_initial[0] + theta_initial[1]*X1 + theta_initial[2]*X2)
    err = np.sum(err**2)
    err = err/len(Y)

    print(
        f"For test data error is {err} for batch size: {batch_size}, threshold: {threshold}, avg_over: {avg_over} in {number_of_iterations} iterations")

    theta_0_vals = [t[0] for t in animation_data]
    theta_1_vals = [t[1] for t in animation_data]
    theta_2_vals = [t[2] for t in animation_data]
    J_val = [t[3] for t in animation_data]

    hover_text = [f'Theta 0: {t0}<br>Theta 1: {t1}<br>Theta 2: {t2}<br>Cost: {J:.2f}'
                  for t0, t1, t2, J in zip(theta_0_vals, theta_1_vals, theta_2_vals, J_val)]

    scplt = go.Scatter3d(
        x=theta_0_vals, y=theta_1_vals, z=theta_2_vals,
        marker=dict(
            size=4,
            color=J_val,
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        ),
        hovertemplate=hover_text,
        name=f"Batch size:{batch_size}, Threshold:{threshold}"
    )
    fig_datas.append(scplt)

fig = go.Figure(data=fig_datas)

fig.update_layout(
    scene=dict(
        aspectratio=dict(x=1, y=1, z=0.7),
        aspectmode='manual',
        xaxis_title='Theta 0',
        yaxis_title='Theta 1',
        zaxis_title='Theta 2',
    ),
    showlegend=True,
    title='Stochastic Gradient Descent Path',
)

fig.write_html('stochastic_path_all.html')
