from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
import numpy as np

###################
## Chris's Notes ##
###################

# Functions should only return one thing
# Make global values static, You don't want them to be changed


# Time Step at 0.1s for testing out code
t_step = .01  # (s)
time_stop = 10  # (s)

# Initialize Plot
fig = plt.figure(figsize=(15, 10))
gs = gs.GridSpec(nrows=3, ncols=3)
ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])
ax4 = fig.add_subplot(gs[0, 2])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 2])


# Plot function - shows the position of the arm and the target on the left.
# Shows some key system parameters on the right
def plot(t, r_target, r0, r1, r2, theta_0, delta_theta_1, theta_1, r_error, r_cm, delta_r_cm, r_cm_test, theta_r,
         delta_l, error_integral):

    ax0.axis([-2.5, 2.5, -3, 6])
    ax0.scatter(r_target[0], r_target[1], color='red')
    #ax0.scatter(r_cm[0], r_cm[1])
    #ax0.scatter(r_cm_test[0], r_cm_test[1])
    ax0.plot([r0[0], r1[0]], [r0[1], r1[1]], color="black", linewidth=3)
    ax0.plot([r1[0], r2[0]], [r1[1], r2[1]], color="black", linewidth=3)
    ax0.set_title("Model")

    ax1.set_title("X Position", size=8)
    ax1.axis([0, 1, 1.25, 1.75])
    ax1.plot(t, r2[0], '-go')
    ax1.plot(t, r_target[0], '-ro')


    ax2.set_title("Y Position", size=8)
    ax2.axis([0, 1, 2.25, 2.75])
    ax2.plot(t, r_target[1], '-ro')
    ax2.plot(t, r2[1], '-go')

    ax3.set_title("Theta 1", size=8)
    ax3.plot(t, theta_1, '-bo')

    ax4.set_title("Theta R", size=8)
    ax4.plot(t, theta_r, '-go')

    ax5.set_title("Error Integral", size=8)
    ax5.plot(t, error_integral, '-yo')

    ax6.set_title("Error in X and Y", size=8)
    ax6.plot(t, r_error[0], '-bo')
    ax6.plot(t, r_error[1], '-ro')

    plt.pause(0.5)
    #ax0.clear()




# PRE-iterative functions - used before the time loop
# Set initial conditions, all given in the problem parameters
def initial_conditions():
    t = 0  # s
    joint_x = 0.9  # m
    joint_y = 1.5  # m
    theta_0 = np.pi / 6  # rad
    theta_1 = np.pi / 3  # rad
    m01 = 100  # kg
    d01 = 2.5  # m
    m12 = 50  # kg
    d12 = 1.5  # m
    theta_r = 0  # rad m^2 /s
    error_integral = 0

    return t, [joint_x, joint_y], theta_0, theta_1, m01, d01, m12, d12, theta_r, error_integral


# To get the robots initial conditions, we can use some simple geometry based on the image in the problem statement
def robot_position_initial(r1, theta_0, theta_1, d01, d12):
    # Position of end of first limb, wrt the joint
    x0 = r1[0] - d01 * np.cos(theta_0)
    y0 = r1[1] - d01 * np.sin(theta_0)
    # Position of end effector
    x2 = r1[0] - d12 * np.cos(theta_1 - theta_0)
    y2 = r1[1] + d12 * np.sin(theta_1 - theta_0)
    return [x0, y0], [x2, y2]


# Next we need the get the center of mass of the robot in its start position.
def robot_center_of_mass_initial(m01, m12, d01, d12, r1, theta_0, theta_1):
    rcm_x = (m01 * (r1[0] - 0.5 * d01 * np.cos(theta_0)) + m12 * (r1[0] - 0.5 * d12 * np.cos(theta_1 - theta_0))) / (
            m01 + m12)
    rcm_y = (m01 * (r1[1] - 0.5 * d01 * np.sin(theta_0)) + m12 * (r1[1] + 0.5 * d12 * np.sin(theta_1 - theta_0))) / (
            m01 + m12)
    rcm = [rcm_x, rcm_y]
    return rcm, [rcm_x, rcm_y]


# Target location - as outlined in the problem parameters
def target_location(t):
    x = 1 + .5 * np.cos((2 * np.pi * t) / 5)
    y = 2.5
    return [x, y]


# TIME LOOP FUNCTIONS - These are called everytime we step forward
# Control Loop
def pid_control(kp, kd, r1, r2, theta_1, r_target, error_integral, last_error_x, last_error_y):
    error_x = r_target[0] - r2[0]
    error_y = r_target[1] - r2[1]
    control_output = kp * (error_x - error_y) + kd * (error_x - last_error_x - (error_y - last_error_y)) * t_step
    error_integral += t_step * (abs(error_x) + abs(error_y)) / 2

    # error_theta = np.arctan2((r_target[1] - r1[1]), (r_target[0] - r1[0])) - theta_1
    # kp = 1
    # kd = .9
    return control_output, [error_x, error_y], error_integral


# This calculate the new location center of mass for a change in theta1,
# if the 01 link were to be fixed in place
def robot_center_of_mass_incremental(m01, m12, d12, theta_0, theta_1, delta_theta_1):
    delta_rcm_x = 0.5 * (m12 * d12) / (m01 + m12) * (
            -np.cos(theta_1 + delta_theta_1 - theta_0) + np.cos(theta_1 - theta_0))
    delta_rcm_y = 0.5 * (m12 * d12) / (m01 + m12) * (
            np.sin(theta_1 + delta_theta_1 - theta_0) - np.sin(theta_1 - theta_0))

    return [delta_rcm_x, delta_rcm_y]


# Calculate the new positions, give that the whole robot moves to offset the motion of the center of mass
# COM stays in the same place
# Also shows the angle of the last link changing
def robot_position_incremental_displacement(r0, r1, delta_r_cm, delta_theta_1, d12, theta_0, theta_1, r_cm):
    x1 = r1[0] - delta_r_cm[0]
    y1 = r1[1] - delta_r_cm[1]
    # Position of end of first limb, wrt the joint
    x0 = r0[0] - delta_r_cm[0]
    y0 = r0[1] - delta_r_cm[1]
    # Position of end effector
    x2 = r1[0] - d12 * np.cos(theta_1 - theta_0 + delta_theta_1) - delta_r_cm[0]
    y2 = r1[1] + d12 * np.sin(theta_1 - theta_0 + delta_theta_1) - delta_r_cm[1]

    theta_1 += delta_theta_1
    r_cm[0] += delta_r_cm[0]
    r_cm[1] += delta_r_cm[1]
    theta_0 = np.arctan((r1[1] - r0[1]) / (r1[0] - r0[0]))

    return [x0, y0], [x1, y1], [x2, y2], theta_1, r_cm, theta_0


def robot_center_of_mass_test(m01, m12, d01, d12, r1, theta_0, theta_1):
    rcm_x = (m01 * (r1[0] - 0.5 * d01 * np.cos(theta_0)) + m12 * (r1[0] - 0.5 * d12 * np.cos(theta_1 - theta_0))) / (
            m01 + m12)
    rcm_y = (m01 * (r1[1] - 0.5 * d01 * np.sin(theta_0)) + m12 * (r1[1] + 0.5 * d12 * np.sin(theta_1 - theta_0))) / (
            m01 + m12)
    rcm = [rcm_x, rcm_y]
    return rcm


def new_values(theta_1, delta_theta_1, r_cm, delta_r_cm, r0, r1):
    theta_1 += delta_theta_1
    r_cm[0] += delta_r_cm[0]
    r_cm[1] += delta_r_cm[1]
    theta_0 = np.arctan((r1[1] - r0[1]) / (r1[0] - r0[0]))
    return theta_1, r_cm, theta_0


def angular_momentum_incremental(theta_r, r1, m01, m12, d01, d12, theta_0, theta_1, r_cm, delta_r_cm):
    v1_x = m01 * (r1[0] - 0.5 * d01 * np.cos(theta_0) - delta_r_cm[0] - r_cm[0])
    v1_y = m01 * (r1[1] - 0.5 * d01 * np.sin(theta_0) - delta_r_cm[1] - r_cm[1])
    v1 = [v1_x, v1_y]
    v3 = [-delta_r_cm[0], -delta_r_cm[1]]

    delta_l01 = np.cross(v1, v3)

    v2_x = m12 * (r1[0] - 0.5 * d12 * np.cos(theta_1 - theta_0) - delta_r_cm[0] - r_cm[0])
    v2_y = m12 * (r1[1] + 0.5 * d12 * np.sin(theta_1 - theta_0) - delta_r_cm[1] - r_cm[1])
    v2 = [v2_x, v2_y]

    delta_l12 = np.cross(v2, v3)

    # print(v1_x, v2_x, v1_y, v2_y)
    # print(delta_l01, delta_l12)
    delta_l = delta_l01 + delta_l12

    r01_x = r1[0] - 0.5 * d01 * np.cos(theta_0) - delta_r_cm[0]
    r01_y = r1[1] - 0.5 * d01 * np.sin(theta_0) - delta_r_cm[1]
    r01 = np.sqrt(r01_x ** 2 + r01_y ** 2)

    r12_x = r1[0] - 0.5 * d12 * np.cos(theta_1 - theta_0) - delta_r_cm[0]
    r12_y = r1[1] + 0.5 * d12 * np.sin(theta_1 - theta_0) - delta_r_cm[1]
    r12 = np.sqrt(r12_x ** 2 * r12_y ** 2)

    moment_of_inertia = m01 * r01 ** 2 + m12 * r12 ** 2

    delta_theta_r = -delta_l / moment_of_inertia
    theta_r = theta_r + delta_theta_r

    return theta_r, delta_l


# Position of the robot
def robot_position_incremental_rotation(r0, r1, r2, theta_r, r_cm, delta_r_cm):
    x0 = r_cm[0] + ((r0[0] - r_cm[0]) * np.cos(theta_r) - (r0[1] - r_cm[1]) * np.sin(theta_r))
    y0 = r_cm[1] + ((r0[0] - r_cm[0]) * np.sin(theta_r) + (r0[1] - r_cm[1]) * np.cos(theta_r))
    x1 = r_cm[0] + ((r1[0] - r_cm[0]) * np.cos(theta_r) - (r1[1] - r_cm[1]) * np.sin(theta_r))
    y1 = r_cm[1] + ((r1[0] - r_cm[0]) * np.sin(theta_r) + (r1[1] - r_cm[1]) * np.cos(theta_r))
    x2 = r_cm[0] + ((r2[0] - r_cm[0]) * np.cos(theta_r) - (r2[1] - r_cm[1]) * np.sin(theta_r))
    y2 = r_cm[1] + ((r2[0] - r_cm[0]) * np.sin(theta_r) + (r2[1] - r_cm[1]) * np.cos(theta_r))

    #	x0 = (r_cm[0] + r0[0] * np.cos(theta_r) - r0[1] * np.sin(theta_r) - delta_r_cm[0] * np.cos(theta_r) - delta_r_cm[
    #		1] * np.sin(theta_r) - r_cm[0] * np.cos(theta_r) - r_cm[1] * np.sin(theta_r))
    #	y0 = (r_cm[1] + r0[0] * np.sin(theta_r) + r0[1] * np.cos(theta_r) - delta_r_cm[0] * np.sin(theta_r) + delta_r_cm[
    #		1] * np.cos(theta_r) - r_cm[0] * np.sin(theta_r) + r_cm[1] * np.cos(theta_r))
    #	x1 = (r_cm[0] + r1[0] * np.cos(theta_r) - r1[1] * np.sin(theta_r) - delta_r_cm[0] * np.cos(theta_r) - delta_r_cm[
    #		1] * np.sin(theta_r) - r_cm[0] * np.cos(theta_r) - r_cm[1] * np.sin(theta_r))
    #	y1 = (r_cm[1] + r1[0] * np.sin(theta_r) + r1[1] * np.cos(theta_r) - delta_r_cm[0] * np.sin(theta_r) + delta_r_cm[
    #		1] * np.cos(theta_r) - r_cm[0] * np.sin(theta_r) + r_cm[1] * np.cos(theta_r))
    #	x2 = (r_cm[0] + r2[0] * np.cos(theta_r) - r2[1] * np.sin(theta_r) - delta_r_cm[0] * np.cos(theta_r) - delta_r_cm[
    #		1] * np.sin(theta_r) - r_cm[0] * np.cos(theta_r) - r_cm[1] * np.sin(theta_r))
    #	y2 = (r_cm[1] + r2[0] * np.sin(theta_r) + r2[1] * np.cos(theta_r) - delta_r_cm[0] * np.sin(theta_r) + delta_r_cm[
    #		1] * np.cos(theta_r) - r_cm[0] * np.sin(theta_r) + r_cm[1] * np.cos(theta_r))

    return [x0, y0], [x1, y1], [x2, y2]


# This is where the good stuff happens - time loop stepping through the different system equations
def time_loop(kp, kd, plot_bool):
    t, r1, theta_0, theta_1, m01, d01, m12, d12, theta_r, error_integral = initial_conditions()
    r0, r2 = robot_position_initial(r1, theta_0, theta_1, d01, d12)
    r_cm, r_cm_init = robot_center_of_mass_initial(m01, m12, d01, d12, r1, theta_0, theta_1)
    r_error = [0, 0]
    success = False

    while t < time_stop:
        #  Start by calculating the target position for the time
        r_target = target_location(t)

        #  Now we calculate the change in theta1 based on the controller
        delta_theta_1, r_error, error_integral = pid_control(kp, kd,
                                                             r1, r2, theta_1, r_target, error_integral, r_error[0],
                                                             r_error[1])

        #  With a change in angle calculated, we can now calculate what our new center of mass would be
        # IF the first link didn't move at all
        delta_r_cm = robot_center_of_mass_incremental(
            m01, m12, d12, theta_0, theta_1, delta_theta_1)

        # Now that we know what that center of mass displacement would be if the robot were fixed
        # we know how much the robot must be displaced by such that the center of mass doesn't move
        # (which it does not in micro gravity)
        # Calculate the new position of each of the points on the robot, r2 does swing because theta1 changes
        r0, r1, r2, theta_1, r_cm, theta_0 = robot_position_incremental_displacement(
            r0, r1, delta_r_cm, delta_theta_1, d12, theta_0, theta_1, r_cm)

        # quick test to confirm that the center of mass of the robot does not in fact move
        rcm_test = robot_center_of_mass_test(m01, m12, d01, d12, r1, theta_0, theta_1)

        # With the robots translation established, we now need to make sure angular momentum is conserved.
        # Calculate theta_r of the whole robot such that it cancels out the angular momentum
        theta_r, delta_l = angular_momentum_incremental(
            theta_r, r1, m01, m12, d01, d12, theta_0, theta_1, r_cm, delta_r_cm)

        r0, r1, r2 = robot_position_incremental_rotation(r0, r1, r2, theta_r, r_cm, delta_r_cm)

        if plot_bool:
            plot(t, r_target, r0, r1, r2, theta_0, delta_theta_1, theta_1, r_error,
                 r_cm, delta_r_cm, rcm_test, theta_r, delta_l, error_integral)
            label = str(t) + ".png"
            fig.savefig(label)
            ax0.clear()
        if 0.025 > r_target[0] - r2[0] > -0.025 and 0.025 > r_target[1] - r2[1] > -0.025:
            success = True
            if plot_bool:
                #print(r_error)
                print("Time: ", t, "\n")
                print("X Effector: ", r2[0], " X Target: ", r_target[0], "\n")
                print("Y Effector: ", r2[1], " Y Target: ", r_target[1], "\n")
                plt.savefig("plots.png")
            return error_integral, success
        if plot_bool and success:
            print(t,r2,r_target)

        t += t_step
    return error_integral, success


def get_pd_parameters():
    kp_list = np.linspace(0, 2, 10)
    kd_list = np.linspace(0, 10, 10)
    output = []
    best_match = []
    best = 10
    plot_bool = False
    for kp in kp_list:
        for kd in kd_list:
            error_integral, success = time_loop(kp, kd, plot_bool)
            output.append([kp, kd, error_integral, success])
            print(kp, kd)
            if abs(error_integral) < abs(best) and success:
                best_match = [kp, kd, error_integral]
                best = error_integral

    print("\n The optimal k values are kp: ", best_match[0], "and kd: ", best_match[1], "\n")
    return best_match[0], best_match[1]


def main():
    #kp, kd = get_pd_parameters()
    kp = 0.9
    kd = 6.0
    time_loop(kp, kd, True)


if __name__ == "__main__":
    main()
