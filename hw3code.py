import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify, pi, diff, lambdify, atan2, sqrt
import matplotlib.pyplot as plt

# Define symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1 theta2 theta3 theta4 theta5 theta6')
q = Matrix([theta1, theta2, theta3, theta4, theta5, theta6])

# Define DH parameters for UR3 robot
dh_params = {
    'd1': 151.9, 'd2': 0, 'd3': 0, 'd4': 112.35, 'd5': 85.35, 'd6': 81.9,
    'a1': 0, 'a2': 243.65, 'a3': 213.25, 'a4': 0, 'a5': 0, 'a6': 0,
    'alpha1': pi/2, 'alpha2': 0, 'alpha3': 0, 'alpha4': pi/2, 'alpha5': -pi/2, 'alpha6': 0
}

def dh_transform(theta, d, a, alpha):
    """Create the D-H transformation matrix for a single link."""
    return Matrix([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Create transformation matrices
T01 = dh_transform(theta1, dh_params['d1'], dh_params['a1'], dh_params['alpha1'])
T12 = dh_transform(theta2, dh_params['d2'], dh_params['a2'], dh_params['alpha2'])
T23 = dh_transform(theta3, dh_params['d3'], dh_params['a3'], dh_params['alpha3'])
T34 = dh_transform(theta4, dh_params['d4'], dh_params['a4'], dh_params['alpha4'])
T45 = dh_transform(theta5, dh_params['d5'], dh_params['a5'], dh_params['alpha5'])
T56 = dh_transform(theta6, dh_params['d6'], dh_params['a6'], dh_params['alpha6'])

# Cumulative transformation matrices
T02 = simplify(T01 * T12)
T03 = simplify(T02 * T23)
T04 = simplify(T03 * T34)
T05 = simplify(T04 * T45)
T06_sym = simplify(T05 * T56)

# Extract Jacobian
origins = [Matrix([0, 0, 0])]
z_axes = [Matrix([0, 0, 1])]
cumulative_transforms = [T01, T02, T03, T04, T05, T06_sym]

for T in cumulative_transforms:
    origins.append(T[:3, 3])
    z_axes.append(T[:3, 2])

position = T06_sym[:3, 3]
J = Matrix.zeros(6, 6)
for i in range(6):
    J[:3, i] = z_axes[i].cross(position - origins[i])
    J[3:, i] = z_axes[i]

J_simplified = simplify(J)

# Create a lambda function for evaluation of the Jacobian
J_func = lambdify((theta1, theta2, theta3, theta4, theta5, theta6), J_simplified, 'numpy')

# Define trajectory
def trajectory(t):
    """Piecewise trajectory for the end-effector."""
    total_time = 200  # Total time for the trajectory
    t_normalized = t / total_time

    if t_normalized <= 0.25:  # First quarter: half circle from (0,0) to (-10,0)
        theta = pi * (1 - t_normalized * 4)
        x = 5 * cos(theta) - 5
        y = 5 * sin(theta)
    elif t_normalized <= 0.5:  # Second quarter: straight line from (-10,0) to (-10,-5)
        x = -10
        y = -20 * (t_normalized - 0.25)
    elif t_normalized <= 0.75:  # Third quarter: straight line from (-10,-5) to (0,-5)
        x = 40 * (t_normalized - 0.5) - 10
        y = -5
    else:  # Fourth quarter: straight line from (0,-5) to (0,0)
        x = 0
        y = 20 * (t_normalized - 0.75) - 5

    z = 0.5  # Constant height
    return np.array([float(x), float(y), float(z)])

# Robot link masses and centers of mass
masses = [2.0, 3.42, 1.26, 0.8, 0.8, 0.35]
centers_of_mass = [
    [0, 0, 0.05], [0, 0.1, 0], [0.1, 0, 0],
    [0, -0.05, 0], [-0.05, 0, 0], [0, 0, -0.05]
]

# Compute gravitational torques
def compute_gravity_torques(J, masses, centers_of_mass, g):
    gravity = np.array([0, 0, -g])
    torque_total = np.zeros(6)
    for i, (m, com) in enumerate(zip(masses, centers_of_mass)):
        com_vector = np.array(com)
        J_linear = J[:3, :]
        torque_total += J_linear.T @ (m * gravity)
    return torque_total

# Simulate over time
times = np.linspace(0, 200, 1000)
external_force = np.array([0, 0, -5, 0, 0, 0])  # 5 N force downward, no moments
g = 9.81  # Gravitational acceleration

torques = []

# Simple inverse kinematics
def simple_inverse_kinematics(pos):
    x, y, z = pos
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    s = z - dh_params['d1']
    D = (r**2 + s**2 - dh_params['a2']**2 - dh_params['a3']**2) / (2 * dh_params['a2'] * dh_params['a3'])
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
    theta2 = np.arctan2(s, r) - np.arctan2(dh_params['a3'] * np.sin(theta3), dh_params['a2'] + dh_params['a3'] * np.cos(theta3))
    theta4 = 0  # Simplified
    theta5 = -np.pi/2 - theta2 - theta3  # To keep end-effector horizontal
    theta6 = 0  # Simplified
    return [theta1, theta2, theta3, theta4, theta5, theta6]

for t in times:
    pos_ee = trajectory(t)
    joint_angles = simple_inverse_kinematics(pos_ee)
    J_numeric = J_func(*joint_angles)
    torque_gravity = compute_gravity_torques(J_numeric, masses, centers_of_mass, g)
    torque_force = J_numeric.T @ external_force
    torque_total = torque_gravity + torque_force
    torques.append(torque_total)

torques = np.array(torques)

# Plot joint torques
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.plot(times, torques[:, i])
    plt.title(f'Joint {i+1} Torque vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot trajectory
trajectory_points = np.array([trajectory(t) for t in times])
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2])
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('End-Effector Trajectory')
ax.set_xlim(-12, 2)
ax.set_ylim(-6, 6)
ax.set_zlim(0, 1)
plt.show()

# Plot 2D projections of the trajectory
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
plt.title('XY Projection')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.grid(True)
plt.axis('equal')
plt.xlim(-12, 2)
plt.ylim(-6, 6)

# Add markers for key points
key_points = [(0, 0), (-10, 0), (-10, -5), (0, -5)]
for point in key_points:
    plt.plot(point[0], point[1], 'ro')

plt.subplot(132)
plt.plot(trajectory_points[:, 0], trajectory_points[:, 2])
plt.title('XZ Projection')
plt.xlabel('X (cm)')
plt.ylabel('Z (cm)')
plt.grid(True)
plt.xlim(-12, 2)
plt.ylim(0, 1)

plt.subplot(133)
plt.plot(trajectory_points[:, 1], trajectory_points[:, 2])
plt.title('YZ Projection')
plt.xlabel('Y (cm)')
plt.ylabel('Z (cm)')
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
