import numpy as np
import matplotlib.pyplot as plt

# Functions for target movement patterns
def linear_movement(t, initial_pos, velocity):
    return initial_pos + velocity * t

def circular_movement(t, radius, angular_velocity, center):
    return center + radius * np.array([np.cos(angular_velocity * t), np.sin(angular_velocity * t)])

def sinusoidal_movement(t, amplitude, frequency, initial_pos):
    return initial_pos + amplitude * np.array([t, np.sin(2 * np.pi * frequency * t)])

# LOS pure pursuit
def los_pure_pursuit(t, target_pos_func, target_params, initial_missile_pos, missile_speed):
    missile_pos = initial_missile_pos
    missile_trajectory = [missile_pos]
    dt = 0.01

    for time in np.arange(0, t, dt):
        target_pos = target_pos_func(time, *target_params)
        los_vector = target_pos - missile_pos
        los_direction = los_vector / np.linalg.norm(los_vector)
        missile_pos = missile_pos + missile_speed * los_direction * dt
        missile_trajectory.append(missile_pos)

    return np.array(missile_trajectory)

# Parameters
simulation_time = 10
missile_speed = 1
initial_missile_pos = np.array([3, 3])

# Linear movement
linear_initial_pos = np.array([5, 5])
linear_velocity = np.array([0.2, 0.5])
linear_trajectory = los_pure_pursuit(simulation_time, linear_movement, [linear_initial_pos, linear_velocity], initial_missile_pos, missile_speed)

# Circular movement
circle_center = np.array([10, 0])
circle_radius = 5
angular_velocity = 0.1
circular_trajectory = los_pure_pursuit(simulation_time, circular_movement, [circle_radius, angular_velocity, circle_center], initial_missile_pos, missile_speed)

# Sinusoidal movement
sinusoidal_initial_pos = np.array([0, 0])
sinusoidal_amplitude = 2
sinusoidal_frequency = 0.5
sinusoidal_trajectory = los_pure_pursuit(simulation_time, sinusoidal_movement, [sinusoidal_amplitude, sinusoidal_frequency, sinusoidal_initial_pos], initial_missile_pos, missile_speed)

# Plotting
plt.figure(figsize=(10, 10))

plt.plot(linear_trajectory[:, 0], linear_trajectory[:, 1], label='Linear Movement')
plt.plot(circular_trajectory[:, 0], circular_trajectory[:, 1], label='Circular Movement')
plt.plot(sinusoidal_trajectory[:, 0], sinusoidal_trajectory[:, 1], label='Sinusoidal Movement')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LOS Seeker System with Different Target Movement Patterns")
plt.grid(True)
plt.show()

