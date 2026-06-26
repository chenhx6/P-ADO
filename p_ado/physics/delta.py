import numpy as np


# Convert the mixing angle in degrees to delta = tan(theta).
def theta_deg_to_delta(theta_deg):
    return np.tan(np.deg2rad(theta_deg))


# Convert delta back to the mixing angle in degrees.
def delta_to_theta_deg(delta):
    return np.rad2deg(np.arctan(delta))
