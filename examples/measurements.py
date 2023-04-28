import numpy as np

def get_body_measurements(vertices):
    # Get x, y, and z coordinates of each vertex
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Calculate body measurements
    chest_circumference = np.max(x) - np.min(x)
    waist_circumference = np.max(y) - np.min(y)
    hip_circumference = np.max(z) - np.min(z)
    inseam_length = np.min(y) - np.min(z)
    outseam_length = np.max(y) - np.max(z)
    thigh_circumference = np.max(x[:np.argmin(y)]) - np.min(x[:np.argmin(y)])
    calf_circumference = np.max(x[np.argmin(y):]) - np.min(x[np.argmin(y):])

    return chest_circumference, waist_circumference, hip_circumference, inseam_length, outseam_length, thigh_circumference, calf_circumference
   
