import cv2
import torch
import numpy as np
import openpose
import smplx

# Initialize OpenPose
params = openpose.get_params()
openpose.init_openpose(params)

# Initialize SMPL-X model
model_folder = 'models'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = smplx.create(model_folder, model_type='smplx', gender='neutral',
                     num_betas=10, use_face_contour=True).to(device)

# Load input image
img = cv2.imread('images/test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect keypoints using OpenPose
keypoints = openpose.detect_keypoints(img)

# Convert keypoints to numpy array
keypoints = np.array(keypoints)

# Rescale keypoints to match input image size
scale = np.array([img.shape[1], img.shape[0]])
keypoints = keypoints * scale

# Fit SMPL-X to detected keypoints
betas, pose, trans = smplx.fit_smplx_from_2d(keypoints, model, device)

# Print betas and pose
print('Betas:', betas)
print('Pose:', pose)
