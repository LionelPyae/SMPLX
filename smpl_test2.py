import cv2
import torch
import numpy as np
from smplx import SMPL
from smplx.lbs import batch_rodrigues, lbs
from cnn import CNNModel

# Load SMPL model with 10 shape coefficients
smpl = SMPL('models/smpl', batch_size=1, create_transl=False,
            gender='neutral', use_face_contour=False, num_betas=10)

# Load 2D image and resize it to the desired shape
img = cv2.imread('images/example.png')
img = cv2.resize(img, (224, 224))

# Convert image to tensor and normalize it
img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
img_tensor = img_tensor / 255.0
img_tensor = img_tensor.unsqueeze(0)

# Pass image through a pre-trained CNN to get feature vector
# Load the state dictionary of the model
state_dict = torch.load("models/cnn")

# Create an instance of the CNN model
cnn = CNNModel()
# Load the state dictionary into the model
cnn.load_state_dict(state_dict)
# Set the model to evaluation mode
cnn.eval()
with torch.no_grad():
    cnn_features = cnn(img_tensor)

# Initialize model parameters
betas = torch.zeros(1, 10, dtype=torch.float32)
global_orient = torch.zeros(1, 3, dtype=torch.float32)
body_pose = torch.zeros(1, 72, dtype=torch.float32)

# Compute rotation matrices from body pose
rot_mats = batch_rodrigues(body_pose.view(-1, 3)).view(1, -1, 3, 3)

# Reshape the pose tensor
pose = body_pose.view(1, -1, 3)

# Compute vertices of the SMPLX model
output = smpl(betas=betas, body_pose=pose, global_orient=global_orient,
              pose2rot=False, skip_translation=True, rotate=False, return_verts=True)

# Print the shape of the vertices tensor
print(output.vertices.shape)
