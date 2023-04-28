import torch
from torchvision import transforms
import cv2
import numpy as np
from smplx import SMPL
from smplx import BodyModel

# Load SMPL model
smpl_model = BodyModel(bm_path="models/smpl",
                       num_betas=10,
                       batch_size=1,
                       create_transl=False)

# Load the trained CNN model
cnn = torch.load("models/cnn")

# Load image and preprocess it for the CNN model
img_path = "images/example.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = transforms.functional.to_tensor(img)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

# Pass the image through the CNN model to get the feature vector
with torch.no_grad():
    cnn_features = cnn(img_tensor)

# Generate 3D mesh using SMPL model and CNN features
output = smpl_model(betas=cnn_features,
                    body_pose=torch.zeros((1, 72)),
                    global_orient=torch.zeros((1, 3)))
vertices = output.vertices.detach().cpu().numpy().squeeze()

# Save the mesh as an OBJ file
output_path = "path/to/save/output.obj"
with open(output_path, 'w') as f:
    for v in vertices:
        f.write(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
