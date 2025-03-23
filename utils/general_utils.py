#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

'''
build_rotation function constructs a 3D rotation matrix from a quaternion. 
Quaternions are a compact and efficient way to represent rotations in 3D space, avoiding issues like gimbal lock that can occur with Euler angles.
'''
def build_rotation(r):
    '''
    r: (N,4) quaternion. 
    Normalize the quaternion. 
    The quaternion is normalized to ensure it represents a valid rotation. A valid quaternion has a unit norm
    '''
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    '''
    Initialize a 3x3 rotation matrix R with zeros. 
    A zero tensor of shape (N, 3, 3) is created to store the rotation matrices for all N quaternions.
    '''
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    '''
    Extract the quaternion components.
    The scalar (r) and vector (x, y, z) components of the quaternion are extracted for further computation.
    '''
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    '''
    Compute the rotation matrix.
    The elements of the 3×3 rotation matrix are computed using the quaternion-to-matrix conversion formula. 
    This formula ensures that the resulting matrix is orthogonal and represents a valid rotation.
    Ref: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    '''
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
'''
The build_scaling_rotation function constructs a transformation matrix that combines scaling and rotation operations in 3D space. 
This matrix is used to transform 3D points or objects by applying scaling along the principal axes (x, y, z) 
and then rotating them using a quaternion-based rotation.
Inputs: 
s: A tensor of shape (N, 3) representing scaling factors for each of the three axes (x, y, z) for N transformations.
Example: [s = [[2.0, 3.0, 4.0]]](http://vscodecontentref/2) scales the x-axis by 2, the y-axis by 3, and the z-axis by 4.
r: A tensor of shape (N, 4) representing quaternions that define the rotation for each of the N transformations.
Quaternions are a compact way to represent 3D rotations without the risk of gimbal lock.
Output:
L: A tensor of shape (N, 3, 3) representing the transformation matrix that combines scaling and rotation operations.
'''
def build_scaling_rotation(s, r):
    '''
    Initialize a 3x3 scaling matrix L with zeros.
    '''
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    '''
    Compute the rotation matrix R using the build_rotation function.
    '''
    R = build_rotation(r)
    '''
    Set the diagonal elements of the scaling matrix L.
    The diagonal elements of the scaling matrix are set to the scaling factors s along the x, y, and z axes.
    '''
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    '''
    Compute the final transformation matrix.
    The transformation matrix is computed by multiplying the scaling matrix L with the rotation matrix R.
    '''
    L = R @ L
    
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
