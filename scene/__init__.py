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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        '''
        Find the maximum iteration number.
        '''
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        '''
        Initialize the train and test cameras.
        '''
        self.train_cameras = {}
        self.test_cameras = {}
        '''
        Load the scene.
        This code snippet determines the type of scene data being processed based on the presence of specific files or directories in the provided `args.source_path`. 
        It uses conditional checks to identify whether the scene corresponds to a "Colmap" dataset or a "Blender" dataset 
        and then invokes the appropriate callback function from the `sceneLoadTypeCallbacks` dictionary to load the scene information.
        '''
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            '''
            If the `"sparse"` directory is found, the code assumes that the scene corresponds to a "Colmap" dataset. 
            It then calls the "Colmap" callback function from `sceneLoadTypeCallbacks`, 
            passing arguments like the source path, images, depths, evaluation mode, and train-test experiment flag.
            '''
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            '''
            If the `"sparse"` directory is not found, the code checks for the presence of a file named `"transforms_train.json"` in the same directory.
            If this file exists, it prints a message indicating that the dataset is assumed to be a "Blender" dataset.
            It then calls the "Blender" callback function from `sceneLoadTypeCallbacks`,
            passing arguments like the source path, white background flag, depths, and evaluation mode.
            '''
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        '''
        Initializes or prepares a scene by copying a 3D model file and generating a JSON file containing camera information. 
        It is executed only if self.loaded_iter is not set, indicating that the scene has not been loaded previously.
        '''
        if not self.loaded_iter:
            '''
            Opens the source 3D model file, located at scene_info.ply_path, in binary read mode ('rb'). 
            Simultaneously, it opens a destination file named "input.ply" in the directory specified by self.model_path in binary write mode ('wb'). 
            The contents of the source file are read and written directly to the destination file, effectively copying the 3D model file into the working directory.
            '''
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            '''
            Initializes an empty list, json_cams, to store camera information in JSON format. 
            It also initializes camlist, which will hold all the cameras associated with the scene. 
            '''
            json_cams = []
            camlist = []
            '''
            If scene_info.test_cameras exists, it appends these cameras to camlist. 
            Similarly, if scene_info.train_cameras exists, it appends those as well. 
            This ensures that both test and training cameras are included in the list.
            '''
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            '''
            Iterates over the combined list of cameras (camlist) using Python's enumerate, 
            which provides both the index (id) and the camera object (cam). 
            For each camera, it calls the camera_to_JSON function, which converts the camera's properties (e.g., position, rotation, and intrinsic parameters) 
            into a JSON-serializable dictionary. These dictionaries are appended to the json_cams list.
            '''
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            '''
            Writes the json_cams list to a file named "cameras.json" in the directory specified by self.model_path. 
            The file is opened in write mode ('w'), and the json.dump function is used to serialize the list into JSON format and save it to the file. 
            This process ensures that the camera information is stored in a structured and easily accessible format for later use.
            '''
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        '''
        Random shuffle of the training and testing camera lists (scene_info.train_cameras and scene_info.test_cameras) if the shuffle flag is set to True. 
        '''
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        '''
        Initializes the cameras_extent variable with the radius value from the scene_info.nerf_normalization dictionary.
        '''
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        '''
        Iterates over the resolution_scales list, which contains different scaling factors for rendering or processing the scene at various resolutions. 
        '''
        for resolution_scale in resolution_scales:
            '''
            Loads the training and test cameras for the specified resolution scale.
            '''
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            '''
            Loads the test cameras for the specified resolution scale.
            '''
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
        '''
        Determines how the Gaussian primitives for the scene are loaded or created. 
        The decision is based on whether a specific iteration (self.loaded_iter) has been previously loaded.
        '''
        if self.loaded_iter:
            '''
            If self.loaded_iter is set, indicating that the scene has been processed in a prior iteration, 
            and the corresponding Gaussian data should be loaded from a precomputed file.
            
            The code constructs the file path to the point cloud data using os.path.join, 
            combining the model path (self.model_path), the "point_cloud" directory, the iteration-specific subdirectory ("iteration_" + str(self.loaded_iter)), and the "point_cloud.ply" file. 
            This file contains the Gaussian primitives in a format compatible with the load_ply method of the self.gaussians object. 
            The load_ply method reads the .ply file, extracts relevant data (e.g., positions, features, scaling, rotation, and opacity), and initializes the Gaussian model accordingly. 
            Additionally, if args.train_test_exp is set to True, the method attempts to load exposure data from a related JSON file.
            '''
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            '''
            Otherwise, The `self.loaded_iter` is not set, it implies that the scene is being initialized for the first time. 
            In this case, the Gaussian primitives are created from scratch using the `create_from_pcd` method of the `self.gaussians` object. 
            This method takes the point cloud data (`scene_info.point_cloud`), the training camera information (`scene_info.train_cameras`), 
            and the spatial extent of the scene (`self.cameras_extent`) as inputs. 
            The `create_from_pcd` method processes the raw point cloud data, computes initial values for features like scaling, rotation, and opacity, 
            and initializes the Gaussian model with these values.
            '''
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
