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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

'''
The searchForMaxIteration function is designed to find the maximum iteration number from a list of saved files in a given folder. 
This is useful in scenarios where files are named with iteration numbers (e.g., checkpoints or logs), and you want to resume from the latest iteration.
'''
def searchForMaxIteration(folder):
    '''
    The os.listdir() function in Python is used to retrieve a list of all entries (files and directories) in a specified directory. 
    It is part of the os module, which provides functions for interacting with the operating system.
    '''
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
