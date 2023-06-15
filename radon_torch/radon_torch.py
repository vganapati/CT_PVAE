### Author: Hojune Kim
### Date: June 15, 2023
### Last Updated: Jun 15, 2023
### Description: Radon Transform (obj to sinogram)

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import os
import time
import numpy as np

from helper import create_folder, plot_2_img_tensor
from sinogram_class import ImageRotator, ImgToSinogram


def main():
    ###############################################################
    # Specify the file paths
    sinogram_path = os.path.join(os.pardir, 'dataset_foam', 'x_train_sinograms.npy')
    obj_path = os.path.join(os.pardir, 'foam_training.npy')

    # Load the .npy files
    sinogram_data = np.load(sinogram_path)
    obj_data = np.load(obj_path)

    # Convert the data to PyTorch tensors
    sinogram_tensor = torch.from_numpy(sinogram_data)
    obj_tensor = torch.from_numpy(obj_data)

    # Preprocess
    image_batch = obj_tensor.unsqueeze(1) # Size = (n,y,x)
    angles = np.arange(1, 181, dtype=float) # Angle list
    
    # Create an instances for the class
    rotata = ImageRotator()
    sinooo = ImgToSinogram()

    # Pass the image batch and angles through the model
    rotata.forward(image_batch, angles) 
    sinooo.forward(rotata.rotated_batch) 

    # Define Variables to save results
    rotated_images_batch = rotata.rotated_batch # torch.Size([50, 180, 128, 128])
    sinogram_batch = sinooo.sinogram_batch # torch.Size([50, 180, 128])
    ###############################################################
    
    # Trying to save images
    img_type = "synthetic_foam"
    save_path = img_type + "_output"
    create_folder(save_path)
    create_folder(save_path +"/")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')

