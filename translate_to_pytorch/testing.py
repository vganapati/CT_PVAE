import argparse
import time
import numpy as np
from helper import create_folder, get_images
from sinogram import radon


def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n','--num-train', type=int, help='number of points', default=15000)
    parser.add_argument('--toy', action='store_true', dest='use_toy_data', 
                        help='toy examples of 2 x 2 pixels')
    args = parser.parse_args()

    ### INPUT ###
    use_toy_data = args.use_toy_data
    
    if use_toy_data:
        # theta = np.array([0])
        # theta = np.array([np.pi/2])
        theta = np.array([0, np.pi/2])
        img_type = 'toy_discrete2' # 'mnist' or 'foam'
        pad = False
        discrete = True
    else:
        theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles
        img_type = 'foam' # 'mnist' or 'foam'
        pad = True
        discrete = False
        
    truncate_dataset = args.num_train
    
    #############
    
    save_path = 'dataset_' + img_type
    create_folder(save_path)
    
    # pull images that are normalized from 0 to 1
    # 0th dimension should be the batch dimension
    # 1st and 2nd dimensions should be spatial x and y coords, respectively
    x_train_imgs = get_images(img_type = img_type)
    
    x_train_imgs = x_train_imgs[0:truncate_dataset]
    
    if discrete:
        # theta == 0:
        proj_0 = np.sum(x_train_imgs, axis=1)
        # theta == np.pi/2
        proj_1 = np.sum(x_train_imgs, axis=2)[::-1]
        x_train_sinograms = np.stack((proj_0,proj_1), axis=1)
    else:
        x_train_sinograms = []
        for b in range(x_train_imgs.shape[0]):
            print(b)
            img = x_train_imgs[b]
            sinogram = radon(img)
            x_train_sinograms.append(np.expand_dims(sinogram, axis=0))
        # shape is truncate_dataset x num_angles x num_proj_pix
        x_train_sinograms = np.concatenate(x_train_sinograms, axis=0) 
    
    num_proj_pix = x_train_sinograms.shape[-1]
    
    x_train_sinograms[x_train_sinograms<0]=0
    
    np.save(save_path + '/x_train_sinograms.npy', x_train_sinograms)
    np.save(save_path + '/dataset_parameters.npy', np.array([theta,
                                                   num_proj_pix], dtype = object))

    np.save(save_path + '/x_size.npy', x_train_imgs.shape[1]) # size of original image
    np.save(save_path + '/y_size.npy', x_train_imgs.shape[2]) # size of original image
    
    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')