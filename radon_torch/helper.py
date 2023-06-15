import os
import matplotlib.pyplot as plt

def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def plot_2_img_tensor(image1, image2, title1, title2):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image1, cmap='gray')
    axs[0].set_title(title1)
    axs[1].imshow(image2, cmap='gray')
    axs[1].set_title(title2)
    plt.show()