import torch
import torch.nn as nn
import torchvision.transforms.functional as F

# Class Definition
class ImageRotator(nn.Module):
    def __init__(self):
        super(ImageRotator, self).__init__()
        
    def forward(self, image_batch, angles):
        # Check input shapes
        assert len(image_batch.shape) == 4, "Invalid image_batch shape. Expected (n, c, y, x)."
        assert len(angles.shape) == 1, "Invalid angles shape. Expected (k,)."

        n, c, y, x = image_batch.shape
        k = angles.shape[0]
        
        # Initialize a tensor to store the rotated images
        rotated_imgs = torch.zeros((n, k, y, x), device=image_batch.device)

        for i in range(n):
            image = image_batch[i]  # Get current image
            for j in range(k):
                image_rotated = F.rotate(image, angles[j])  # Rotate the image
                rotated_imgs[i][j] = image_rotated
        self.rotated_batch = rotated_imgs

class ImgToSinogram(nn.Module):
    def __init__(self):
        super(ImgToSinogram, self).__init__()

    def forward(self, image_batch):
        # Check input shape
        assert len(image_batch.shape) == 4, "Invalid image_batch shape. Expected (n, k, y, x)."

        n, k, y, x = image_batch.shape

        # Initialize a tensor to store the sinograms
        sinograms = torch.zeros((n, k, x), device=image_batch.device)

        for i in range(n):
            sinogram = torch.zeros((k, x), device=image_batch.device)
            for j in range(k):
                image = image_batch[i, j]  # Get current image
                line_integrated = torch.sum(image, dim=1)  # Line integration along the y-axis
                sinogram[j, :] = line_integrated
            sinograms[i] = sinogram

        self.sinogram_batch = sinograms
###############################################################