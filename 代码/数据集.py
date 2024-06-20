import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# Define transformations
trans_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
CIFAR10_dataset = datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=trans_cifar)

# Organize data by class
class_images = {i: [] for i in range(10)}  # 10 classes in CIFAR-10
for img, label in CIFAR10_dataset:
    if len(class_images[label]) < 10:  # Collect 10 images per class
        class_images[label].append(img)
    if all(len(imgs) == 10 for imgs in class_images.values()):
        break  # Stop once we collect enough images per class

# Stack images in order per class and create a grid
image_list = [img for sublist in class_images.values() for img in sublist]
image_tensor = torch.stack(image_list)
image_grid = utils.make_grid(image_tensor, nrow=10, padding=2, pad_value=1)  # White padding

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(fname="/home/ubuntu/gyl/FedPer1/cifar/pic/CIFAR10_dataset.svg", format="svg", dpi=300)
    plt.show()

# Show images
print("Displaying images from CIFAR10:")
imshow(image_grid)
