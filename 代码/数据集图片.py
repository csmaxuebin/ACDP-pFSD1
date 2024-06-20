# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define a transform to normalize the data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
# ])
#
# # Load the datasets
# trainsets = {
#     # 'FMNIST': torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform),
#     # 'EMNIST': torchvision.datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform),
#     # 'SVHN': torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform),
#     'CIFAR10': torchvision.datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=transform),
#     # 'CIFAR100': torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# }
#
# # Function to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # Get some random training images
# # dataiter = {name: iter(torch.utils.data.DataLoader(trainsets[name], batch_size=4, shuffle=True, num_workers=2)) for name in trainsets}
# # images = {name: dataiter[name].next()[0] for name in dataiter}
# # Get some random training images
# dataiter = {name: iter(torch.utils.data.DataLoader(trainsets[name], batch_size=4, shuffle=True, num_workers=2)) for name in trainsets}
# images = {name: next(dataiter[name])[0] for name in dataiter}
#
#
# # Show images
# for name, image_batch in images.items():
#     print(f"Displaying images from: {name}")
#     imshow(torchvision.utils.make_grid(image_batch))
from email import utils

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# import transforms as transforms

# Define a transform to normalize the data and resize images
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize images to 128x128
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
# ])

# Load the dataset
trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                                                     (0.19803012, 0.20101562,
                                                                             0.19703614))])
SVHN_dataset = torchvision.datasets.SVHN(root='../data/SVHN', split="train", download=True, transform=trans_svhn)
data_loader = torch.utils.data.DataLoader(SVHN_dataset, batch_size=100, shuffle=True, num_workers=2)
# trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# CIFAR10_dataset = datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=trans_cifar)
# trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# EMNIST_dataset = torchvision.datasets.EMNIST('../data/emnist/', train=True, download=True, transform=trans_emnist,split='balanced')
# data_loader = torch.utils.data.DataLoader(EMNIST_dataset, batch_size=100, shuffle=True, num_workers=2)
# trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# FMNIST_dataset = torchvision.datasets.FashionMNIST('../data/FashionMNIST/', train=True, download=True,
#                                               transform=trans_fmnist)
# DataLoaders

#train_data = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)





# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))  # Increase the figure size
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='lanczos')  # Use Lanczos interpolation
    plt.axis('off')  # Turn off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove margin
    plt.savefig(fname="/home/ubuntu/gyl/FedPer1/cifar/pic/SVHN_dataset.svg", format="svg",dpi=300)  # Save in SVG format for high-quality vector graphics
    plt.show()

# Get some random training images
data_iter = iter(data_loader)
images, labels = next(data_iter)
image_grid = torchvision.utils.make_grid(images, nrow=10, padding=2,pad_value=1)  # White padding
imshow(image_grid)
# Show images
print(f"Displaying images from CIFAR10:")
# imshow(torchvision.utils.make_grid(images, padding=2))
#imshow(torchvision.utils.make_grid(images, nrow=5, padding=2))



# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define a transform to normalize the data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
# ])
#
# # Load the datasets
# trainsets = {
#     'FMNIST': torchvision.datasets.FashionMNIST(root='../data/FashionMNIST/', train=True, download=True, transform=transform),
#     'EMNIST': torchvision.datasets.EMNIST(root='../data/emnist', split='balanced', train=True, download=True, transform=transform),
#     'SVHN': torchvision.datasets.SVHN(root='../data/SVHN', split='train', download=True, transform=transform),
#     'CIFAR10': torchvision.datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=transform),
#     'CIFAR100': torchvision.datasets.CIFAR100(root='../data/cifar', train=True, download=True, transform=transform)
# }
#
# # Function to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# # Get some random training images
# dataiter = {name: iter(torch.utils.data.DataLoader(trainsets[name], batch_size=4, shuffle=True, num_workers=2)) for name in trainsets}
# images = {name: next(dataiter[name])[0] for name in dataiter}
#
# # Show images
# for name, image_batch in images.items():
#     print(f"Displaying images from: {name}")
#     imshow(torchvision.utils.make_grid(image_batch))
