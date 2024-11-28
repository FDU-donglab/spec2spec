import os
import torch
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset



class TiffDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    def min_max_normalize(self, image_tensor):
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        return (image_tensor - min_val) / (max_val - min_val)
    
    def integral_at_w(self, image_tensor,dim = -1):
        return image_tensor.sum(dim=dim)
    
    def sampling(self, image_tensor):
        even_indices = image_tensor[:, ::2]
        odd_indices = image_tensor[:, 1::2]
        # Determine the length of the shorter tensor
        min_length = min(even_indices.size(1), odd_indices.size(1))
        # Truncate both tensors to the length of the shorter one
        even_indices = even_indices[:, :min_length]
        odd_indices = odd_indices[:, :min_length]
        return even_indices, odd_indices
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = io.imread(img_path).astype('float32')
        if self.transform:
            image = self.transform(image)
        image = self.integral_at_w(image)
        image = self.min_max_normalize(image)
        input,target = self.sampling(image)
        return input,target

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    def min_max_normalize(self, image_tensor):
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        return (image_tensor - min_val) / (max_val - min_val)
    
    def integral_at_w(self, image_tensor,dim = -1):
        return image_tensor.sum(dim=dim)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = io.imread(img_path).astype('float32')
        if self.transform:
            image = self.transform(image)
        image = self.integral_at_w(image)
        image = self.min_max_normalize(image)
        return image,img_name

def plot_image(image):
    """
    Plot a 1D image using matplotlib.
    :param image: 1D tensor or array
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 2))
    plt.plot(image.numpy())
    plt.title('1D Image')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.show()
    
if __name__ == "__main__":
    # Define the directory containing the tiff images
    image_dir = './data'
    
    # Define any transformations if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create the dataset
    dataset = TiffDataset(image_dir, transform=transform)
    
    # Access an image from the dataset
    image,target = dataset[4]
    print(f"Image shape: {image.shape}")
    print(f"Image shape: {image.min(),image.max()}")
    