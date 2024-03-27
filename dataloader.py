from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import torch
import os

class RealVsGeneratedDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        for label, kind in enumerate(['Real', 'Generated']):
            img_dir = os.path.join(root_dir, kind, phase, 'images')
            mask_dir = os.path.join(root_dir, kind, phase, 'masks')
            images = os.listdir(img_dir)
            for img_name in images:
                self.data.append((os.path.join(img_dir, img_name), os.path.join(mask_dir, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # Convert mask to RGB if it's not already

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Concatenate image and mask along the color channels
        combined = torch.cat((image, mask), 0)

        return img_path, combined, label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Adjusted for 3 channels
                            std=[0.229, 0.224, 0.225]),
        ])

    dataset = RealVsGeneratedDataset('dataset', 'val', transform=transform)
    print(len(dataset))
    print(dataset[0])

