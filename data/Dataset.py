import os
import torch
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from PIL import Image
from torchvision import datasets
class MyData(Dataset) :
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.labels = []
        self.list_img_path = []
        self.root_dir = root_dir
        self.num_classes = [i for i in os.listdir(root_dir)]
        self.num_classes.remove(self.num_classes[0])
        if not os.path.exists(self.root_dir):
            raise FileExistsError(f"The directory {self.root_dir} does not exist. Set download=True to download it.")
        
        for classes in self.num_classes:
            sub_path_data = os.path.join(root_dir, classes)
            for image_id, image_name in enumerate(os.listdir(sub_path_data)):
                image_path = os.path.join(sub_path_data, image_name)
                self.list_img_path.append(image_path)
                self.labels.append(0 if classes == "NORMAL" else 1)

    def __len__(self):
        return len(self.list_img_path)
    
    def __getitem__(self, index):
        images = Image.open(self.list_img_path[index]).convert("RGB")
        labels = self.labels[index]
        if self.transform:
            images = transform(images)
        return images, labels
    

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = MyData("C:/Machine Learning/Real/Chest_X-Ray/chest_xray/train", transform=transform)
test_dataset = MyData("C:/Machine Learning/Real/Chest_X-Ray/chest_xray/test", transform=transform)
val_dataset = MyData("C:/Machine Learning/Real/Chest_X-Ray/chest_xray/val", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
