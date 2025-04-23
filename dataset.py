import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ThymeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir


        self.stage_mapping = {
            'stage1_returning': 0, 
            'stage2_initial_flowering': 1,  
            'stage3_full_flowering': 2, 
            'stage4_fruiting': 3 
        }

        self.classes = list(self.stage_mapping.keys())

 
        self.images = []
        self.labels = []


        print("\n类别映射关系：")
        for stage_name, idx in self.stage_mapping.items():
            print(f"{stage_name}: {idx}")


        for stage_name, label in self.stage_mapping.items():
            stage_dir = os.path.join(root_dir, stage_name)
            if os.path.isdir(stage_dir):
                for img_name in sorted(os.listdir(stage_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(stage_dir, img_name)
                        if os.path.exists(img_path):
                            self.images.append(img_path)
                            self.labels.append(label)


        print("\n各类别样本数量：")
        label_counts = {}
        for label in self.labels:
            stage_name = self.classes[label]
            label_counts[stage_name] = label_counts.get(stage_name, 0) + 1

        for stage_name, count in label_counts.items():
            print(f"{stage_name}: {count} 个样本")


        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:

            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            if not torch.is_tensor(image):
                raise ValueError(f"图像转换失败: {img_path}")
            if image.shape != (3, 224, 224):
                raise ValueError(f"图像维度错误 {image.shape}: {img_path}")

            return image, label

        except Exception as e:
            print(f"处理图像出错 {img_path}: {str(e)}")
 
            return self.__getitem__((idx + 1) % len(self))

    def get_class_names(self):
        return self.classes

    def get_class_counts(self):

        counts = {}
        for label in self.labels:
            stage_name = self.classes[label]
            counts[stage_name] = counts.get(stage_name, 0) + 1
        return counts
