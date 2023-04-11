import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler


class VideoDataset(Dataset):
    def __init__(self, data_path, train_ratio=0.8):
        self.data_path = data_path
        self.image_sequences = []
        self.labels = []
        self.emotion_map = {'Anger': 0, 'Fear': 1, 'Happiness': 2, 'Sadness': 3, 'Surprise': 4}

        for emotion in os.listdir(self.data_path):
            emotion_dir = os.path.join(self.data_path, emotion)
            image_files = os.listdir(emotion_dir)
            image_files = [os.path.join(emotion_dir, f) for f in image_files]
            sequence = []
            for i, image_file in enumerate(image_files):
                if i % 100 == 0 and i != 0:
                    self.image_sequences.append(sequence)
                    self.labels.append(emotion)
                    sequence = []
                sequence.append(image_file)
            if sequence:
                self.image_sequences.append(sequence)
                self.labels.append(emotion)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

        # Split into train and validation sets
        dataset_size = len(self.image_sequences)
        indices = list(range(dataset_size))
        split = int(train_ratio * dataset_size)
        train_indices, val_indices = indices[:split], indices[split:]
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def __getitem__(self, index):
        sequence = self.image_sequences[index]
        label_str = self.labels[index]
        images = []
        for image_path in sequence:
            image = Image.open(image_path).convert('L')
            image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        label = torch.tensor(self.emotion_map[label_str], dtype=torch.long)
        return images, label

    def __len__(self):
        return len(self.image_sequences)
