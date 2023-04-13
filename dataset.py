import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset, Subset
import numpy as np

import random

class VideoDataset(Dataset):
    def __init__(self, data_path, train_ratio=0.8):
        self.data_path = data_path
        self.image_sequences = []
        self.labels = []
        self.class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        self.num_classes = len(self.class_names)

        for emotion_idx, emotion in enumerate(self.class_names):
            emotion_dir = os.path.join(self.data_path, emotion)
            image_files = os.listdir(emotion_dir)
            image_files = [os.path.join(emotion_dir, f) for f in image_files]
            sequence = []
            for i, image_file in enumerate(image_files):
                if i % 3 == 0 and i != 0:
                    self.image_sequences.append(sequence)
                    label = emotion_idx
                    self.labels.append(label)
                    sequence = []
                sequence.append(image_file)
            if sequence:
                self.image_sequences.append(sequence)
                label = emotion_idx
                self.labels.append(label)

        # Shuffle image sequences and labels
        zipped = list(zip(self.image_sequences, self.labels))
        random.shuffle(zipped)
        self.image_sequences, self.labels = zip(*zipped)

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

        self.train_subset = Subset(self, train_indices)
        self.val_subset = Subset(self, val_indices)

    def __getitem__(self, index):
        sequence = self.image_sequences[index]
        labels = []
        images = []
        for i in range(0, len(sequence), 3):
            image_paths = sequence[i:i+3]
            grayscale_images = []
            for image_path in image_paths:
                image = Image.open(image_path)
                image = self.transform(image)
                grayscale_images.append(image)
            label = self.labels[index]
            labels.append(label)
        labels = torch.tensor(labels)
        images = torch.stack(grayscale_images, dim=0)
        return images, labels
    
    def __len__(self):
        return len(self.image_sequences)

