from model import CNNLSTM
from dataset import VideoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim, manual_seed, no_grad
import torch

# Define the dataset and dataloaders
# Set random seed for reproducibility
manual_seed(0)

# Define hyperparameters
batch_size = 10
learning_rate = 0.001
num_epochs = 10

# Create dataset and dataloaders
dataset = VideoDataset(data_path='C:\\Users\\bwim_erik\\Desktop\\BME_MSc\\multimedia\\HF_lstm\\lstm\\data\\image_data\\data\\video_data')
train_sampler = dataset.train_sampler
val_sampler = dataset.val_sampler
trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Define the model and optimizer
model = CNNLSTM()
# define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set the model to training mode
model.train()

# Train the network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.view(-1, 1, 224, 224)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[Epoch %d, Batch %5d] Loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
    # Validation
    model.eval()    # set the model to evaluation mode
    with no_grad():
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(valloader):
            inputs, labels = data
            inputs = inputs.view(-1, 1, 224, 224)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        # Calculate the validation accuracy
        val_accuracy = 100 * total_correct / total_samples
        print('Validation accuracy: %.2f%%' % val_accuracy)
        
    model.train()   # set the model back to training mode