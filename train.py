from model import CNNLSTM, ResNetLSTM, ResNetLSTM2, CustomResNet50
from cnn_lstm_model import LSTMNet
from dataset import VideoDataset
from torch.utils.data import DataLoader, random_split
from torch import nn, optim, manual_seed, no_grad
import torch
import numpy as np
import time

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define the dataset and dataloaders
# Set random seed for reproducibility
manual_seed(0)

# Define hyperparameters
batch_size = 32
learning_rate = 0.0005
num_epochs = 40
patience = 4

# Create dataset and dataloaders
dataset = VideoDataset(data_path='C:\\Users\\bwim_erik\\Desktop\\BME_MSc\\multimedia\\HF_lstm\\lstm\\data\\image_data\\data\\video_data')
train_subset = dataset.train_subset
val_subset = dataset.val_subset
trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

# Define the model and optimizer
model = CustomResNet50()
# move model to device
model.to(device)

# Define the loss function
criterion = nn.NLLLoss()

# Define the optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True, factor=0.5)

# Set the model to training mode
model.train()

# Define variables to store the best validation accuracy and corresponding model weights
best_val_acc = 0
best_weights = None
best_val_loss = 1000

# Define lists to store train and validation loss and accuracy
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

# Train the network
for epoch in range(num_epochs):
    running_loss = 0.0

    # Train loop
    for i, data in enumerate(trainloader):
        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.view(-1, 3, 224, 224)
        # move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        labels = labels.view(outputs.shape[0])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2 == 0:    # print every 10 mini-batches
            print('[Epoch %d, Batch %5d] Loss: %.3f' % (epoch + 1, i, running_loss / 2))
            running_loss = 0.0

    # Compute the training loss and accuracy
    with torch.no_grad():
        model.eval()
        train_loss, total_train_samples, total_train_correct = 0, 0, 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.view(-1, 3, 224, 224)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.view(outputs.shape[0])
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            total_train_correct += (predicted == labels).sum().item()
            train_loss += criterion(outputs, labels).item()

        train_acc = 100 * total_train_correct / total_train_samples
        train_loss = train_loss / len(trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

    # Validation loop
    model.eval()    # set the model to evaluation mode
    with torch.no_grad():
        total_correct, total_samples, val_loss = 0, 0, 0
        for i, data in enumerate(valloader):
            inputs, labels = data
            inputs = inputs.view(-1, 3, 224, 224)
            # move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.view(outputs.shape[0])
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            # calculate validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        # Calculate the validation accuracy and loss
        val_accuracy = 100 * total_correct / total_samples
        val_loss /= len(valloader)
        print('Validation accuracy: %.2f%%, Validation Loss: %.3f' % (val_accuracy, val_loss))

        # save train and validation loss and accuracy in lists
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping, no improvement in validation loss in %d epochs." % patience)
            break
    
    # set model back to training mode
    model.train()

# save best model weights
torch.save(best_weights, 'best_model_weights.pth')

# save train and validation loss and accuracy to txt file
with open('loss_and_acc.txt', 'w') as f:
    f.write(' '.join(map(str, train_loss_list)))
    f.write('\n')
    f.write(' '.join(map(str, train_acc_list)))
    f.write('\n')
    f.write(' '.join(map(str, val_loss_list)))
    f.write('\n')
    f.write(' '.join(map(str, val_acc_list)))

print("Training complete.")