from model import CNNLSTM, ResNetLSTM
from cnn_lstm_model import LSTMNet
from dataset import VideoDataset
from torch.utils.data import DataLoader, random_split
from torch import nn, optim, manual_seed, no_grad
import torch

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define the dataset and dataloaders
# Set random seed for reproducibility
manual_seed(0)

# Define hyperparameters
batch_size = 2
learning_rate = 0.001
num_epochs = 10

# Create dataset and dataloaders
dataset = VideoDataset(data_path='C:\\Users\\bwim_erik\\Desktop\\BME_MSc\\multimedia\\HF_lstm\\lstm\\data\\image_data\\data\\video_data')
train_subset = dataset.train_subset
val_subset = dataset.val_subset
trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

# Define the model and optimizer
model = ResNetLSTM()
# move model to device
model.to(device)
# define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the model to training mode
model.train()

# Zero the parameter gradients
optimizer.zero_grad()
# Train the network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):

        torch.cuda.empty_cache()

        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.view(-1, 1, 224, 224)
        # move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        shape=outputs.shape
        labels=labels.view(shape[0], 5)
        loss = criterion(outputs,labels)
        loss.backward()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[Epoch %d, Batch %5d] Loss: %.3f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / 10))
            running_loss = 0.0
            #the steps are here because so the batch size is larger (can't pass large batches to gpu do we used small batch size instead-> 
            # but it is almost the same because after x*small_bath size will be the weight updated)
            optimizer.step()
            optimizer.zero_grad()
            
    # Validation
    model.eval()    # set the model to evaluation mode
    with no_grad():
        total_correct = 0
        total_samples = 0
        for i, data in enumerate(valloader):
            inputs, labels = data
            inputs = inputs.view(-1, 1, 224, 224)
            # move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            shape=outputs.shape
            labels=labels.view(shape[0], 5)
            _, predicted = torch.max(outputs.data, 1)
            _, expected = torch.max(labels.data, 1)
            total_samples += expected.size(0)
            total_correct += (predicted == expected).sum().item()
        
        # Calculate the validation accuracy
        val_accuracy = 100 * total_correct / total_samples
        print('Validation accuracy: %.2f%%' % val_accuracy)
        
    model.train()   # set the model back to training mode

