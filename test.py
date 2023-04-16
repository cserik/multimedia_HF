import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CustomResNet50
from collections import Counter

# Define the transformation to be applied to each image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

images_data='C:\\Users\\bwim_erik\\Desktop\\BME_MSc\\multimedia\\HF_lstm\\lstm\\test_data\\test_images'

class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Load the best PyTorch model weights
model = CustomResNet50()
model.load_state_dict(torch.load('best_model_weights.pth'))
model.eval()

final_predictions=[]
# Iterate over every created directory in images_data
for root, dirs, files in os.walk(images_data):
    for dir in dirs:
        try:
            # Transform every loaded image
            grayscale_images = []
            image_paths = os.listdir(os.path.join(images_data, dir))
            labels=[]
            for i in range(len(image_paths)):
                image_path = os.path.join(images_data, dir, image_paths[i])
                image = Image.open(image_path)
                image = transform(image)
                grayscale_images.append(image)

                # Every 3 following grayscale images should be merged to one tensor
                if (i+1) % 3 == 0:
                    images = torch.stack(grayscale_images, dim=0)
                    images=images.view(-1, 3, 224, 224)
                    # Feed the images to the model and print the output to stdout
                    with torch.no_grad():
                        output = model(images)
                        _, index = torch.max(output.data, 1)
                        labels.append(int(index[0]))
                    grayscale_images = [] # Reset the grayscale images for the next set of images
            most_common = max(set(labels), key=labels.count)
            final_predictions.append(class_names[most_common])
            print(len(final_predictions))
        except Exception as e:
            print(f"Error processing images in directory {dir}: {e}")

# Open a new text file in write mode
with open('final_predictions.txt', 'w') as file:
    # Iterate over the final predictions list and write each item to a new line in the file
    for prediction in final_predictions:
        file.write(prediction + '\n')
    


