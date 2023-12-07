
from dataseto import MultimodalDataset,preprocess_and_save,presave,preprocess_and_save2
from mmodel import MultimodalClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel, Wav2Vec2Model, AdamW
import torchaudio
from PIL import Image
import torchvision.transforms as transforms
from torch.optim import AdamW
# File paths
audio_root = '../H1_AUDIO'  # Replace with your actual audio root path
image_root = '../H1_IMAGE'  # Replace with your actual image root path
label_root = '../H1_ZONELABELS'  # Replace with your actual label root path
# label_root='../H1label'

# Initialize dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instantiate the dataset

# datastore=preprocess_and_save(audio_root=audio_root, image_root=image_root, label_root=label_root, csv_output_path='output.csv')
# datastore=preprocess_and_save2(audio_root=audio_root, image_root=image_root, label_root=label_root, csv_output_path='output_new.csv')
# datasave=presave('output.csv')
# exit(0)
dataset = MultimodalDataset('output_2.csv',transform=transform)


# Split the dataset
total_size = len(dataset)
train_size = int(total_size * 0.7)

val_size = total_size - train_size

# Seed the random number generator for reproducibility
torch.manual_seed(3407)

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)





# Load the pre-trained Wav2Vec2 and ViT models
# wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Instantiate the model
model= MultimodalClassifier(
    audio_feature_size=768,  # This should match the feature size of Wav2Vec2
    image_feature_size=768,  # This should match the feature size of ViT
    num_classes=2
)

# Move the model to the GPU if available
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# # Define optimizer
# optimizer = AdamW(model.parameters(), lr=1e-5)
#
# # Define loss function
# criterion = nn.CrossEntropyLoss()


# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_function = nn.CrossEntropyLoss()


train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
# Training and validation loop
num_epochs = 30
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for waveforms, images,env, labels in train_loader:
        # waveforms=waveforms.unsqueeze(1)
        waveforms,env, labels = waveforms.to(device),  env.to(device),labels.to(device)
        imgs_dev=[]
        for image in images:
            image = image.to(device)
            imgs_dev.append(image)
        optimizer.zero_grad()

        # import numpy as np
        # from transformers import Wav2Vec2Processor, Wav2Vec2Model
        #
        # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # batch_waveforms = [np.random.randn(799) for _ in range(16)]  # Example waveforms
        #
        # # Normalize and process each waveform in the batch
        # processed_waveforms = [waveform / np.max(np.abs(waveform)) for waveform in batch_waveforms]
        # processed_waveforms = processor(processed_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        # processed_waveforms = processed_waveforms.input_values
        # processed_waveforms = processed_waveforms.to(device)

        logits = model(waveforms, imgs_dev,env)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
       # print('forward')

    # Calculate average training loss
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for waveforms, images, env,labels in val_loader:
            waveforms,  env,labels = waveforms.to(device),env.to(device), labels.to(device)

            imgs_dev = []
            for image in images:
                image = image.to(device)
                imgs_dev.append(image)

            logits = model(waveforms, imgs_dev,env)
            loss = loss_function(logits, labels)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(model.state_dict(), 'image_best_modal.pth')
        print(f"Epoch {epoch + 1}: New best model saved with loss {best_val_loss:.4f}")

# Save the model
# torch.save(model.state_dict(), 'multimodal_classifier_img.pth')

print("Training completed.")
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.png')

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy.png')
