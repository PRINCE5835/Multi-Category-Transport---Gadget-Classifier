import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

batch = 32
epochs = 5
data_path = "../dataset"
save_path = "classifier.pth"

my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_data = datasets.ImageFolder(data_path, transform=my_transform)
train_len = int(0.8 * len(full_data))
val_len = len(full_data) - train_len
train_data, val_data = torch.utils.data.random_split(full_data, [train_len, val_len])
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch)

class_names = full_data.classes
print("classes found:", class_names)
json.dump(class_names, open("class_names.json", "w"))

my_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for param in my_model.parameters():
    param.requires_grad = False

num_feats = my_model.classifier[1].in_features
my_model.classifier[1] = nn.Linear(num_feats, len(class_names))
my_model = my_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.classifier[1].parameters(), lr=0.001)

for ep in range(epochs):
    my_model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = my_model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {ep+1} done, loss: {total_loss/len(train_loader):.4f}")

    my_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = my_model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print(f"val accuracy: {100*correct/total:.2f}%")

torch.save(my_model.state_dict(), save_path)
print("saved model to", save_path)
