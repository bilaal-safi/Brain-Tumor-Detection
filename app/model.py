import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# Your CNNModel definition from training
class CNNModel(nn.Module):  
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.cnv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.cnv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.leakyRelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, 2)
      
    def forward(self, x):
        out = self.leakyRelu(self.cnv1(x))
        out = self.maxpool1(out)
        out = self.leakyRelu(self.cnv2(out))
        out = self.maxpool2(out)
        out = self.leakyRelu(self.cnv3(out))
        out = self.maxpool3(out)
        out = self.leakyRelu(self.cnv4(out))
        out = self.maxpool4(out)
        out = out.view(out.size(0), -1)
        out = self.leakyRelu(self.fc1(out))
        out = self.fc2(out)
        return out


base = os.path.dirname(__file__)
# or you can write the whole path till app folder: base = 'your whole path/Brain-tumor-classification/app'
model_path = os.path.join(base, 'cnn_model.pth')
model = CNNModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode


def image_pre(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def predict(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()  # 0 or 1
