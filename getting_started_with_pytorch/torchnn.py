# Import the dependancies
import os
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get Data
os.makedirs("./data", exist_ok=True)
mnist_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(mnist_data, batch_size=32)

# Image Classifier
class ImageClassifier(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to("cpu")
optimizer = Adam(clf.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10):
        for batch in dataset:
            X, y = batch
            X, y = X.to("cpu"), y.to("cpu")
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            
            # Apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch:{epoch} loss is {loss.item()}")
    
    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)
        
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))
        
    img = Image.open('img_3.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to("cpu")
    print(f"The Image is of digit: {torch.argmax(clf(img_tensor))}")
    
    import glob
    
    for image in glob.glob("./*.jpg"):
        print(os.path.basename(image))