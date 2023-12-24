# Import the required dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load the MNIST dataset
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

# Define the Image Classifier class
class ImageClassifier(nn.Module):
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
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)   
        )
        
    def forward(self, x):
        return self.model(x)

# Create an instance of the neural network, loss, and the optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fun = nn.CrossEntropyLoss()

# Training loop
if __name__ == "__main__":
    with open('model_state.pt', 'wb') as f:
      clf.load_state_dict(load(f))

    img=Image.open('0.jpg') 
    img_tensor=ToTensor()(img).unsqueeze(0).to('cuda')
    print(torch.argmax(clf(img_tensor)))
  #  for epoch in range(10):
       ## for batch in dataset:
       #     x, y = batch
        #    x, y = x.to('cuda'), y.to('cuda')
         ##   yhat = clf(x)
          #  loss = loss_fun(yhat, y)
            # Apply backpropagation
         #   opt.zero_grad()
          #  loss.backward()
      #      opt.step()
      #  print(f"epoch {epoch} loss is {loss.item()}")

    # Save the model state
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)
