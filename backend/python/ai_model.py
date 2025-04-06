import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2

class FingerprintCNN(nn.Module):
    """
    CNN model for fingerprint feature extraction.
    """
    def __init__(self):
        super(FingerprintCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Feature vector size
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SiameseNetwork(nn.Module):
    """
    Siamese network for fingerprint matching.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = FingerprintCNN()
        
    def forward(self, input1, input2):
        # Get feature vectors for both inputs
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        
        return output1, output2

def train_model(train_loader, val_loader, num_epochs=10):
    """
    Train the fingerprint recognition model.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        
    Returns:
        nn.Module: Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    
    # Loss function and optimizer
    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(data1, data2)
            loss = criterion(output1, output2, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data1, data2, target in val_loader:
                data1, data2, target = data1.to(device), data2.to(device), target.to(device)
                output1, output2 = model(data1, data2)
                loss = criterion(output1, output2, target)
                val_loss += loss.item()
                
                # Calculate accuracy
                similarity = F.cosine_similarity(output1, output2)
                predicted = (similarity > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {100*correct/total:.2f}%')
    
    return model

def save_model(model, path):
    """
    Save the trained model.
    
    Args:
        model (nn.Module): Trained model
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(path):
    """
    Load a trained model.
    
    Args:
        path (str): Path to the saved model
        
    Returns:
        nn.Module: Loaded model
    """
    model = SiameseNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def extract_features(model, img):
    """
    Extract features from a fingerprint image using the model.
    
    Args:
        model (nn.Module): Trained model
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        numpy.ndarray: Feature vector
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Preprocess image
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.cnn(img_tensor)
    
    return features.cpu().numpy() 