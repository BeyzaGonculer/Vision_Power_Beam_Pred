import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from multi_feature_dataset import MultiFeatureDataset
from models.vision_signal_fusion import VisionSignalFusionNet

CSV_PATH = "./dataset/scenario5_dev_train.csv"
CAMERA_DIR = "./dataset/unit1/camera_data"
POWER_DIR = "./dataset/unit1/mmWave_data"
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = MultiFeatureDataset(CSV_PATH, CAMERA_DIR, POWER_DIR, transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VisionSignalFusionNet(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        all_preds = []
        all_labels = []

        for images, powers, labels in train_loader:
            images = images.to(DEVICE)
            powers = powers.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, powers)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f" Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
    torch.save(model.state_dict(), "model.pth") 
