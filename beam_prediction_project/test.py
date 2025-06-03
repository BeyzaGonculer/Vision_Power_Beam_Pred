import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from multi_feature_dataset import MultiFeatureDataset
from models.vision_signal_fusion import VisionSignalFusionNet

# 📍 Ayarlar
CSV_PATH = "./dataset/scenario5_dev_test.csv"
CAMERA_DIR = "./dataset/unit1/camera_data"
POWER_DIR = "./dataset/unit1/mmWave_data"
BATCH_SIZE = 32
NUM_CLASSES = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🔽 Test Dataset
test_dataset = MultiFeatureDataset(CSV_PATH, CAMERA_DIR, POWER_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🧠 Model
model = VisionSignalFusionNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))  # ✅ MODEL YÜKLE
model.eval()

# 🎯 Top-K hesaplama
def evaluate_topk(k=3):
    correct_topk = [0] * k
    total = 0

    with torch.no_grad():
        for images, powers, labels in test_loader:
            images = images.to(DEVICE)
            powers = powers.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, powers)
            _, topk_preds = outputs.topk(k, dim=1)
            topk_preds = topk_preds.cpu()

            for i in range(len(labels)):
                label = labels[i].item()
                for kk in range(k):
                    if topk_preds[i][kk].item() == label:
                        correct_topk[kk] += 1
                        break
                total += 1

    print("\n🎯 Test Top-k Accuracy:")
    for i in range(k):
        topk_acc = sum(correct_topk[:i+1]) / total
        print(f"Top-{i+1} Accuracy: {topk_acc:.4f}")

# ▶️ Çalıştır
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    evaluate_topk()
