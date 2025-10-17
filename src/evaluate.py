# src/evaluate.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import EuroSATDataset
from src.model import get_model

def visualize_predictions(model_path, csv_path, data_root, num_images=8):
    """
    Show a few sample predictions from the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = EuroSATDataset(csv_path, data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    class_names = dataset.data["ClassName"].unique().tolist()

    plt.figure(figsize=(16, 8))
    shown = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            img = images[0].permute(1, 2, 0).cpu().numpy()

            plt.subplot(2, num_images // 2, shown + 1)
            plt.imshow(img)
            plt.title(
                f"Pred: {class_names[preds.item()]}\nTrue: {class_names[labels.item()]}",
                color="green" if preds.item() == labels.item() else "red"
            )
            plt.axis("off")
            shown += 1

            if shown == num_images:
                break

    plt.suptitle("EuroSAT Predictions (Before vs After Model Learning)", fontsize=14)
    plt.tight_layout()
    plt.show()
