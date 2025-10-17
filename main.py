# main.py
from src.train import train

if __name__ == "__main__":
    data_root = "data/EuroSAT"
    train_csv = "data/train.csv"
    val_csv = "data/validation.csv"

    train(train_csv, val_csv, data_root, epochs=15)
