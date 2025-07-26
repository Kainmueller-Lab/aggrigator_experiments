import tkinter as tk
from tkinter import ttk
import torch
#from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import os

from datasets.Weedsgalore.weedsgalore_dataset_creation import weedsgalore_dataset


# Dummy Dataset (replace with your own)
class DummySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        self.name = name
        self.length = 10

    def __getitem__(self, idx):
        img = np.random.rand(128, 128, 3)
        mask = np.random.rand(128, 128)
        return {
            "image": img,
            "mask": mask,
            "id": f"{self.name}_{idx}"
        }

    def __len__(self):
        return self.length

# Available datasets
datasets = {
    "Dataset A": DummySegmentationDataset("DatasetA"),
    "Dataset B": DummySegmentationDataset("DatasetB"), 
    "Weedsgalore":weedsgalore_dataset(image_path="C:/Users/cwinklm/Documents/Data/weedsgalore-dataset", 
                          mask_path="C:/Users/cwinklm/Documents/Data/weedsgalore-dataset", 
                          uq_map_path="C:/Users/cwinklm/Documents/Data/weedsgalore/rgb_test",
                          prediction_path="C:/Users/cwinklm/Documents/Data/weedsgalore/rgb_test",
                          task = "crops_vs_weed", 
                          uq_method= "dropout", 
                          decomp= "pu", 
                          semantic_mapping_path="")
}

# GUI Class
class SegmentationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentation Viewer")

        # Dataset selection
        self.dataset_var = tk.StringVar()
        self.dataset_menu = ttk.Combobox(root, textvariable=self.dataset_var, values=list(datasets.keys()))
        self.dataset_menu.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.dataset_menu.bind("<<ComboboxSelected>>", self.load_dataset)

        # Matplotlib figure
        self.fig, self.axs = plt.subplots(1, 2, figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

        # Notes field
        self.notes_entry = tk.Entry(root, width=60)
        self.notes_entry.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # Buttons
        self.prev_btn = tk.Button(root, text="Previous", command=self.prev_sample)
        self.prev_btn.grid(row=3, column=0, padx=5, pady=5)

        self.next_btn = tk.Button(root, text="Next", command=self.next_sample)
        self.next_btn.grid(row=3, column=1, padx=5, pady=5)

        self.save_btn = tk.Button(root, text="Save", command=self.save_sample)
        self.save_btn.grid(row=3, column=2, padx=5, pady=5)

        # State
        self.dataset = None
        self.loader = None
        self.iterator = None
        self.current_sample = None
        self.current_index = 0

    def load_dataset(self, event=None):
        name = self.dataset_var.get()
        self.dataset = datasets[name]
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.iterator = iter(self.loader)
        self.current_index = 0
        self.show_sample(self.current_index)

    def get_sample(self, index):
        if index < 0 or index >= len(self.dataset):
            return None
        return self.dataset[index]

    def show_sample(self, index):
        sample = self.get_sample(index)
        if sample is None:
            return
        self.current_sample = sample
        img = sample["image"]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        mask = sample["mask"]

        self.axs[0].imshow(img)
        self.axs[0].set_title("Image")
        self.axs[0].axis("off")

        self.axs[1].imshow(mask, cmap="gray")
        self.axs[1].set_title("Mask")
        self.axs[1].axis("off")

        self.canvas.draw()

    def next_sample(self):
        if self.dataset is None:
            return
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self.show_sample(self.current_index)

    def prev_sample(self):
        if self.dataset is None:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.show_sample(self.current_index)

    def save_sample(self):
        if self.current_sample is None:
            return

        img = self.current_sample["image"]
        mask = self.current_sample["mask"]
        sample_id = self.current_sample["id"]
        notes = self.notes_entry.get()

        os.makedirs("saved_samples", exist_ok=True)
        plt.imsave(f"saved_samples/{sample_id}_image.png", img)
        plt.imsave(f"saved_samples/{sample_id}_mask.png", mask, cmap="gray")

        csv_path = "saved_samples/annotations.csv"
        df = pd.DataFrame([[sample_id, notes]], columns=["id", "notes"])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        print(f"Saved: {sample_id} with notes.")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationViewer(root)
    root.mainloop()
