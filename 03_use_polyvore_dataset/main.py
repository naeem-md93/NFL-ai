import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
EMB_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
DATASET_PATH = "./../../datasets/Polyvore"
BATCH_SIZE = 64
OPT_LR = 1e-4
MAX_EPOCHS = 50


def embed_image(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = EMB_PROCESSOR(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = EMB_MODEL.get_image_features(**inputs)
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0]


def read_json_file(path):
    with open(path) as json_file:
        data = json.load(json_file)
        return data
    
    
categories_df = pd.read_csv(f"{DATASET_PATH}/polyvore/categories.csv", header=None)


categories_dict = {}
for i, row in categories_df.iterrows():
    if row[1] in ("pants", "shirt"):
        categories_dict[row[0]] = row[1]


orig_data = read_json_file(f"{DATASET_PATH}/polyvore/train_no_dup.json")

final_data = {}
for set_data in tqdm(orig_data):
    tmp_items = {}
    for item_data in set_data["items"]:
        if item_data["categoryid"] in categories_dict:
            cat_name = categories_dict[item_data["categoryid"]]

            tmp_items[cat_name] = {
                "index": item_data["index"],
                "price": item_data["price"],
                "likes": item_data["likes"],
                "image": item_data["image"],
                "categoryid": item_data["categoryid"],
                "categoryname": cat_name,
                "path": f"{DATASET_PATH}/maryland-polyvore-images-1/maryland-polyvore-images/versions/1/images/{set_data['set_id']}/{item_data['index']}.jpg",
            }
            tmp_items[cat_name]["embedding"] = embed_image(tmp_items[cat_name]["path"])
    if len(tmp_items) == 2:
        final_data[set_data["set_id"]] = {
            "set_id": set_data["set_id"],
            "likes": set_data["likes"],
            "items": tmp_items
        }
        

class CompatibilityDataset(Dataset):
    def __init__(self, _data):
        super().__init__()
        pos_samples = self.get_positive_samples(_data)
        neg_samples = self.get_negative_samples(_data, len(pos_samples))
        self.data = pos_samples + neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        score = torch.tensor([d["score"]], dtype=torch.float32)
        pants_emb = torch.tensor(d["pants"], dtype=torch.float32)
        shirt_emb = torch.tensor(d["shirt"], dtype=torch.float32)
        outfit_emb = torch.cat((shirt_emb, pants_emb), dim=0)

        return outfit_emb, score

    @staticmethod
    def get_positive_samples(_data):
        _samples = []
        for _set_data in _data:
            _samples.append({
                "score": 1,
                "shirt": _set_data["items"]["shirt"]["embedding"],
                "pants": _set_data["items"]["pants"]["embedding"],
            })
        return _samples

    @staticmethod
    def get_negative_samples(_data, n):
        _cats = {"pants": [], "shirt": []}
        for _set_data in _data:
            _cats["pants"].append(_set_data["items"]["pants"]["embedding"])
            _cats["shirt"].append(_set_data["items"]["shirt"]["embedding"])

        _samples = []
        for _ in range(n):
            _p = random.choice(_cats["pants"])
            _s = random.choice(_cats["shirt"])
            _samples.append({
                "score": 0,
                "shirt": _s,
                "pants": _p,
            })

        return _samples


class CompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(in_features=256, out_features=32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        return x


def save_checkpoint(_path, _epoch, _model, _optimizer, _prev_loss, _current_loss):
        sd = {
            "epoch": _epoch,
            "model_state_dict": _model.state_dict(),
            "optimizer_state_dict": _optimizer.state_dict(),
        }

        torch.save(sd, os.path.join(_path, "last.pt"))

        if _prev_loss > _current_loss:
            torch.save(sd, os.path.join(_path, "best.pt"))
            

current_loss = np.inf
prev_loss = current_loss

train_data, test_data = list(final_data.values())[:-120], list(final_data.values())[-120:]

train_dl = DataLoader(CompatibilityDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dl = DataLoader(CompatibilityDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model = CompatibilityModel()
optimizer = torch.optim.Adam(model.parameters(), lr=OPT_LR)

model = model.to(DEVICE)

for ep in range(1, MAX_EPOCHS + 1):
    LOSSES = []
    model.train()
    prev_loss = current_loss
    t = tqdm(train_dl)
    for bn, (outfits, scores) in enumerate(t):
        outfits = outfits.to(DEVICE)
        scores = scores.to(DEVICE)

        logits = model(outfits)
        loss = F.binary_cross_entropy_with_logits(logits, scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LOSSES.append(loss.item())

        t.set_description_str(f"Train ep: {ep} | Loss: {np.mean(LOSSES):.4f}")

    LOSSES = []
    model.eval()
    t = tqdm(test_dl)
    for bn, (outfits, scores) in enumerate(t):
        outfits = outfits.to(DEVICE)
        scores = scores.to(DEVICE)

        with torch.no_grad():
            logits = model(outfits)
            loss = F.binary_cross_entropy_with_logits(logits, scores)

        LOSSES.append(loss.item())

        t.set_description_str(f"Test ep: {ep} | Loss: {np.mean(LOSSES):.4f}")

    current_loss = np.mean(LOSSES)

    save_checkpoint("./", ep, model, optimizer, prev_loss, current_loss)
