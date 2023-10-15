import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AgeData(Dataset):
    def __init__(self, files):
        super().__init__()
        self.x = files['image_url']
        self.y = files['label']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(224, 224))
        img = np.array(img) / 255.
        img = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Resize(224),
                        # transforms.Normalize(mean=0.5, std=0.5)
                    ])(img)
        img = img.view(3, 224, 224)
        return img.float().to(device), torch.tensor(int(self.y[idx]) / 1.0).to(device)

if __name__ == '__main__':

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train['image_url'] = train['image_url'].apply(lambda x : 'data/'+x[15:])
    test['image_url'] = test['image_url'].apply(lambda x: 'data/'+x[15:])

    train_ds = AgeData(train)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

    test_ds = AgeData(test)
    test_dl = DataLoader(test_ds, batch_size=16, drop_last=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.required_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )
    model = model.to(device)

    loss = nn.L1Loss()
    criterion = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        model.train()
        epoch_losses = []
        for idx, batch in enumerate(iter(train_dl)):
            criterion.zero_grad()
            img, y = batch
            pred = model(img).squeeze(-1)
            batch_loss = loss(pred, y)
            batch_loss.backward()
            criterion.step()
            epoch_losses.append(batch_loss.cpu().detach().numpy())
            if idx % 100 == 0:
                print(batch_loss.item())
        epoch_loss = np.array(epoch_losses).mean()

        val_mae_list = []
        for idx, batch in enumerate(iter(test_dl)):
            model.eval()
            val_img, val_y = batch
            val_pred = model(val_img).squeeze(-1)
            val_mae_list.append(torch.abs(val_pred - val_y).cpu().detach().numpy())

        epoch_mae = np.array(val_mae_list).mean()
        print(f"Epoch {epoch} Train Loss {epoch_loss} Val MAE {epoch_mae}")

    torch.save(model.state_dict(), 'model/resnet18.pt')





