import os, json, random,torch,torch.nn as nn,torch.nn.functional as F, torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from vit_pytorch.efficient import ViT
from os.path import join
from natsort import natsorted
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 42

train_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

test_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

class CatsDogsDataset1(Dataset):
    def __init__(self, file_list=None, transform=None, train=True, test=False):
        root = "/data/wuxiaopeng/datasets/compare_2_picture/centernet/train_generate/generate/label_final"
        print(len("/data/wuxiaopeng/datasets/compare_2_picture/"))
        self.data = []
        self.data_label = []
        for i in os.listdir(root):
            json_path = join(root, i)

            with open(json_path, "r") as f:
                json_data = json.load(f)

                path = json_data["path"][:44] + "centernet/" + json_data["path"][44:]
                if os.path.exists(path):
                    self.data.extend([(path, 0), (path, 1)])

        self.test = test
        self.imgs = []
        self.label = []

        random.seed(5)
        random.shuffle(self.data)
        imgs_num = len(self.data)
        # print(self.data)
        if self.test:
            pass
        else:
            if train:
                # self.data = self.data[:int(0.8*imgs_num)]
                self.data = self.data[:-2000]
            else:
                # self.data = self.data[int(0.8*imgs_num):]
                self.data = self.data[-2000:]

        self.file_list = self.data
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),transforms.RandomResizedCrop(224),
             transforms.ToTensor(),])   #transforms.RandomHorizontalFlip()

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx][0]
        label = self.file_list[idx][1]
        data = Image.open(img_path)
        if label == 1:
            data = data.rotate(180)
        img_transformed = self.transform(data)

        # label = img_path.split("/")[-1].split(".")[0]
        # label = 1 if label == "dog" else 0

        return img_transformed, label
class CatsDogsDataset(Dataset):
    def __init__(self, file_list=None, transform=None, train=True, test=False):
        root = "/data/wuxiaopeng/datasets/detect_inverse_image/image224x224"
        self.data = []
        for i in os.listdir(root):
            path = os.path.join(root, i)
            self.data.extend([(path, 0), (path, 1)])

        random.seed(5)
        random.shuffle(self.data)

        if train:
            self.data = self.data[:-2000]
        else:
            self.data = self.data[-2000:]

        self.file_list = self.data
        if train:
            self.transform = transforms.Compose(
                [
                 # transforms.Resize((224, 224)),
                 transforms.RandomResizedCrop(224),
                 transforms.ToTensor(),])   #transforms.RandomHorizontalFlip()
        else:
            self.transform = transforms.Compose(
                [
                 # transforms.Resize((224, 224)),
                 transforms.ToTensor(), ])  # transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(224),

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx][0]
        label = self.file_list[idx][1]
        data = Image.open(img_path)
        data = data.convert("RGB")
        if label == 1:
            data = data.rotate(180)
        img_transformed = self.transform(data)
        return img_transformed, label

class CatsDogsDataset_test(Dataset):
    def __init__(self, file_list=None, transform=None, train=True, test=False):
        aaa = "/data/wuxiaopeng/workspace/compare_2_picture/tmp1/image"
        self.file_list = []
        for i in natsorted(os.listdir(aaa)):
            if "0" in i.split("_")[1]:
                path = os.path.join(aaa, i)
                self.file_list.append(path)
        self.transform = transforms.Compose( [transforms.Resize((224, 224)), transforms.ToTensor(),])
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = Image.open(img_path)
        data = data.convert("RGB")
        img_transformed = self.transform(data)
        return img_transformed, 1

train_data = CatsDogsDataset(train=True)
valid_data = CatsDogsDataset(train=False)
test_data = CatsDogsDataset_test()
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

efficient_transformer = Linformer(dim=128,seq_len=49+1,depth=12,heads=8, k=64)
model = ViT(dim=128, image_size=224,patch_size=32,num_classes=2,transformer=efficient_transformer,channels=3,).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for numi, (data, label) in  enumerate(train_loader):#for data, label in tqdm(train_loader):
        # if numi==50:
        #     break
        print("\r"+str(numi), end="", flush=True)
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for numi,(data, label ) in enumerate(valid_loader):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

        total = 0
        for data, label in test_loader:
            data = data.to(device)
            output = model(data)
            prob = F.softmax(output, dim=1)
            predict = torch.argmax(prob,dim=1)
            acc = torch.sum(predict)
            total+=acc.item()

    print(
        f"\r Epoch:{epoch} loss:{epoch_loss:.4f},{epoch_accuracy:.4f} val_loss:{epoch_val_loss:.4f},{epoch_val_accuracy:.4f};{total}\n"
    )

    torch.save(model.state_dict(), "./models/"+str(epoch)+".pth")
