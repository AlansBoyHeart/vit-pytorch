import  os, torch
from natsort import natsorted
from torchvision import transforms
from linformer import Linformer
from vit_pytorch.efficient import ViT
from PIL import Image
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
def test(path):
	efficient_transformer = Linformer(
		dim=128,
		seq_len=49 + 1,  # 7x7 patches + 1 cls-token
		depth=12,
		heads=8,
		k=64
	)
	model = ViT(
		dim=128,
		image_size=224,
		patch_size=32,
		num_classes=2,
		transformer=efficient_transformer,
		channels=3,
	)
	model.load_state_dict(torch.load("./models/194.pth",map_location="cpu"))
	model.to(device)


	with torch.no_grad():
		img = Image.open(path)
		img = img.rotate(180)
		img = train_transforms(img)
		img = img.unsqueeze(0)
		img = torch.autograd.Variable(img).to(device)
		output = model(img )
		prob = F.softmax(output, dim=1)
		predict = torch.argmax(prob)
		return predict, prob



def main1():
	aaa = "/data/wuxiaopeng/workspace/compare_2_picture/tmp1/image"
	for i in natsorted(os.listdir(aaa)):
		if i in ['33_0.jpg', '70_0.jpg', '114_0.jpg', '167_0.jpg', '168_0.jpg', '187_0.jpg', '229_0.jpg', '252_0.jpg', '286_0.jpg', '312_0.jpg', '409_0.jpg', '417_0.jpg', '460_0.jpg', '471_0.jpg', '526_0.jpg', '544_0.jpg', '570_0.jpg', '613_0.jpg', '654_0.jpg']:
			if "0" in i.split("_")[1]:
				path = os.path.join(aaa, i)
				predict, prediction = test(path)
				if predict == 0:
					res.append(i)
					print(i, prediction)
	print(res)




if __name__ == '__main__':
	res = []
	main1()

