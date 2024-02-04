from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import sys
import torch
sys.path.append("D:\\projects\\CLIP")

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
# model = model.to(device)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('clipseg_weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

# load and normalize image
input_image = Image.open('D:/projects/chatgpt/yunxiaochu/1.jpg')
input_image = Image.open('D:/projects/chatgpt/covered/covered_text_human.jpeg')

# or load from URL...
# image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
# input_image = Image.open(requests.get(image_url, stream=True).raw)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])
img = transform(input_image).unsqueeze(0).to(device)

prompts = ["human", "reflection", "what covers the text", "obstruction"]
# prompts = ["what covers text area", "human","person"]
# inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

# predict
with torch.no_grad():
    preds = model(img.repeat(4,1,1,1), prompts)[0]

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(input_image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
[ax[i+1].text(0, -15, prompts[i]) for i in range(4)]

filename = f"mask.png"
# here we save the second mask
plt.imsave(filename,torch.sigmoid(preds[0][0]))

import cv2

img2 = cv2.imread(filename)

gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, bw_image) = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# fix color format
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

Image.fromarray(bw_image)