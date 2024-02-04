from PIL import Image
import requests
import cv2
import torch
import matplotlib.pyplot as plt
import datetime


def scale(image_path, mask_path):
    img1 = cv2.imread(image_path)  # 替换为Img1的路径
    img2 = cv2.imread(mask_path)  # 替换为Img2的路径

    # 获取第一个图像的尺寸
    height, width = img1.shape[:2]

    # 缩放第二个图像以匹配第一个图像的尺寸
    img2_resized = cv2.resize(img2, (width, height))

    # 保存或显示结果
    cv2.imwrite(mask_path, img2_resized)


device = torch.device('cuda')
url = "https://github.com/snowymo/hugging_learn/blob/image/covered_text_reflection2.jpeg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
# cv2.imwrite("input.jpg",image)
# filename = f"input.jpg"
# here we save the second mask
# plt.imsave(filename,image)
input_url = 'D:/projects/chatgpt/covered/covered_text_reflection4.jpeg'
input_image = Image.open(input_url)

cv_img = cv2.imread(input_url)
cv2.imwrite("input.jpg", cv_img)

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

print(datetime.datetime.now())
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
print(datetime.datetime.now())
# prompts = ["what covers text area", "human", "person"]
prompts = ["light", "reflection", "what covers the text", "obstruction"]
prompts = ["light"]
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt").to(device)

# predict
for i in range(1):
    print("before predict", datetime.datetime.now())
    with torch.no_grad():
        outputs = model(**inputs)
    print("after predict", datetime.datetime.now())

    preds = outputs.logits.unsqueeze(1).cpu()

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
[ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))]

filename = f"covered_man_mask.png"
# here we save the second mask
plt.imsave(filename, torch.sigmoid(preds[0][0]))
scale("input.jpg", "covered_man_mask.png")

img2 = cv2.imread(filename)
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, bw_image) = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)

# fix color format
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

Image.fromarray(bw_image)
