# %%
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

def encode(text_or_image_list):
    if (isinstance(text_or_image_list[0], str)):
        inputs = {
            "text": text_or_image_list,
            'image': [None] * len(text_or_image_list),
            'tokenizer': tokenizer
        }
    else:
        inputs = {
            "text": [''] * len(text_or_image_list),
            'image': text_or_image_list,
            'tokenizer': tokenizer
        }
    outputs = model(**inputs)
    attention_mask = outputs.attention_mask
    hidden = outputs.last_hidden_state

    reps = weighted_mean_pooling(hidden, attention_mask)   
    embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
    return embeddings

model_name_or_path = "./VisRAG-Ret"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
model.eval()

# %%
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

PlotQA_corpus_ds = load_dataset("./dataset/VisRAG-Ret-Test-PlotQA", name="corpus", split="train")
image_embeddings = []

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

for item in tqdm(PlotQA_corpus_ds):
    image_id = item['corpus-id']
    image = item['image']
    width, height = image.size
    
    # print('image size', image.size)

    crop_width = width // 8
    crop_height = height // 8

    # # 将图像裁剪成块并转换为张量
    # cropped_images = [
    #     transform(image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')).cuda()
    #     for i in range(8) for j in range(8)
    # ]

    # # 将张量转换回PIL图像列表
    # cropped_images_pil = [transforms.ToPILImage()(img.cpu()) for img in cropped_images]
    
    cropped_images_pil = []

    for i in range(8):
        for j in range(8):
            # 裁剪图像块
            cropped_image = image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')
            # print('cropped image size', cropped_image.size)

            # 创建一个全黑的mask
            full_image = Image.new('RGB', (width, height))
            # print(encode([full_image]))
            
            # 应用mask
            full_image.paste(cropped_image, (j * crop_width, i * crop_height))

            cropped_images_pil.append(full_image.convert('RGB'))
            
            # 裁剪图像块
            # cropped_image = image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')

            # # 将裁剪后的图像块转换为张量并移动到GPU
            # cropped_image_tensor = transform(cropped_image).cuda()

            # # 创建一个与原始图像相同尺寸的全黑图像张量
            # full_image_tensor = torch.zeros((3, height, width)).cuda()

            # # 将裁剪图像块粘贴到全黑图像张量的相应位置
            # full_image_tensor[:, i * crop_height:(i + 1) * crop_height, j * crop_width:(j + 1) * crop_width] = cropped_image_tensor

            # # 将张量转换回PIL图像并添加到列表中
            # full_image = transforms.ToPILImage()(full_image_tensor.cpu())
            # cropped_images_pil.append(full_image)
            
    # fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(cropped_images_pil[i])
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.savefig('cropped_images.png')

    # 编码图像
    with torch.no_grad():
        embedding = encode(cropped_images_pil)
    
    image_embeddings.append(embedding)

image_embeddings = np.array(image_embeddings)    
np.save(f"embeddings\\PlotQA_corpus_embeddings_8x8_mask.npy", image_embeddings)