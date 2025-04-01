# %%
import os
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

# PlotQA_corpus_ds = load_dataset("./dataset/VisRAG-Ret-Test-PlotQA", name="corpus", split="train")
image_embeddings = []

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

folder_path = './image_plus_summary/PlotQA/above/'

# 遍历文件夹下的所有文件
for file_name in tqdm(os.listdir(folder_path)):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file_name)
    # 打开图像
    image = Image.open(file_path)
    width, height = image.size

    crop_width = width // 4
    crop_height = height // 4

    # 将图像裁剪成块并转换为张量
    # cropped_images = [
    #     transform(image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')).cuda()
    #     for i in range(4) for j in range(4)
    # ]
    
    cropped_images = [
        transform(image.crop((0, i * crop_height, width, (i + 1) * crop_height)).convert('RGB')).cuda()
        for i in range(4)
    ]
    
    # # 指定输出文件夹路径
    # output_dir = './tmp/'
    # # 创建输出文件夹（如果不存在）
    # os.makedirs(output_dir, exist_ok=True)
    # # 遍历裁剪后的图像并使用plt保存到tmp文件夹下
    # for i, cropped_image in enumerate(cropped_images):
    #     # 使用plt画出裁剪后的图像并保存
    #     plt.imshow(cropped_image.cpu().permute(1, 2, 0))  # 将张量转换为图像格式
    #     plt.axis('off')  # 隐藏坐标轴
    #     # 生成保存文件名
    #     save_path = os.path.join(output_dir, f"cropped_image_{i}.png") 
    #     # 保存图像
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #     plt.close()
    # exit(0)
    
    # 将张量转换回PIL图像列表
    cropped_images_pil = [transforms.ToPILImage()(img.cpu()) for img in cropped_images]

    # 编码图像
    with torch.no_grad():
        embedding = encode(cropped_images_pil)
    
    image_embeddings.append(embedding)

image_embeddings = np.array(image_embeddings)    
np.save(f"embeddings\\PlotQA_corpus_embeddings_4x1_with_summary_above.npy", image_embeddings)