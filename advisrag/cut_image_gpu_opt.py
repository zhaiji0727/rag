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

# datasets = ['ArxivQA', 'ChartQA', 'InfoVQA', 'MP-DocVQA', 'PlotQA', 'SlideVQA']
datasets = ['PlotQA']
# datasets = ['ArxivQA', 'ChartQA', 'InfoVQA', 'MP-DocVQA']

height_divisor = 10
width_divisor = 10

for dataset in datasets:
    corpus_ds = load_dataset(f"./dataset/VisRAG-Ret-Test-{dataset}", name="corpus", split="train")
    image_embeddings = []

    # 定义转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    for item in tqdm(corpus_ds, desc=f'Embedding {dataset} corpus: '):
        image_id = item['corpus-id']
        image = item['image']
        width, height = image.size

        crop_width = width // width_divisor
        crop_height = height // height_divisor

        # 将图像裁剪成块并转换为张量
        cropped_images = [
            transform(image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')).cuda()
            for i in range(height_divisor) for j in range(width_divisor)
        ]

        # 将张量转换回PIL图像列表
        cropped_images_pil = [transforms.ToPILImage()(img.cpu()) for img in cropped_images]

        # 编码图像
        with torch.no_grad():
            embedding = encode(cropped_images_pil)
        
        image_embeddings.append(embedding)

    image_embeddings = np.array(image_embeddings)    
    np.save(f"embeddings/{dataset.replace('-', '_')}_corpus_embeddings_{height_divisor}x{width_divisor}.npy", image_embeddings)
    print(f'File has been saved as embeddings/{dataset.replace("-", "_")}_corpus_embeddings_{height_divisor}x{width_divisor}.npy')