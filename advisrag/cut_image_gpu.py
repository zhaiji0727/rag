# %%
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO

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

SlideVQA_corpus_ds = load_dataset("./dataset/VisRAG-Ret-Test-SlideVQA", name="corpus", split="train")
image_embeddings = []

for item in tqdm(SlideVQA_corpus_ds):
    image_id = item['corpus-id']
    image = item['image']
    width, height = image.size
    # print(width, height)
    # crop_width = width // 2
    # crop_height = height // 2
    # cropped_images = [
    #     image.crop((0, 0, crop_width, crop_height)).convert('RGB'),
    #     image.crop((crop_width, 0, width, crop_height)).convert('RGB'),
    #     image.crop((0, crop_height, crop_width, height)).convert('RGB'),
    #     image.crop((crop_width, crop_height, width, height)).convert('RGB')
    # ]
    crop_width = width // 8
    crop_height = height // 8
    cropped_images = [
        image.crop((j * crop_width, i * crop_height, (j + 1) * crop_width, (i + 1) * crop_height)).convert('RGB')
        for i in range(8) for j in range(8)
    ]
    # overlap_ratio = 0.2
    # # 根据图像尺寸和重叠比例计算重叠部分的大小
    # overlap_width = int(crop_width * overlap_ratio)
    # overlap_height = int(crop_height * overlap_ratio)

    # # 将图像裁剪成指定数量的块，并在每个方向上增加重叠部分
    # cropped_images = []
    # for i in range(4):
    #     for j in range(4):
    #         left = max(j * crop_width - overlap_width, 0)
    #         upper = max(i * crop_height - overlap_height, 0)
    #         right = min((j + 1) * crop_width + overlap_width, width)
    #         lower = min((i + 1) * crop_height + overlap_height, height)
    #         cropped_images.append(image.crop((left, upper, right, lower)).convert('RGB'))
    
    # fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    # for i in range(4):
    #     for j in range(4):
    #         axes[i, j].imshow(cropped_images[i * 4 + j])
    #         axes[i, j].axis('off')  # Hide the axes
            
    # plt.show()
    
    embedding = encode(cropped_images)
    image_embeddings.append(embedding)
    
    
    # break
    
image_embeddings = np.array(image_embeddings)    
np.save(f"embeddings\\SlideVQA_corpus_embeddings_8x8.npy", image_embeddings)
    
    