import json
import numpy as np
from tqdm import tqdm
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

image_data = {}
with open('./PlotQA_image_keywords.json', 'r', encoding='utf-8') as f:
    image_data = json.load(f)
    
keyword_embeddings_list = []
INSTRUCTION = 'Represent these key words extracted from image for being retrieved: '

for image_id in tqdm(image_data):
    
    # title = image_data[image_id]['Title']
    keywords = image_data[image_id]['Keywords']
    # description = image_data[image_id]['Description']
    # image_type = image_data[image_id]['Image Type']
        
    # title_vector = encode([title]).tolist()
    # listt = [INSTRUCTION + ', '.join(keywords)]
    # print(listt)
    # exit(0)
    keywords_vector = encode([INSTRUCTION + ', '.join(keywords)])
    # description_vector = encode([description]).tolist()
    
    # encode_image_data[image_id] = {
    #     'title_vector': title_vector,
    #     'keywords_vector': keywords_vector,
    #     'description_vector': description_vector
    # }
    
    keyword_embeddings_list.append(keywords_vector)
    
np.save('embeddings/PlotQA_keyword_embeddings_with_instruction.npy', keyword_embeddings_list)
    
# with open('encoded_PlotQA_image_keywords.json', 'w', encoding='utf-8') as f:
#     json.dump(encode_image_data, f, ensure_ascii=False)