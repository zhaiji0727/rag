import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import requests
from io import BytesIO
from datasets import load_dataset
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
import matplotlib.pyplot as plt

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


# Load datasets
# MP_DocVQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-MP-DocVQA", name="corpus", split="train")
# MP_DocVQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-MP-DocVQA", name="queries", split="train")

# ArxivQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-ArxivQA", name="corpus", split="train")
# ArxivQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-ArxivQA", name="queries", split="train")

# ChartQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-ChartQA", name="corpus", split="train")
# ChartQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-ChartQA", name="queries", split="train")

# InfoVQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-InfoVQA", name="corpus", split="train")
# InfoVQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-InfoVQA", name="queries", split="train")

PlotQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-PlotQA", name="corpus", split="train")
PlotQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-PlotQA", name="queries", split="train")

# SlideVQA_corpus_ds = load_dataset("dataset/VisRAG-Ret-Test-SlideVQA", name="corpus", split="train")
# SlideVQA_queries_ds = load_dataset("dataset/VisRAG-Ret-Test-SlideVQA", name="queries", split="train")

def remove_color(image):
    
    # 将图像转换为RGBA模式
    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        # 计算像素的饱和度
        max_val = max(item[:3])
        min_val = min(item[:3])
        saturation = max_val - min_val
        
        # 如果饱和度较低且明度低于阈值，将像素转换为黑色
        brightness_threshold = 200  # 明度阈值
        if saturation < 30 and max_val < brightness_threshold:  # 设定饱和度和明度阈值
            new_data.append((0, 0, 0, item[3]))
        else:
            new_data.append(item)
    image.putdata(new_data)
    image = image.convert("RGBA")
    data = image.getdata()

    threshold = 110  # 设定阈值

    new_data = []
    for item in data:
        # 检查像素是否为非黑色和灰色
        if max(item[:3]) > threshold:
            # 将非黑色和灰色像素替换为白色
            new_data.append((255, 255, 255, item[3]))
        else:
            new_data.append(item)

    image.putdata(new_data)
    return image.convert("RGB")

def encode_and_save_embeddings(dataset, dataset_name):
    image_embeddings = []
    corpus_ids = []
    # total_examples = len(dataset)  

    for i, example in enumerate(tqdm(dataset)):  
        image = example['image'].convert('RGB')
        # corpus_id = example['corpus-id']
        # if i == 0:
        #     continue
        processed_image = remove_color(image)
        # plt.imshow(processed_image)
        # plt.title(f"Processed Image {i + 1}")
        # plt.show()
        # plt.close()
        
        # output_dir = "./image_plus_summary/PlotQA/grayscale/"
        # os.makedirs(output_dir, exist_ok=True)
        
        # base_name, ext = os.path.splitext(corpus_id)
        # # 生成新的文件名
        # file_name = f"{i}{ext}"
        # # 保存图像
        # processed_image.save(os.path.join(output_dir, file_name))
        
        with torch.no_grad():
            embedding = encode([processed_image])
        image_embeddings.append(embedding)
        # corpus_ids.append(example['corpus-id'])
        
        # if (i + 1) % 10 == 0 or (i + 1) == total_examples:  # 每处理10个样本或最后一个样本时输出进度  
        #     progress = (i + 1) / total_examples * 100  
        #     print(f"Processing {dataset_name}: {i + 1}/{total_examples} ({progress:.2f}%)")  

    # 将嵌入列表转换为numpy数组
    image_embeddings = np.vstack(image_embeddings)

    # 保存嵌入和corpus-id到文件
    np.save(f"embeddings/{dataset_name}_embeddings_grayscale.npy", image_embeddings)
    # np.save(f"embeddings/{dataset_name}_corpus_ids.npy", np.array(corpus_ids))

# encode_and_save_embeddings(MP_DocVQA_corpus_ds, "MP_DocVQA_corpus")
# encode_and_save_embeddings(SlideVQA_corpus_ds, "SlideVQA_corpus")
encode_and_save_embeddings(PlotQA_corpus_ds, "PlotQA_corpus")
# encode_and_save_embeddings(ChartQA_corpus_ds, "ChartQA_corpus")
# encode_and_save_embeddings(InfoVQA_corpus_ds, "InfoVQA_corpus")
# encode_and_save_embeddings(ArxivQA_corpus_ds, "ArxivQA_corpus")
# import concurrent.futures

# # Define the datasets and their corresponding names
# datasets = [
#     (ArxivQA_corpus_ds, "ArxivQA_corpus"),
#     (ChartQA_corpus_ds, "ChartQA_corpus"),
#     (InfoVQA_corpus_ds, "InfoVQA_corpus"),
#     (PlotQA_corpus_ds, "PlotQA_corpus")
# ]

# # Use ThreadPoolExecutor to execute the functions concurrently
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(encode_and_save_embeddings, ds, name) for ds, name in datasets]
    
#     # Wait for all futures to complete
#     concurrent.futures.wait(futures)

# print("All functions have been executed concurrently.")

# def add_text_below_image(image, text):
#     # 创建一个新的图像，包含原图像和文字区域
#     font = ImageFont.load_default()
#     draw = ImageDraw.Draw(image)
    
#     # 计算文字区域的高度
#     text_bbox = draw.textbbox((0, 0), text, font=font)
#     text_height = text_bbox[3] - text_bbox[1]
#     padding = 10  # 文字区域的上下边距
#     total_text_height = text_height + 2 * padding
    
#     new_image = Image.new('RGB', (image.width, image.height + total_text_height), (255, 255, 255))
    
#     # 将原图像粘贴到新图像上
#     new_image.paste(image, (0, 0))
    
#     # 在新图像的下方添加文字
#     draw = ImageDraw.Draw(new_image)
#     text_position = (10, image.height + padding)  # 文字位置，可以根据需要调整
#     draw.text(text_position, text, font=font, fill="black")
    
#     return new_image
def add_text_below_image(image, text):
    # 创建一个新的图像，包含原图像和文字区域
    font = ImageFont.load_default(size=20)
    draw = ImageDraw.Draw(image)
    
    # 计算每行文字的最大宽度
    max_width = image.width * 0.85  # 留出左右边距
    wrapped_text = textwrap.fill(text, width=max_width // (20 // 2))
    # print(wrapped_text)

    
    # 计算文字区域的高度
    lines = wrapped_text.split('\n')
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
    padding = 10  # 文字区域的上下边距
    total_text_height = text_height + 2 * padding
    
    new_image = Image.new('RGB', (image.width, image.height + total_text_height), (255, 255, 255))
    
    # 将原图像粘贴到新图像上
    new_image.paste(image, (0, 0))
    
    # 在新图像的下方添加文字
    draw = ImageDraw.Draw(new_image)
    y_text = image.height + padding
    for line in lines:
        x_text = 10
        words = line.split(' ')
        for word in words:
            for char in word:
                draw.text((x_text, y_text), char, font=font, fill="black")
                x_text += draw.textbbox((0, 0), char, font=font)[2] - draw.textbbox((0, 0), char, font=font)[0] + 1
            x_text += 10  # 增加单词之间的间距
        y_text += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
    
    return new_image

def add_text_above_image(image, text):
    # 创建一个新的图像，包含文字区域和原图像
    font = ImageFont.load_default(size=20)
    draw = ImageDraw.Draw(image)
    
    # 计算每行文字的最大宽度
    max_width = image.width * 0.85  # 留出左右边距
    wrapped_text = textwrap.fill(text, width=max_width // (20 // 2))
    
    # 计算文字区域的高度
    lines = wrapped_text.split('\n')
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
    padding = 10  # 文字区域的上下边距
    total_text_height = text_height + 2 * padding
    
    new_image = Image.new('RGB', (image.width, image.height + total_text_height), (255, 255, 255))
    
    # 将原图像粘贴到新图像的下方
    new_image.paste(image, (0, total_text_height))
    
    # 在新图像的上方添加文字
    draw = ImageDraw.Draw(new_image)
    y_text = padding
    for line in lines:
        x_text = 10
        words = line.split(' ')
        for word in words:
            for char in word:
                draw.text((x_text, y_text), char, font=font, fill="black")
                x_text += draw.textbbox((0, 0), char, font=font)[2] - draw.textbbox((0, 0), char, font=font)[0] + 1
            x_text += 10  # 增加单词之间的间距
        y_text += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
    
    return new_image

def encode_and_save_image_plus_summary_embeddings(dataset, dataset_name):
    image_embeddings = []
    corpus_ids = []
    total_examples = len(dataset)  
    
    with open("./PlotQA_image_keywords.json", 'r', encoding='utf-8') as f:
        summaries = json.load(f)

    for i, example in enumerate(tqdm(dataset)):  
        image = example['image'].convert('RGB')
        corpus_id = example['corpus-id']
        summary = summaries[corpus_id]['Description']
        title = summaries[corpus_id]['Title']
        
        # print(summary)
        
        
        new_image = add_text_below_image(image, summary)
        new_image = add_text_above_image(new_image, title)
        
        
        # output_dir = "./image_plus_summary/PlotQA/TitleAndSummary/"
        # os.makedirs(output_dir, exist_ok=True)
        # new_image.save(os.path.join(output_dir, corpus_id))
        
        # base_name, ext = os.path.splitext(corpus_id)
        # 生成新的文件名
        # file_name = f"{i}{ext}"
        # 保存图像
        # new_image.save(os.path.join(output_dir, corpus_id))
        # exit(0)
        with torch.no_grad():
            embedding = encode([new_image])
        image_embeddings.append(embedding)
        # corpus_ids.append(example['corpus-id'])
        
        # if (i + 1) % 10 == 0 or (i + 1) == total_examples:  # 每处理10个样本或最后一个样本时输出进度  
        #     progress = (i + 1) / total_examples * 100  
        #     print(f"Processing {dataset_name}: {i + 1}/{total_examples} ({progress:.2f}%)")  

    # 将嵌入列表转换为numpy数组
    image_embeddings = np.vstack(image_embeddings)

    # 保存嵌入和corpus-id到文件
    np.save(f"embeddings/{dataset_name}_image_plus_summary_plus_title_embeddings.npy", image_embeddings)
    # np.save(f"embeddings/{dataset_name}_corpus_ids.npy", np.array(corpus_ids))
    
# encode_and_save_image_plus_summary_embeddings(PlotQA_corpus_ds, "PlotQA_corpus")

def encode_and_save_query_embeddings(dataset, dataset_name):  
    query_embeddings = []
    query_ids = []  
    total_examples = len(dataset)  
  
    for i, example in enumerate(dataset):  
        query = example['query']
        embedding = encode([query])  
        query_embeddings.append(embedding)
        query_ids.append(example['query-id'])  
  
        if (i + 1) % 10 == 0 or (i + 1) == total_examples:  # 每处理10个样本或最后一个样本时输出进度  
            progress = (i + 1) / total_examples * 100  
            print(f"Processing {dataset_name}: {i + 1}/{total_examples} ({progress:.2f}%)")  
  
    # 将嵌入列表转换为numpy数组
    query_embeddings = np.vstack(query_embeddings)

    # 保存嵌入和corpus-id到文件
    np.save(f"embeddings/{dataset_name}_embeddings.npy", query_embeddings)
    np.save(f"embeddings/{dataset_name}_query_ids.npy", np.array(query_ids))
  
# encode_and_save_query_embeddings(MP_DocVQA_queries_ds, "MP_DocVQA_queries")  
# encode_and_save_query_embeddings(SlideVQA_queries_ds, "SlideVQA_queries")  
# encode_and_save_query_embeddings(ArxivQA_queries_ds, "ArxivQA_queries")  
# encode_and_save_query_embeddings(ChartQA_queries_ds, "ChartQA_queries")  
# encode_and_save_query_embeddings(InfoVQA_queries_ds, "InfoVQA_queries")  
# encode_and_save_query_embeddings(PlotQA_queries_ds, "PlotQA_queries")  
