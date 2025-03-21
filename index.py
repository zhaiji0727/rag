#!/usr/bin/env python3
"""
优化后的 PubMed 文章处理脚本（适用于大批量文章，如30000篇）

该版本使用 asyncio 及异步调用包装 BGEM3FlagModel.encode 实现嵌入请求，
并使用 asyncio.to_thread 包装文件解析和存储操作。
另外，在处理每个文件时，通过设置 max_parallel_batches 参数来控制每个文件内部并发批次的数量，
以限制同时执行的嵌入请求。
"""

import os
import sys
import gzip
import io
import pickle
import numpy as np
import asyncio
import logging
import queue
from shutil import move
import urllib3
from lxml import etree
import faiss

# 导入 BGEM3FlagModel
from FlagEmbedding import BGEM3FlagModel

# Disable SSL warnings (optional)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ColorFormatter(logging.Formatter):
    """
    自定义日志格式，为不同日志级别添加 ANSI 颜色
    """
    COLOR_CODES = {
        logging.DEBUG: "\033[34m",    # 蓝色
        logging.INFO: "\033[32m",     # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",    # 红色
        logging.CRITICAL: "\033[1;31m",  # 粗体红
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, self.RESET_CODE)
        formatted_message = super(ColorFormatter, self).format(record)
        return f"{color}{formatted_message}{self.RESET_CODE}"

# 配置日志记录
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 根据需要调整日志级别
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = ColorFormatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)

# 全局 worker pool（当前并未使用模型复制，但保留供未来扩展）
worker_pool = None

def initialize_worker_pool(max_workers: int):
    global worker_pool
    worker_pool = queue.Queue(max_workers)
    for i in range(1, max_workers + 1):
        worker_pool.put(i)
    logger.info(f"Initialized worker pool with IDs: {list(range(1, max_workers + 1))}")

def parse_xml_content(xml_content):
    """
    解析 PubmedArticleSet 格式的 XML，提取文章的 title, abstract, pmid 和 url。
    """
    with io.BytesIO(xml_content) as f:
        tree = etree.parse(f)
        root = tree.getroot()
        if root.tag != 'PubmedArticleSet':
            raise ValueError("Root element is not 'PubmedArticleSet'")
        result = []
        for child in root:
            entry = {"title": None, "abstract": None, "pmid": None, "url": None}
            if child.tag in ['PubmedArticle', 'PubmedBookArticle']:
                entry["pmid"] = child.findtext('.//PMID')
                entry["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{entry['pmid']}/"
                if child.tag == 'PubmedArticle':
                    entry["title"] = child.findtext('.//ArticleTitle')
                    abstract_elem = child.find('.//Abstract')
                    if abstract_elem is not None:
                        abstract_texts = abstract_elem.findall('.//AbstractText')
                        entry["abstract"] = ' '.join([text.text for text in abstract_texts if text.text])
                elif child.tag == 'PubmedBookArticle':
                    book_document = child.find('BookDocument')
                    if book_document is not None:
                        entry["title"] = (
                            book_document.findtext('ArticleTitle') or
                            book_document.findtext('VernacularTitle') or
                            (book_document.find('Book').findtext('BookTitle') 
                             if book_document.find('Book') is not None else None)
                        )
                        abstract_elem = book_document.find('Abstract')
                        if abstract_elem is not None:
                            abstract_texts = abstract_elem.findall('.//AbstractText')
                            entry["abstract"] = ' '.join([text.text for text in abstract_texts if text.text])
            elif child.tag == 'DeleteCitation':
                continue
            if entry["pmid"]:
                result.append(entry)
            else:
                logger.warning("Element has no PMID, skipped.")
        root.clear()
        return result

def load_articles_from_file(file_path):
    """
    从 xml.gz 文件中加载并解析文章信息
    """
    try:
        with gzip.open(file_path, 'rb') as f:
            file_content = f.read()
        articles = parse_xml_content(file_content)
        return articles
    except Exception as e:
        logger.error(f"Error loading/parsing {file_path}: {e}")
        return []

def batch_generator(lst, batch_size):
    """
    将列表 lst 分批，每批大小为 batch_size
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

async def embed_batch(model, batch, idx: int, total: int):
    """
    异步请求单个批次的嵌入数据，使用 asyncio.to_thread 包装同步调用
    """
    try:
        # 同步调用 encode 方法，并在返回结果中获取 dense_vecs
        result = await asyncio.to_thread(model.encode, batch, return_dense=True)
        embeddings = np.array(result['dense_vecs'], dtype='float32')
        logger.info(f"Batch {idx}/{total} processed, got {embeddings.shape[0]} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error in batch {idx}: {e}")
        raise

async def bulk_get_embeddings_batched(texts, model, batch_size, max_parallel_batches):
    """
    分批调用 BGEM3FlagModel 的 encode 接口，对文本列表进行嵌入处理
    使用 semaphore 限制每个文件内部同时执行的嵌入批次数量
    返回所有嵌入结果拼接成的 numpy 数组
    """
    batches = list(batch_generator(texts, batch_size))
    total_batches = len(batches)
    logger.info(f"Total texts: {len(texts)}, Processing in {total_batches} batches (batch size = {batch_size})")
    
    semaphore = asyncio.Semaphore(max_parallel_batches)
    
    async def sem_embed_batch(batch, idx):
        async with semaphore:
            return await embed_batch(model, batch, idx, total_batches)
    
    tasks = [
        asyncio.create_task(sem_embed_batch(batch, idx))
        for idx, batch in enumerate(batches, start=1)
    ]
    results = await asyncio.gather(*tasks)
    return np.vstack(results)

async def process_single_file_async(file_path, model):
    """
    异步处理单个 xml.gz 文件：
    解析文章，分批获取嵌入，并构建 FAISS 索引。
    文件解析和保存操作通过 asyncio.to_thread 包装为阻塞任务。
    """
    logger.info(f"Start processing file: {os.path.basename(file_path)}")
    
    articles = await asyncio.to_thread(load_articles_from_file, file_path)
    if not articles:
        logger.warning(f"No articles found or failed to parse {os.path.basename(file_path)}")
        return None

    texts = []
    pmids = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('abstract', '')}".strip()
        if text:
            texts.append(text)
            pmids.append(article["pmid"])
    if not texts:
        logger.warning(f"No valid texts to embed for {os.path.basename(file_path)}")
        return None

    try:
        # 设置 max_parallel_batches 控制每个文件内部同时处理的嵌入批次数量
        embeddings_array = await bulk_get_embeddings_batched(texts, model, batch_size=1000, max_parallel_batches=2)
    except Exception as e:
        logger.error(f"Batch embedding failed for {os.path.basename(file_path)}: {e}")
        return None

    try:
        dimension = embeddings_array.shape[1]
    except IndexError:
        logger.error(f"Embedding array shape error in {os.path.basename(file_path)}")
        return None

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    logger.info(f"Finished processing file: {os.path.basename(file_path)}; Embedded {len(pmids)} articles.")
    return {
        "filename": os.path.basename(file_path),
        "index": index,
        "pmids": pmids,
        "texts": texts,
        "source_path": file_path   # 保存原始路径供后续移动使用
    }

def save_file_index(data, output_dir):
    """
    保存 FAISS 索引及对照的 PMIDs 到 output_dir 中
    """
    prefix = os.path.splitext(data["filename"])[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    index_file = os.path.join(output_dir, f"{prefix}_faiss_index.index")
    pmids_file = os.path.join(output_dir, f"{prefix}_document_pmids.pkl")
    
    faiss.write_index(data["index"], index_file)
    logger.info(f"FAISS index for {data['filename']} saved to: {index_file}")
    
    with open(pmids_file, "wb") as f:
        pickle.dump(data["pmids"], f)
    logger.info(f"Document PMIDs for {data['filename']} saved to: {pmids_file}")

async def process_files_concurrently_async(folder_path, model, output_dir, max_files):
    """
    异步并发处理文件夹中所有的 .xml.gz 文件，但每一批最多处理 max_files 个文件
    """
    all_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.xml.gz')]
    total_files = len(all_files)
    logger.info(f"Found {total_files} files to process.")

    # 将所有文件分成批次，每批处理 max_files 个文件
    file_batches = list(batch_generator(all_files, max_files))

    for batch_num, files in enumerate(file_batches, start=1):
        logger.info(f"Processing batch {batch_num}/{len(file_batches)} with {len(files)} files.")
        tasks = [asyncio.create_task(process_single_file_async(file_path, model)) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
                continue
            if result:
                await asyncio.to_thread(save_file_index, result, output_dir)
                processed_dir = os.path.join(folder_path, 'processed')
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir)
                src_path = result["source_path"]
                dst_path = os.path.join(processed_dir, os.path.basename(result["filename"]))
                await asyncio.to_thread(move, src_path, dst_path)
                logger.info(f"Moved {result['filename']} to processed folder")

async def main():
    folder_path = "/mnt/data/haoquan/pubmed/"    # xml.gz 文件存放目录
    output_dir = "/mnt/data/haoquan/pubmed/faiss_index_data_training13bft/"  # 输出目录保存索引及 PMIDs
    
    # 使用 BGEM3FlagModel 初始化模型（模型文件路径根据需要调整）
    embed_model = BGEM3FlagModel('/mnt/data/haoquan/model/finetine_training13b')
    
    max_workers = 4
    max_files = 1  # 每个批次同时处理的最大文件数量

    initialize_worker_pool(max_workers)

    logger.info("Starting concurrent processing of all XML.GZ files asynchronously in batches...")
    await process_files_concurrently_async(folder_path, embed_model, output_dir, max_files)
    logger.info("Processing complete. All indexes saved and files moved.")

if __name__ == "__main__":
    asyncio.run(main())
