import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
import nltk
from nltk.tokenize import sent_tokenize

nltk.data.path.append("/home/samsung/haoquan/nltk_data")
import json
import pickle
import heapq
import numpy as np
import faiss
import ollama
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel
import threading

_bgem_semaphore = threading.BoundedSemaphore(1)


class ColorFormatter(logging.Formatter):
    """
    自定义日志格式，为不同日志级别添加 ANSI 颜色
    """

    COLOR_CODES = {
        logging.DEBUG: "\033[34m",  # 蓝色
        logging.INFO: "\033[32m",  # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",  # 红色
        logging.CRITICAL: "\033[1;31m",  # 粗体红
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, self.RESET_CODE)
        formatted_message = super(ColorFormatter, self).format(record)
        return f"{color}{formatted_message}{self.RESET_CODE}"


class LoggerManager:
    def __init__(self, log_file_path):
        self.console_logger = self.setup_console_logger()
        self.file_logger = self.setup_file_logger(log_file_path)

    def setup_console_logger(self):
        """
        Setup console logger with ColorFormatter and StreamHandler.
        """
        logger = logging.getLogger("console_logger")
        logger.setLevel(logging.DEBUG)  # 根据需要调整日志级别

        # StreamHandler for console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(console_handler)

        return logger

    def setup_file_logger(self, log_file_path):
        """
        Setup file logger with FileHandler.
        """
        logger = logging.getLogger("file_logger")
        logger.setLevel(logging.DEBUG)  # 根据需要调整日志级别

        # FileHandler for file output
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(file_handler)

        return logger


def timer(func):
    """
    Decorator to measure and print the function's execution time
    in hours, minutes, and seconds.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        console_logger.debug(
            f"{func.__name__} executed in {int(hours)}h {int(minutes)}m {seconds:.2f}s"
        )
        file_logger.debug(
            f"{func.__name__} executed in {int(hours)}h {int(minutes)}m {seconds:.2f}s"
        )
        return result

    return wrapper


def load_index_and_pmids(index_file, pmids_file, device="cpu", gpu_device=0):
    """
    Load a FAISS index from the given file and the corresponding PMIDs.

    Parameters:
        index_file (str): Path to the FAISS index file.
        pmids_file (str): Path to the pickle file containing PMIDs.
        device (str): 'cpu' or 'gpu'. If 'gpu' is specified, the index is transferred to GPU.
        gpu_device (int): GPU device ID to use if device is 'gpu'. Default is 0.

    Returns:
        tuple: (index, pmids) where index is a FAISS index and pmids is a list loaded from the pickle.
    """
    # cpu_index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
    cpu_index = faiss.read_index(index_file)
    if device.lower() == "gpu":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_device, cpu_index)
    else:
        index = cpu_index

    with open(pmids_file, "rb") as f:
        pmids = pickle.load(f)
    return index, pmids


def embed_text(text, embedding_method="ollama", model_instance=None):
    """
    Unified function to encode input text using the specified embedding method.
    When using 'bgem', the requests are queued to allow at most 2 concurrent calls.

    Parameters:
        text (str or list of str): The input text(s) to be encoded.
        embedding_method (str): 'ollama' or 'bgem'. Default is 'ollama'.
        model_instance: Required if embedding_method is 'bgem'.

    Returns:
        numpy.ndarray: The encoded vectors as a float32 numpy array.
    """
    # Ensure text is a list.
    if isinstance(text, str):
        text = [text]

    if embedding_method.lower() == "ollama":
        response = ollama.embed(model="bge-m3", input=text)
        return np.array(response["embeddings"], dtype="float32")
    elif embedding_method.lower() == "bgem":
        if model_instance is None:
            raise ValueError(
                "BGEM model instance must be provided for 'bgem' embedding method."
            )
        with _bgem_semaphore:
            result = model_instance.encode(text, return_dense=True)
        return np.array(result["dense_vecs"], dtype="float32")
    else:
        raise ValueError(
            "Unsupported embedding method. Choose either 'bgem' or 'ollama'."
        )


def search_in_index(index, query_vector, k=10):
    """
    Search the given FAISS index for k nearest neighbors to the query vector.

    Parameters:
        index: FAISS index.
        query_vector (numpy.ndarray): Encoded query vector.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        tuple: (distances, indices) for the k nearest neighbors.
    """
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def process_index(
    index_file, index_dir, query_vector, device="cpu", current_index=None, total=None
):
    """
    Processes a single index file: loads the FAISS index and PMIDs, performs search,
    and returns a list of (distance, pmid) tuples.

    Parameters:
        index_file (str): Filename of the FAISS index.
        index_dir (str): Directory containing the index files.
        query_vector (numpy.ndarray): Encoded query vector.
        device (str): 'cpu' or 'gpu'.
        current_index (int, optional): The current index number in processing, used for logging progress.
        total (int, optional): The total number of index files being processed.

    Returns:
        list: List of tuples in the form (distance, pmid).
    """
    prefix = index_file.split("_faiss_index.index")[0]
    pmids_file = prefix + "_document_pmids.pkl"
    index_file_path = os.path.join(index_dir, index_file)

    # 如果提供了当前进度信息，则显示当前/总的
    if current_index is not None and total is not None:
        console_logger.info(
            f"Processing index {current_index}/{total}: {index_file_path}"
        )
    else:
        console_logger.info(f"Processing index: {index_file_path}")

    index, pmids = load_index_and_pmids(
        index_file_path, os.path.join(index_dir, pmids_file), device=device
    )
    distances, indices = search_in_index(index, query_vector, k=10)
    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(pmids):
            continue
        results.append((dist, pmids[idx]))
    return results


def run_es_query(query, es_client, es_index=["pubmed25"]):
    """
    Run an Elasticsearch query on the specified index using an existing client.

    Parameters:
        query (dict or str): The Elasticsearch query (dict or JSON string).
        es_client (Elasticsearch): The initialized Elasticsearch client.
        es_index (list): List of indices to query.

    Returns:
        list: A list of result dictionaries containing 'id', 'title', and 'abstract'.
    """
    if isinstance(query, str) and not isinstance(query, dict):
        query_dict = json.loads(query)
    else:
        query_dict = query

    console_logger.info("\nRunning ES query:")
    file_logger.info("\nRunning ES query:")
    console_logger.info(query_dict)
    file_logger.info(query_dict)
    response = es_client.search(index=es_index, body=query_dict)

    results = []
    if response.get("hits", {}).get("hits"):
        for hit in response["hits"]["hits"]:
            result = {
                "id": "http://www.ncbi.nlm.nih.gov/pubmed/" + str(hit["_id"]),
                "title": hit["_source"].get("title", "No title available"),
                "abstract": hit["_source"].get("abstract", "No abstract available"),
            }
            results.append(result)
    console_logger.info(f"Docs found: {len(results)}")
    file_logger.info(f"Docs found: {len(results)}")
    return results


# @timer
# def run_faiss_search(index_files, index_dir, query_vector, max_workers=4):
#     """
#     Perform the FAISS search and return the top 10 similar PMIDs.

#     Parameters:
#         index_files (list): List of FAISS index files.
#         index_dir (str): Directory containing FAISS index files.
#         query_vector (numpy.ndarray): Encoded query vector.
#         max_workers (int): Maximum number of concurrent workers.

#     Returns:
#         list: Top 10 candidate tuples in the form (distance, pmid).
#     """
#     # Maintain top candidates using a max-heap (by negative distance)
#     candidates_heap = []  # Each element: (-distance, (distance, pmid))
#     max_candidates = 20

#     def process_and_update_heap(
#         index_file, current_index, total, index_dir, query_vector
#     ):
#         """
#         Process a single index file and update the local heap with top candidates.

#         Parameters:
#             index_file (str): The FAISS index file to process.
#             current_index (int): The current index number in processing, used for logging progress.
#             total (int): The total number of index files being processed.
#             index_dir (str): Directory containing the index files.
#             query_vector (numpy.ndarray): The encoded query vector used for searching the FAISS index.

#         Returns:
#             list: A local heap of top candidates represented as tuples (-distance, (distance, pmid)).
#         """
#         results = process_index(
#             index_file, index_dir, query_vector, "cpu", current_index, total
#         )
#         local_heap = []
#         for candidate in results:
#             if len(local_heap) < max_candidates:
#                 heapq.heappush(local_heap, (-candidate[0], candidate))
#             else:
#                 if -local_heap[0][0] > candidate[0]:
#                     heapq.heappushpop(local_heap, (-candidate[0], candidate))
#         return local_heap

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         total = len(index_files)
#         for i, index_file in enumerate(index_files, start=1):
#             futures.append(
#                 executor.submit(process_and_update_heap, index_file, i, total, index_dir, query_vector)
#             )

#         for future in futures:
#             local_heap = future.result()
#             for item in local_heap:
#                 if len(candidates_heap) < max_candidates:
#                     heapq.heappush(candidates_heap, item)
#                 else:
#                     if -candidates_heap[0][0] > item[0]:
#                         heapq.heappushpop(candidates_heap, item)

#     # Retrieve and sort candidates in ascending order (lowest distance first)
#     top_candidates = [item[1] for item in candidates_heap]
#     top_candidates = sorted(top_candidates, key=lambda x: x[0])
#     top10 = top_candidates[:10]

#     console_logger.info("\nGlobal top 10 results from FAISS search:")
#     file_logger.info("\nGlobal top 10 results from FAISS search:")
#     for rank, (dist, pmid) in enumerate(top10, start=1):
#         console_logger.info(f"Rank {rank}: PMID: {pmid}, Distance: {dist}")
#         file_logger.info(f"Rank {rank}: PMID: {pmid}, Distance: {dist}")

#     return top10


@timer
def run_faiss_search(index_files, index_dir, query_vector):
    """
    Perform the FAISS search and return the top 10 similar PMIDs without parallelization.

    Parameters:
        index_files (list): List of FAISS index files.
        index_dir (str): Directory containing FAISS index files.
        query_vector (numpy.ndarray): Encoded query vector.
        max_workers (int): Maximum number of concurrent workers (unused in this sequential version).

    Returns:
        list: Top 10 candidate tuples in the form (distance, pmid).
    """
    # Maintain top candidates using a max-heap (by negative distance)
    candidates_heap = []  # Each element: (-distance, (distance, pmid))
    max_candidates = 10

    def process_and_update_heap(
        index_file, current_index, total, index_dir, query_vector
    ):
        """
        Process a single index file and update the local heap with top candidates.

        Parameters:
            index_file (str): The FAISS index file to process.
            current_index (int): The current index number in processing, used for logging progress.
            total (int): The total number of index files being processed.
            index_dir (str): Directory containing the index files.
            query_vector (numpy.ndarray): The encoded query vector used for searching the FAISS index.

        Returns:
            list: A local heap of top candidates represented as tuples (-distance, (distance, pmid)).
        """
        results = process_index(
            index_file, index_dir, query_vector, "cpu", current_index, total
        )
        local_heap = []
        for candidate in results:
            if len(local_heap) < max_candidates:
                heapq.heappush(local_heap, (-candidate[0], candidate))
            else:
                if -local_heap[0][0] > candidate[0]:
                    heapq.heappushpop(local_heap, (-candidate[0], candidate))
        return local_heap

    total = len(index_files)
    for i, index_file in enumerate(index_files, start=1):
        local_heap = process_and_update_heap(
            index_file, i, total, index_dir, query_vector
        )
        for item in local_heap:
            if len(candidates_heap) < max_candidates:
                heapq.heappush(candidates_heap, item)
            else:
                if -candidates_heap[0][0] > item[0]:
                    heapq.heappushpop(candidates_heap, item)

    # Retrieve and sort candidates in ascending order (lowest distance first)
    top_candidates = [item[1] for item in candidates_heap]
    top_candidates = sorted(top_candidates, key=lambda x: x[0])
    top10 = top_candidates[:10]

    console_logger.info("\nGlobal top 10 results from FAISS search:")
    file_logger.info("\nGlobal top 10 results from FAISS search:")
    for rank, (dist, pmid) in enumerate(top10, start=1):
        console_logger.info(f"Rank {rank}: PMID: {pmid}, Distance: {dist}")
        file_logger.info(f"Rank {rank}: PMID: {pmid}, Distance: {dist}")

    return top10


@timer
def run_es_queries(top10_candidates, es_client, es_index=["pubmed25"]):
    """
    For each PMID in the top 10 candidates, perform an Elasticsearch query to retrieve document details.

    Parameters:
        top10_candidates (list): List of (distance, pmid) tuples.
        es_client (Elasticsearch): The initialized Elasticsearch client.
        es_index (list): List of ES indices.

    Returns:
        list: Aggregated Elasticsearch results.
    """
    es_results = []
    for dist, pmid in top10_candidates:
        es_query = {"query": {"term": {"_id": pmid}}}
        console_logger.info(f"\nQuerying Elasticsearch for PMID: {pmid}")
        file_logger.info(f"\nQuerying Elasticsearch for PMID: {pmid}")
        res = run_es_query(es_query, es_client, es_index=es_index)
        es_results.extend(res)

    console_logger.info("\nFinal Elasticsearch results:")
    # file_logger.info("\nFinal Elasticsearch results:")
    for doc in es_results:
        console_logger.info(f"ID: {doc['id']}")
        # file_logger.info(f"ID: {doc['id']}")
        console_logger.info(f"Title: {doc['title']}")
        # file_logger.info(f"Title: {doc['title']}")
        console_logger.info(f"Abstract: {doc['abstract']}\n")
        # file_logger.info(f"Abstract: {doc['abstract']}\n")

    return es_results


def segment_text(text):
    """
    Segments the provided text into sentences using punctuation as delimiters.
    A simple segmentation using regex; can be replaced with advanced NLP methods if needed.

    Parameters:
        text (str): The text to segment.

    Returns:
        list: List of segmented sentences.
    """
    return sent_tokenize(text)


# @timer
# def get_relevant_snippets(
#     query_vector,
#     documents,
#     embedding_method="ollama",
#     model_instance=None,
#     max_snippets=10,
# ):
#     """
#     Extract up to max_snippets relevant text snippets by combining the title and segmented abstract sentences.
#     Each snippet is encoded and compared with the query, and the top snippets (by similarity) are retained.

#     Each snippet is represented as:
#         {
#             "document": <document URL>,
#             "beginSection": <"title" or "abstract">,
#             "offsetInBeginSection": <start character index>,
#             "endSection": <"title" or "abstract">,
#             "offsetInEndSection": <end character index>,
#             "text": <extracted snippet text>
#         }

#     Parameters:
#         query_vector (numpy.ndarray): Encoded query vector.
#         documents (list): List of documents (each with 'id', 'title', 'abstract').
#         embedding_method (str): 'ollama' or 'bgem' for encoding.
#         model_instance: Required if embedding_method is 'bgem'.
#         max_snippets (int): Maximum number of snippets to retain.

#     Returns:
#         list: List of top snippets ordered by decreasing similarity.
#     """
#     all_snippets = []

#     for doc in documents:
#         # Process the title as a single snippet if available
#         title = doc.get("title", "")
#         if title and title != "No title available":
#             console_logger.info(f"Processing title for document {doc['id']}")
#             # file_logger.info(f"Processing title for document {doc['id']}")
#             snippet_vector = embed_text(title, embedding_method, model_instance)
#             sim = cosine_similarity(query_vector, snippet_vector)
#             snippet = {
#                 "document": doc["id"],
#                 "beginSection": "title",
#                 "offsetInBeginSection": 0,
#                 "endSection": "title",
#                 "offsetInEndSection": len(title) - 1,
#                 "text": title,
#                 "similarity": sim,
#             }
#             all_snippets.append(snippet)

#         # Process the abstract by segmenting into sentences
#         abstract = doc.get("abstract", "")
#         if abstract and abstract != "No abstract available":
#             sentences = segment_text(abstract)
#             start_index = 0
#             for sent in sentences:
#                 sent = sent.strip()
#                 if not sent:
#                     continue
#                 # Determine the starting index of the sentence in the abstract
#                 idx = abstract.find(sent, start_index)
#                 if idx == -1:
#                     idx = start_index
#                 console_logger.info(
#                     f"Processing sentence: '{sent}' in document {doc['id']}"
#                 )
#                 # file_logger.info(
#                 #     f"Processing sentence: '{sent}' in document {doc['id']}"
#                 # )
#                 snippet_vector = embed_text(sent, embedding_method, model_instance)
#                 sim = cosine_similarity(query_vector, snippet_vector)
#                 snippet = {
#                     "document": doc["id"],
#                     "beginSection": "abstract",
#                     "offsetInBeginSection": idx,
#                     "endSection": "abstract",
#                     "offsetInEndSection": idx + len(sent) - 1,
#                     "text": sent,
#                     "similarity": sim,
#                 }
#                 all_snippets.append(snippet)
#                 start_index = idx + len(sent)

#     # Sort snippets by decreasing similarity (higher similarity first)
#     sorted_snippets = sorted(all_snippets, key=lambda x: x["similarity"], reverse=True)

#     # Retain only the top max_snippets and remove the similarity score from final output
#     top_snippets = []
#     for snippet in sorted_snippets[:max_snippets]:
#         snippet.pop("similarity", None)
#         top_snippets.append(snippet)

#     console_logger.info(f"Extracted {len(top_snippets)} relevant snippets.")
#     file_logger.info(f"Extracted {len(top_snippets)} relevant snippets.")

#     return top_snippets


@timer
def get_relevant_snippets(
    query_vector,
    documents,
    embedding_method="ollama",
    model_instance=None,
    max_snippets=10,
):
    """
    Extract up to max_snippets relevant text snippets by combining the title and segmented abstract sentences.
    This version uses embed_text directly to obtain embeddings for all candidate texts in batch.

    Each snippet is represented as:
        {
            "document": <document URL>,
            "beginSection": <"title" or "abstract">,
            "offsetInBeginSection": <start character index>,
            "endSection": <"title" or "abstract">,
            "offsetInEndSection": <end character index>,
            "text": <extracted snippet text>
        }

    Parameters:
        query_vector (numpy.ndarray): Encoded query vector.
        documents (list): List of documents (each with 'id', 'title', 'abstract').
        embedding_method (str): 'ollama' or 'bgem' for encoding.
        model_instance: Required if embedding_method is 'bgem'.
        max_snippets (int): Maximum number of snippets to retain.

    Returns:
        list: List of top snippets ordered by decreasing similarity.
    """
    snippet_candidates = []
    # Process each document to extract candidate snippets.
    for doc in documents:
        # Process title candidate.
        title = doc.get("title", "")
        if title and title != "No title available":
            candidate = {
                "document": doc["id"],
                "beginSection": "title",
                "offsetInBeginSection": 0,
                "endSection": "title",
                "offsetInEndSection": len(title) - 1,
                "text": title,
            }
            snippet_candidates.append(candidate)
            console_logger.info(
                f"Added title candidate for document {doc['id']}: '{title}"
            )

        # Process abstract sentences.
        abstract = doc.get("abstract", "")
        if abstract and abstract != "No abstract available":
            sentences = segment_text(abstract)
            start_index = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                idx = abstract.find(sent, start_index)
                if idx == -1:
                    idx = start_index
                candidate = {
                    "document": doc["id"],
                    "beginSection": "abstract",
                    "offsetInBeginSection": idx,
                    "endSection": "abstract",
                    "offsetInEndSection": idx + len(sent) - 1,
                    "text": sent,
                }
                snippet_candidates.append(candidate)
                console_logger.info(
                    f"Added abstract sentence candidate for document {doc['id']}: '{sent}'"
                )
                start_index = idx + len(sent)

    # Collect candidate texts.
    texts = [cand["text"] for cand in snippet_candidates]

    # Use embed_text directly for batch embedding.
    embeddings = embed_text(texts, embedding_method, model_instance)

    # Compute cosine similarity for each candidate.
    # Ensure query_vector is a 1D vector.
    query_vec = np.squeeze(query_vector)

    def cosine_similarity(v1, v2):
        v1 = np.squeeze(v1)
        v2 = np.squeeze(v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    for idx, candidate in enumerate(snippet_candidates):
        sim = cosine_similarity(query_vec, embeddings[idx])
        candidate["similarity"] = sim

    sorted_candidates = sorted(
        snippet_candidates, key=lambda x: x["similarity"], reverse=True
    )

    # Retain top max_snippets and remove similarity.
    top_snippets = []
    for candidate in sorted_candidates[:max_snippets]:
        candidate.pop("similarity", None)
        top_snippets.append(candidate)

    console_logger.info(f"Extracted {len(top_snippets)} relevant snippets.")
    file_logger.info(f"Extracted {len(top_snippets)} relevant snippets.")
    return top_snippets


# def cosine_similarity(v1, v2):
#     """
#     Computes cosine similarity between two vectors.

#     Parameters:
#         v1 (numpy.ndarray): First vector.
#         v2 (numpy.ndarray): Second vector.

#     Returns:
#         float: Cosine similarity value.
#     """
#     v1 = np.squeeze(v1)
#     v2 = np.squeeze(v2)
#     norm1 = np.linalg.norm(v1)
#     norm2 = np.linalg.norm(v2)
#     if norm1 == 0 or norm2 == 0:
#         return 0.0
#     return np.dot(v1, v2) / (norm1 * norm2)


def process_query(
    question,
    embedding_method,
    index_dir,
    bgem_model,
    es_client,
    query_idx,
    total_queries,
):
    query_id = question.get("id")
    query_type = question.get("type")
    query = question.get("body")

    console_logger.info(
        f"Processing query {query_idx}/{total_queries} - ID: {query_id}, Type: {query_type}, Body: {query}"
    )
    file_logger.info(
        f"Processing query {query_idx}/{total_queries} - ID: {query_id}, Type: {query_type}, Body: {query}"
    )

    # Encode the query text using the unified encoding function
    query_vector = embed_text(query, embedding_method, bgem_model)

    # Retrieve all FAISS index files from the directory
    index_files = [f for f in os.listdir(index_dir) if f.endswith("_faiss_index.index")]

    # Module 1: FAISS search to get top 10 similar PMIDs
    top10_candidates = run_faiss_search(
        index_files, index_dir=index_dir, query_vector=query_vector
    )
    console_logger.info(f"Query {query_idx}/{total_queries} - FAISS search completed.")
    file_logger.info(f"Query {query_idx}/{total_queries} - FAISS search completed.")

    # Module 2: Elasticsearch queries for the top 10 PMIDs
    es_results = run_es_queries(top10_candidates, es_client, es_index=["pubmed25"])
    console_logger.info(
        f"Query {query_idx}/{total_queries} - Elasticsearch queries completed."
    )
    file_logger.info(
        f"Query {query_idx}/{total_queries} - Elasticsearch queries completed."
    )

    # Module 3: Extract up to 10 relevant snippets from the documents.
    snippets = get_relevant_snippets(
        query_vector=query_vector,
        documents=es_results,
        embedding_method=embedding_method,
        model_instance=bgem_model,
        max_snippets=20,
    )
    console_logger.info(
        f"Query {query_idx}/{total_queries} - Relevant snippets extraction completed."
    )
    file_logger.info(
        f"Query {query_idx}/{total_queries} - Relevant snippets extraction completed."
    )

    console_logger.info("\nExtracted Snippets:")
    file_logger.info("\nExtracted Snippets:")
    for snippet in snippets:
        console_logger.info(snippet)
        file_logger.info(snippet)

    return query, top10_candidates, snippets, query_id, query_type


@timer
def main():
    """
    Main function to run the entire process for each query in the JSON file:
    1. Perform FAISS search to get top 10 similar PMIDs.
    2. Query Elasticsearch for document details of the top 10 PMIDs.
    3. Extract relevant snippets from the documents based on the query.
    """
    input_json_filepath = "/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/BioASQ-task11bPhaseA-testset1.json"
    output_json_filepath = "BioASQ-task11bPhaseA-testset1_run_file_snippets_20.json"
    # output_json_filepath = 'test.json'

    # Determine log file path based on output_json_filepath
    log_file_path = os.path.splitext(output_json_filepath)[0] + ".log"

    # Initialize logger
    logger_manager = LoggerManager(log_file_path)
    global console_logger, file_logger
    console_logger = logger_manager.console_logger
    file_logger = logger_manager.file_logger

    # Load queries from JSON file
    with open(input_json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    total_queries = len(questions)

    # Module initialization
    embedding_method = "bgem"  # Choose "ollama" or "bgem"
    index_dir = "/mnt/data/haoquan/pubmed/faiss_index_data"
    # index_dir = "/mnt/data/haoquan/pubmed/test"
    bgem_model = (
        BGEM3FlagModel("/mnt/data/haoquan/model/bge-m3")
        if embedding_method.lower() == "bgem"
        else None
    )

    # Initialize Elasticsearch client
    es_client = Elasticsearch("http://109.105.34.64:9200")

    results = []
    with ThreadPoolExecutor(max_workers=85) as executor:
        future_to_question = {
            executor.submit(
                process_query,
                question,
                embedding_method,
                index_dir,
                bgem_model,
                es_client,
                query_idx + 1,
                total_queries,
            ): question
            for query_idx, question in enumerate(questions)
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                query, top10_candidates, snippets, query_id, query_type = (
                    future.result()
                )
                results.append(
                    (query, top10_candidates, snippets, query_id, query_type)
                )
            except Exception as exc:
                console_logger.error(
                    f"Query ID {question.get('id')} generated an exception: {exc}"
                )
                file_logger.error(
                    f"Query ID {question.get('id')} generated an exception: {exc}"
                )

    # Collect snippets and documents for final JSON output
    all_questions = []

    for query, top10_candidates, snippets, query_id, query_type in results:
        question_dict = {
            "id": query_id,
            "type": query_type,
            "body": query,
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/" + str(pmid)
                for _, pmid in top10_candidates
            ],
            "snippets": snippets,
            "exact_answer": [[]] if query_type in ["list", "factoid"] else "",
            "ideal_answer": "",
        }
        all_questions.append(question_dict)

    final_json_structure = {"questions": all_questions}

    # Write the final JSON structure to a file
    with open(output_json_filepath, "w", encoding="utf-8") as outfile:
        json.dump(final_json_structure, outfile, ensure_ascii=False, indent=4)

    console_logger.info(f"JSON file written to {output_json_filepath}.")
    file_logger.info(f"JSON file written to {output_json_filepath}.")

    console_logger.info(f"Log file written to {log_file_path}.")
    file_logger.info(f"Log file written to {log_file_path}.")


if __name__ == "__main__":
    main()
