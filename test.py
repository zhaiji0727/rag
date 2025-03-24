from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from FlagEmbedding import BGEM3FlagModel
import gzip
import os
from shutil import move
import concurrent.futures
import time
from datetime import timedelta
import traceback
import io
from lxml import etree
import threading
import json

_bgem_semaphore = threading.BoundedSemaphore(1)

# Initialize the embedding model
embed_model = BGEM3FlagModel('/mnt/data/haoquan/model/bge-m3')

# Connect to Elasticsearch
es = Elasticsearch("http://109.105.34.64:9200")

# Ensure that the index does not exist already
try:
    if not es.indices.exists(index="pubmed25_with_vector"):
        resp = es.indices.create(
            index="pubmed25_with_vector",
            body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 1},
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "analyzer": "english"},
                        "abstract": {"type": "text", "analyzer": "english"},
                        "url": {"type": "keyword"},
                        "title_vector": {"type": "dense_vector", "dims": 1024},
                        "abstract_vector": {"type": "dense_vector", "dims": 1024},
                    }
                },
            },
        )
        print(resp)
except Exception as e:
    print(f"Error creating index: {e}")
    print(traceback.format_exc())  # Print the full stack trace

def encode_texts(texts):
    with _bgem_semaphore:
        return embed_model.encode(texts, return_dense=True, return_colbert_vecs=False, return_sparse=False)['dense_vecs']

def parse_xml_content(xml_content):
    with io.BytesIO(xml_content) as f:
        tree = etree.parse(f)
        root = tree.getroot()
        if root.tag != 'PubmedArticleSet':
            raise ValueError("Root element is not 'PubmedArticleSet'")

        result = []
        for child in root:
            entry = {"title": None, "abstract": None, "pmid": None, "url": None}
            if child.tag == 'PubmedArticle' or child.tag == 'PubmedBookArticle':
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
                        entry["title"] = (book_document.findtext('ArticleTitle') or
                                          book_document.findtext('VernacularTitle') or
                                          book_document.find('Book').findtext('BookTitle') if book_document.find('Book') else None)
                        abstract_elem = book_document.find('Abstract')
                        if abstract_elem is not None:
                            abstract_texts = abstract_elem.findall('.//AbstractText')
                            entry["abstract"] = ' '.join([text.text for text in abstract_texts if text.text])
            elif child.tag == 'DeleteCitation':
                continue  # Skip DeleteCitation elements
            if entry["pmid"]:  # Add only if PMID is present
                result.append(entry)
            else:
                raise ValueError("Element has no pmid!")
        root.clear()
        return result

def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def process_file(file_path, processed_directory, temp_directory):
    try:
        with gzip.open(file_path, 'rb') as f:
            file_content = f.read()
            articles = parse_xml_content(file_content)
            titles = [article["title"] for article in articles if article["title"]]
            abstracts = [article["abstract"] for article in articles if article["abstract"]]

            title_vectors = encode_texts(titles) if titles else []
            abstract_vectors = encode_texts(abstracts) if abstracts else []

            if len(title_vectors) != len(titles):
                raise ValueError(f"Mismatch between number of titles({len(titles)}) and title vectors({len(title_vectors)}).")
            if len(abstract_vectors) != len(abstracts):
                raise ValueError(f"Mismatch between number of abstracts({len(abstracts)}) and abstract vectors({len(abstract_vectors)}).")

            title_idx = 0
            abstract_idx = 0
            for article in articles:
                if article["title"]:
                    if title_idx < len(title_vectors):
                        article["title_vector"] = title_vectors[title_idx]
                        title_idx += 1
                    else:
                        raise IndexError("Title vector index out of range.")
                if article["abstract"]:
                    if abstract_idx < len(abstract_vectors):
                        article["abstract_vector"] = abstract_vectors[abstract_idx]
                        abstract_idx += 1
                    else:
                        raise IndexError("Abstract vector index out of range.")

            # Write prepared articles to a temporary file
            temp_file_path = os.path.join(temp_directory, os.path.basename(file_path) + ".json")
            with open(temp_file_path, 'w') as temp_file:
                json.dump(articles, temp_file)

        # Move processed file to the processed directory
        processed_file_path = os.path.join(processed_directory, os.path.basename(file_path))
        move(file_path, processed_file_path)
        print(f"Moved processed file to {processed_file_path} and saved to {temp_file_path}")

    except (OSError, IOError) as e:
        print(f"Error reading file {file_path}: {e}")
        print(traceback.format_exc())  # Print the full stack trace
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        print(traceback.format_exc())  # Print the full stack trace

def bulk_upload(temp_file_path):
    try:
        with open(temp_file_path, 'r') as temp_file:
            articles = json.load(temp_file)

        actions = [{"_index": "pubmed25_with_vector", "_id": article["pmid"], "_source": article} for article in articles]
        for chunk in chunker(actions, 50):
            bulk(es, list(chunk))
        print(f"Bulk uploaded articles from {temp_file_path}")

    except Exception as e:
        print(f"Error in bulk uploading from {temp_file_path}: {e}")
        print(traceback.format_exc())  # Print the full stack trace

def index_directory(directory_path, processed_directory, temp_directory):
    start_time = time.time()
    file_paths = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path) if file_name.endswith(".xml.gz")]
    total_files = len(file_paths)
    processed_files = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file_path, processed_directory, temp_directory): file_path for file_path in file_paths}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()
                processed_files += 1
                elapsed_time = time.time() - start_time
                progress = (processed_files / total_files) * 100
                avg_time_per_file = elapsed_time / processed_files
                remaining_time = avg_time_per_file * (total_files - processed_files)
                print(f"Completed processing {os.path.basename(file_path)}. Progress: {processed_files}/{total_files} ({progress:.2f}%). Estimated time left: {timedelta(seconds=remaining_time)}")
            except Exception as e:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        temp_file_paths = [os.path.join(temp_directory, file_name + ".json") for file_name in os.listdir(temp_directory) if file_name.endswith(".json")]
        future_to_temp_file = {executor.submit(bulk_upload, temp_file_path): temp_file_path for temp_file_path in temp_file_paths}

        for future in concurrent.futures.as_completed(future_to_temp_file):
            temp_file_path = future_to_temp_file[future]
            try:
                future.result()
                print(f"Completed bulk upload from {temp_file_path}")
            except Exception as e:
                print(f"Error in bulk upload from file {temp_file_path}: {e}")

    total_time = timedelta(seconds=time.time() - start_time)
    print(f"Indexing complete. Total time: {total_time}")

def main():
    pubmed_dir = "/mnt/data/haoquan/pubmed"
    processed_dir = "/mnt/data/haoquan/pubmed/processed"
    temp_dir = "/mnt/data/haoquan/pubmed/temp"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        index_directory(pubmed_dir, processed_dir, temp_dir)
        print("Indexing complete.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        print(traceback.format_exc())  # Print the full stack trace

if __name__ == "__main__":
    main()