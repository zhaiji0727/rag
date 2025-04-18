import os
import pickle
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import urllib3
import pandas as pd
from elasticsearch import Elasticsearch
from openai import OpenAI, APIError, APIConnectionError
from unicodedata import normalize
import json
import csv
import subprocess
import argparse
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from typing import Optional
import threading
import random

_bgem_semaphore = threading.BoundedSemaphore(1)
_es_semaphore = threading.BoundedSemaphore(5)
_openai_client_semaphore = threading.BoundedSemaphore(2)

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
# Suppress warnings about Elasticsearch certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ClientRotator:
    def __init__(self, client_semaphore_pairs):
        """
        client_semaphore_pairs: 列表，元素为元组 (client, semaphore)
        """
        self.client_semaphore_pairs = client_semaphore_pairs
        self.index = 0
        self.lock = threading.Lock()  # 保证线程安全的轮询

    def get_next(self):
        """获取下一个客户端及其对应的信号量"""
        with self.lock:
            client, semaphore = self.client_semaphore_pairs[self.index]
            self.index = (self.index + 1) % len(self.client_semaphore_pairs)
            return client, semaphore


def get_chat_completion(
    model: str,
    messages: list,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
    seed: int = 90128538,
):
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_INTERVAL = 3  # 重试间隔（秒）
    for attempt in range(MAX_RETRIES + 1):
        try:
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
            }

            if response_format:
                request_params["response_format"] = response_format

            if enable_multiclient:
                current_client, current_semaphore = client_rotator.get_next()
                with current_semaphore:
                    return current_client.chat.completions.create(**request_params)
            else:
                with _openai_client_semaphore:
                    return client_openai.chat.completions.create(**request_params)

        except APIError as e:
            if e.status_code == 500 and attempt < MAX_RETRIES:
                print(f"服务器错误，第 {attempt+1} 次重试...")
                time.sleep(RETRY_INTERVAL)
            else:
                raise
        except APIConnectionError as e:
            if attempt < MAX_RETRIES:
                print(f"连接异常，第 {attempt+1} 次重试...")
                time.sleep(RETRY_INTERVAL)
            else:
                raise
        except Exception as e:
            print(e)
            raise

    raise RuntimeError(f"超过最大重试次数 {MAX_RETRIES}")


def run_elasticsearch_query(query, index="pubmed25_with_vector"):
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_INTERVAL = 3  # 执行间隔（秒）

    es = Elasticsearch(
        "http://109.105.34.64:9200", verify_certs=False, request_timeout=3000
    )

    if isinstance(query, str):
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON query string: {str(e)}")
    else:
        query_dict = query

    for attempt in range(MAX_RETRIES + 1):  # 0~MAX_RETRIES 共 MAX_RETRIES+1 次尝试
        try:
            with _es_semaphore:  # 信号量控制并发
                print("\nRunning ES query:".ljust(40, "-"))
                print(f"Query: {query_dict}\n")
                response = es.search(index=index, **query_dict)

                # 处理查询结果
                results = []
                if response["hits"]["hits"]:
                    for hit in response["hits"]["hits"]:
                        results.append(
                            {
                                "id": f"http://www.ncbi.nlm.nih.gov/pubmed/{hit['_id']}",
                                "title": hit["_source"].get(
                                    "title", "No title available"
                                ),
                                "abstract": hit["_source"].get(
                                    "abstract", "No abstract available"
                                ),
                            }
                        )
                print(f"Documents found: {len(results)}")
                time.sleep(RETRY_INTERVAL)
                return results

        except Exception as e:
            print(f"Query failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e.info)}")
            if attempt + 1 == MAX_RETRIES:  # 最后一次尝试失败后直接返回
                return []
            print("Sleeping 10s...")
            time.sleep(10)

    return []  # 冗余返回，实际不会执行到这里


"""
def create_query(
    query_string: str,
    query_vector=None,
    text_boost=1.0,
    abstract_boost=0.0,
    title_boost=0.0,
    size=20,
):
    query = {}
    bool_should = []

    # 添加query_string查询子句
    if text_boost != 0:
        query_string_clause = {
            "query_string": {"query": query_string, "boost": text_boost}
        }
        bool_should.append(query_string_clause)

    # 添加abstract_vector KNN查询子句
    if query_vector is not None and abstract_boost != 0:
        abstract_knn = {
            "knn": {
                "field": "abstract_vector",
                "query_vector": query_vector,
                # "k": 20,
                "boost": abstract_boost,
            }
        }
        bool_should.append(abstract_knn)

    # 添加title_vector KNN查询子句
    if query_vector is not None and title_boost != 0:
        title_knn = {
            "knn": {
                "field": "title_vector",
                "query_vector": query_vector,
                # "k": 20,
                "boost": title_boost,
            }
        }
        bool_should.append(title_knn)

    # 构造最终查询
    if bool_should:
        query["query"] = {"bool": {"should": bool_should}}
    elif query_vector is None:
        query["query"] = {"query_string": {"query": query_string}}

    query["size"] = size
    return query
"""


def create_query(
    query_string: str,
    query_vector=None,
    text_boost=1.0,
    abstract_boost=0.0,
    title_boost=0.0,
    size=20,
):
    query = {
        "query": {
            "script_score": {
                # 动态选择主查询：text_boost > 0 时用 query_string，否则用 match_all
                "query": (
                    {
                        # "query_string": {"query": query_string}
                        "bool": {
                            "must": [
                                {
                                    "query_string": {"query": query_string}
                                },  # 原有查询条件
                                {"exists": {"field": "abstract_vector"}},
                                {"exists": {"field": "title_vector"}},
                            ]
                        }
                    }
                    if text_boost != 0
                    else {"match_all": {}}
                ),
                # "script": {
                #     "source": """
                #         double score = 0;
                #         if (params.text_boost > 0) {
                #             score = _score * params.text_boost;
                #         }
                #         if (params.abstract_boost > 0) {
                #             score += cosineSimilarity(params.query_vector, 'abstract_vector') * params.abstract_boost;
                #         }
                #         if (params.title_boost > 0) {
                #             score += cosineSimilarity(params.query_vector, 'title_vector') * params.title_boost;
                #         }
                #         return score;
                #     """,
                "script": {
                    "source": """
                        double score = 0;
                        if (params.text_boost > 0) {
                            score = _score * params.text_boost;
                        }
                        // 检查 abstract_vector 是否存在且非空
                        if (params.abstract_boost > 0 
                            && doc.containsKey('abstract_vector') 
                            && !'abstract_vector'.empty) {
                            score += cosineSimilarity(
                                params.query_vector, 'abstract_vector'
                            ) * params.abstract_boost;
                        }
                        // 检查 title_vector 是否存在且非空
                        if (params.title_boost > 0 
                            && doc.containsKey('title_vector') 
                            && !'title_vector'.empty) {
                            score += cosineSimilarity(
                                params.query_vector, 'title_vector'
                            ) * params.title_boost;
                        }
                        return score;
                    """,
                    "params": {
                        "query_vector": query_vector,
                        "text_boost": text_boost,
                        "abstract_boost": abstract_boost,
                        "title_boost": title_boost,
                    },
                },
            }
        },
        "timeout": "300s",
        "size": size,
    }

    # 如果没有向量查询且 text_boost=0，回退到纯文本查询（或 match_all）
    if query_vector is None and text_boost == 0:
        query["query"] = {"match_all": {}}
    elif query_vector is None:
        query["query"] = {"query_string": {"query": query_string}}

    return query


def rewrite_original_query(query: str, model: str):
    system_message = {
        "role": "system",
        "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain.",
    }
    user_message = {
        "role": "user",
        "content": f"""
        Rewrite the provided medical-related user query into one alternative version while preserving the original intent and meaning. 
        The rewritten query should maintain clinical accuracy, use appropriate medical terminology, and enhance clarity for improved information retrieval. 
        Ensure the restructured query reflects natural healthcare communication patterns without altering diagnostic or treatment-related semantics.

        Example input query: "What are the early warning signs of myocardial infarction?" 
        Example output: What clinical symptoms typically present as initial indicators of a heart attack, and how do they manifest in different patient demographics?
        
        Please generate a query string for the following biomedical question and wrap the final query in ## tags:
        '{query}'
        """,
    }
    messages = [system_message, user_message]
    completion = get_chat_completion(model=model, messages=messages)

    if "deepseek" in model.lower():
        answer = completion.choices[0].message.content.split("</think>")[-1]
    else:
        answer = completion.choices[0].message.content

    return answer


def expand_query_few_shot(df_prior, n, question: str, model: str):
    messages = generate_n_shot_examples_expansion(df_prior, n)
    user_message = {
        "role": "user",
        "content": f"""
        Given a biomedical question, generate an Elasticsearch query string that incorporates synonyms and related terms to improve the search results while maintaining precision and relevance to the original question.

        The index contains the fields 'title' and 'abstract', which use the English stemmer. The query string syntax supports the following operators:
        - '+' and '-' for requiring or excluding terms (e.g., +fox -news)
        - '""' for phrase search (e.g., "quick brown")
        - ':' for field-specific search (e.g., title:(quick OR brown))
        - '*' or '?' for wildcards (e.g., qu?ck bro*)
        - '//' for regular expressions (e.g., title:/joh?n(ath[oa]n)/)
        - '~' for fuzzy matching (e.g., quikc~ or quikc~2)
        - '"..."~N' for proximity search (e.g., "fox quick"~5)
        - '^' for boosting terms (e.g., quick^2 fox)
        - 'AND', 'OR', 'NOT' for boolean matching (e.g., ((quick AND fox) OR (brown AND fox) OR fox) AND NOT news)

        Example:
        Question: What are the effects of vitamin D deficiency on the human body?
        Query string: (("vitamin d" OR "vitamin d3" OR "cholecalciferol") AND (deficiency OR insufficiency OR "low levels")) AND ("effects" OR "impact" OR "consequences") AND ("human body" OR "human health")

        Tips:
        - Focus on the main concepts and entities in the question.
        - Use synonyms and related terms to capture variations in terminology.
        - Be cautious not to introduce irrelevant terms that may dilute the search results.
        - Strike a balance between precision and recall based on the specificity of the question.

        Please generate a query string for the following biomedical question and wrap the final query in ## tags:
        '{question}'
        """,
    }
    messages.append(user_message)

    # print("Prompt Messages:")
    # print(messages)

    # completion = client_openai.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.0,  # randomness of completion
    #     seed=90128538,
    # )
    completion = get_chat_completion(model=model, messages=messages)
    # answer = completion.choices[0].message.content
    if "deepseek" in model.lower():
        answer = completion.choices[0].message.content.split("</think>")[-1]
    else:
        answer = completion.choices[0].message.content
    # print("\n Completion:")
    # print(answer)
    # print("\n")
    return answer


def generate_n_shot_examples_expansion(df, n):
    system_message = {
        "role": "system",
        "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain.",
    }
    messages = [system_message]

    if n < 1:
        top_entries = pd.DataFrame()
    else:
        top_entries = df.sort_values(by="f1_score", ascending=False).head(n)

    for _, row in top_entries.iterrows():
        question = row["question_body"]
        completion = row["completion"]
        question = question.replace("/", "\\\\/")

        user_message = {
            "role": "user",
            "content": f"""
            Given a biomedical question, generate an Elasticsearch query string that incorporates synonyms and related terms to improve the search results while maintaining precision and relevance to the original question.

            The index contains the fields 'title' and 'abstract', which use the English stemmer. The query string syntax supports the following operators:
            - '+' and '-' for requiring or excluding terms (e.g., +fox -news)
            - '""' for phrase search (e.g., "quick brown")
            - ':' for field-specific search (e.g., title:(quick OR brown))
            - '*' or '?' for wildcards (e.g., qu?ck bro*)
            - '//' for regular expressions (e.g., title:/joh?n(ath[oa]n)/)
            - '~' for fuzzy matching (e.g., quikc~ or quikc~2)
            - '"..."~N' for proximity search (e.g., "fox quick"~5)
            - '^' for boosting terms (e.g., quick^2 fox)
            - 'AND', 'OR', 'NOT' for boolean matching (e.g., ((quick AND fox) OR (brown AND fox) OR fox) AND NOT news)

            Example:
            Question: What are the effects of vitamin D deficiency on the human body?
            Query string: (("vitamin d" OR "vitamin d3" OR "cholecalciferol") AND (deficiency OR insufficiency OR "low levels")) AND ("effects" OR "impact" OR "consequences") AND ("human body" OR "human health")

            Tips:
            - Focus on the main concepts and entities in the question.
            - Use synonyms and related terms to capture variations in terminology.
            - Be cautious not to introduce irrelevant terms that may dilute the search results.
            - Strike a balance between precision and recall based on the specificity of the question.

            Please generate a query string for the following biomedical question and wrap the final query in ## tags:
            '{question}'
            """,
        }

        assistant_message = {"role": "assistant", "content": completion}

        messages.extend([user_message, assistant_message])

    return messages


def refine_query_with_no_results(question, original_query, model):
    messages = [
        {
            "role": "system",
            "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain.",
        },
        {
            "role": "user",
            "content": f"""Given that the following search query has returned no documents, please generate a broader query that retains the original question's context and relevance. Return only the query that can directly be used without any explanation text. Focus on maintaining the query's precision and relevance to the original question.

        To generate a broader query, consider the following:

        Identify the main concepts in the original query and prioritize them based on their importance to the question.
        Simplify the query by removing less essential terms or concepts that might be too specific or restrictive.
        Use more general terms or synonyms for the main concepts to expand the search scope while maintaining relevance.
        Reduce the number of Boolean operators (AND, OR) to make the query less restrictive.
        If the original query includes specific drug names, genes, or proteins, consider using their classes or families instead.
        Avoid using too many search fields or specific phrases in quotes, as they can limit the search results.
        Original question: '{question}', Original query that returned no results: '{original_query}' think step by step an wrapp the improved query in ## tags:""",
        },
    ]

    # print("Prompt Messages:")
    # print(messages)

    # completion = client_openai.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.0,  # randomness of completion
    #     seed=90128538,
    # )
    completion = get_chat_completion(model=model, messages=messages)
    # answer = completion.choices[0].message.content
    if "deepseek" in model.lower():
        answer = completion.choices[0].message.content.split("</think>")[-1]
    else:
        answer = completion.choices[0].message.content
    # print("\n Completion:")
    # print(answer)
    # print("\n")
    return answer


def find_extract_json(text):
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, text, re.DOTALL)
    match = matches[0]
    match_clean = match.replace("\\", "\\\\")
    match_clean = match_clean.replace("\t", "\\t")
    return match_clean


def normalize_unicode_string(s, form="NFKC"):
    normalized = normalize("NFKD", s).encode("ascii", "ignore").decode()
    return normalized


def generate_n_shot_examples_extraction(examples, n):
    n_shot_examples = []
    for example in examples[:n]:
        for message in example["messages"]:
            if message["role"] != "system":
                n_shot_examples.append(message)
    return n_shot_examples


def extract_relevant_snippets_few_shot(
    examples, n, article: str, question: str, model: str
) -> str:
    system_message = {
        "role": "system",
        "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain.",
    }
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_extraction(examples, n)
    messages.extend(few_shot_examples)
    user_message = {
        "role": "user",
        "content": f"""Given this question: '{question}' extract relevant sentences or longer snippets from the following article that help answer the question. 
    If no relevant information is present, return an empty array. Return the extracted snippets as a json string array called 'snippets'. ```{article}```""",
    }
    messages.append(user_message)
    # print("Prompt Messages:")
    # print(messages)

    # completion = client_openai.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.0,  # randomness of completion
    #     response_format={"type": "json_object"},
    #     seed=90128538,  # 90128538
    # )

    # print("\n Completion:")
    # print(completion)
    # print("\n")

    # try:
    #     if "deepseek" in model.lower():
    #         answer = completion.choices[0].message.content.split("</think>")[-1]
    #     else:
    #         answer = completion.choices[0].message.content
    #     json_response = find_extract_json(answer)
    #     # json_response = find_extract_json(completion.choices[0].message.content)
    #     sentences = json.loads(json_response)
    # except Exception as e:
    #     print(f"Error parsing response as json: {json_response}: {e}")
    #     traceback.print_exc()
    #     sentences = {"snippets": []}
    # try:
    #     snippets = generate_snippets_from_sentences(article, sentences["snippets"])
    # except Exception as e:
    #     print(f"Error getting snippets from {sentences}: {e}")
    #     snippets = []

    MAX_RETRIES = 3  # 最大重试次数
    RETRY_INTERVAL = 3  # 重试间隔（秒）
    for attempt in range(MAX_RETRIES + 1):  # 0~MAX_RETRIES 共 MAX_RETRIES+1 次尝试
        try:
            if attempt > 0:
                random_seed = random.randint(0, 1000000)
                completion = get_chat_completion(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    seed=random_seed,
                )
            else:
                completion = get_chat_completion(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            if "deepseek" in model.lower():
                answer = completion.choices[0].message.content.split("</think>")[-1]
            else:
                answer = completion.choices[0].message.content

            # JSON 解析与校验
            json_response = find_extract_json(answer)
            sentences = json.loads(json_response)

            snippets = generate_snippets_from_sentences(article, sentences["snippets"])

            break
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
            traceback.print_exc()
            if attempt + 1 == MAX_RETRIES:
                print("⚠️ 达到最大重试次数，启用降级方案")
                sentences = {"snippets": []}
                return []
            else:
                print(f"⏳ {RETRY_INTERVAL}秒后重试...")
                time.sleep(RETRY_INTERVAL)

    return snippets


def find_offset_and_create_snippet(document_id, text, sentence, section):
    text = normalize_unicode_string(text)
    sentence = normalize_unicode_string(sentence)
    offset_begin = text.find(sentence)
    offset_end = offset_begin + len(sentence)
    return {
        "document": document_id,
        "offsetInBeginSection": offset_begin,
        "offsetInEndSection": offset_end,
        "text": sentence,
        "beginSection": section,
        "endSection": section,
    }


def generate_snippets_from_sentences(article, sentences):
    snippets = []

    article_abstract = article.get("abstract") or ""
    article_abstract = normalize_unicode_string(article_abstract)
    article_title = normalize_unicode_string(article.get("title"))

    for sentence in sentences:
        sentence = normalize_unicode_string(sentence)
        if sentence in article_title:
            snippet = find_offset_and_create_snippet(
                article["id"], article["title"], sentence, "title"
            )
            snippets.append(snippet)
        elif sentence in article_abstract:
            snippet = find_offset_and_create_snippet(
                article["id"], article_abstract, sentence, "abstract"
            )
            snippets.append(snippet)
        else:
            # print("\nsentences not found in article: " + sentence + "\n")
            # print(article)
            pass

    return snippets
