import json
import os
import pickle
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Suppress warnings about Elasticsearch certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize OpenAI client
client_openai = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')

def escape_for_json(input_string):
    return json.dumps(input_string)

def run_elasticsearch_query(query, index="pubmed25"):
    es = Elasticsearch("http://109.105.34.64:9200", verify_certs=False, timeout=120)

    if isinstance(query, str) and not isinstance(query, dict):
        query_dict = json.loads(query)
    else:
        query_dict = query

    print("\n running es query:")
    print(query_dict)
    print("\n")
    response = es.search(index=index, body=query_dict)

    results = []
    if response['hits']['hits']:
        for hit in response['hits']['hits']:
            result = {
                "id": "http://www.ncbi.nlm.nih.gov/pubmed/"+str(hit['_id']),
                "title": hit['_source'].get('title', 'No title available'),
                "abstract": hit['_source'].get('abstract', 'No abstract available')
            }
            results.append(result)
    print(f"docs found: {len(results)}")
    return results

def create_query(query_string: str, size=50):
    return {
        "query": {
            "query_string": {
                "query": query_string
            }
        },
        "size": size
    }

def expand_query_few_shot(df_prior, n, question:str, model:str):
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
        """
    }
    messages.append(user_message)

    print("Prompt Messages:")
    print(messages)

    completion = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # randomness of completion
        seed=90128538
    )
    answer = completion.choices[0].message.content
    print("\n Completion:")
    print(answer)
    print("\n")
    return answer

def generate_n_shot_examples_expansion(df, n):
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]

    if n < 1:
        top_entries = pd.DataFrame()
    else:
        top_entries = df.sort_values(by='f1_score', ascending=False).head(n)

    for _, row in top_entries.iterrows():
        question = row['question_body']
        completion = row['completion']
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
            """
        }

        assistant_message = {
            "role": "assistant",
            "content": completion
        }

        messages.extend([user_message, assistant_message])

    return messages

def refine_query_with_no_results(question, original_query, model):
    messages = [
        {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."},
        {"role": "user", "content": f"""Given that the following search query has returned no documents, please generate a broader query that retains the original question's context and relevance. Return only the query that can directly be used without any explanation text. Focus on maintaining the query's precision and relevance to the original question.

        To generate a broader query, consider the following:

        Identify the main concepts in the original query and prioritize them based on their importance to the question.
        Simplify the query by removing less essential terms or concepts that might be too specific or restrictive.
        Use more general terms or synonyms for the main concepts to expand the search scope while maintaining relevance.
        Reduce the number of Boolean operators (AND, OR) to make the query less restrictive.
        If the original query includes specific drug names, genes, or proteins, consider using their classes or families instead.
        Avoid using too many search fields or specific phrases in quotes, as they can limit the search results.
        Original question: '{question}', Original query that returned no results: '{original_query}' think step by step an wrapp the improved query in ## tags:"""}
    ]

    print("Prompt Messages:")
    print(messages)

    completion = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # randomness of completion
        seed=90128538
    )
    answer = completion.choices[0].message.content
    print("\n Completion:")
    print(answer)
    print("\n")
    return answer

def find_extract_json(text):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text, re.DOTALL)
    match = matches[0]
    match_clean = match.replace('\\', "\\\\")
    match_clean = match_clean.replace('\t', "\\t")
    return match_clean

def normalize_unicode_string(s, form='NFKC'):
    normalized = normalize('NFKD', s).encode('ascii','ignore').decode()
    return normalized

def generate_n_shot_examples_extraction(examples, n):
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':
                n_shot_examples.append(message)
    return n_shot_examples

def extract_relevant_snippets_few_shot(examples, n, article:str, question:str, model:str) -> str:
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_extraction(examples, n)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""Given this question: '{question}' extract relevant sentences or longer snippets from the following article that help answer the question. 
    If no relevant information is present, return an empty array. Return the extracted snippets as a json string array called 'snippets'. ```{article}```"""}
    messages.append(user_message)
    print("Prompt Messages:")
    print(messages)

    completion = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # randomness of completion
        response_format={ "type": "json_object" },
        seed=90128538
    )
    print("\n Completion:")
    print(completion)
    print("\n")
    json_response = find_extract_json(completion.choices[0].message.content)
    try:
        sentences = json.loads(json_response)
    except Exception as e:
        print(f"Error parsing response as json: {json_response}: {e}")
        traceback.print_exc()
        sentences = {"snippets": []}
    
    snippets = generate_snippets_from_sentences(article, sentences['snippets'])
    
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
        "endSection": section
    }

def generate_snippets_from_sentences(article, sentences):
    snippets = []

    article_abstract = article.get('abstract') or ''
    article_abstract = normalize_unicode_string(article_abstract)
    article_title = normalize_unicode_string(article.get('title'))

    for sentence in sentences:
        sentence = normalize_unicode_string(sentence)
        if sentence in article_title:
            snippet = find_offset_and_create_snippet(article['id'], article['title'], sentence, "title")
            snippets.append(snippet)
        elif sentence in article_abstract:
            snippet = find_offset_and_create_snippet(article['id'], article_abstract, sentence, "abstract")
            snippets.append(snippet)
        else:
            print("\nsentences not found in article: "+sentence+"\n")
            print(article)

    return snippets

def generate_n_shot_examples_reranking(examples, n):
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':
                n_shot_examples.append(message)
    return n_shot_examples

def rerank_snippets(examples, n, snippets, question:str, model:str) -> str:
    numbered_snippets = [{'id': idx, 'text': snippet['text']} for idx, snippet in enumerate(snippets)]
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_reranking(examples, n)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""Given this question: '{question}' select the top 10 snippets that are most helpful for answering this question from
    this list of snippets, rerank them by helpfulness: ```{numbered_snippets}``` return a json array of their ids called 'snippets'"""}
    messages.append(user_message)
    print("Prompt Messages:")
    print(messages)

    completion = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={ "type": "json_object" },
        seed=90128538
    )
    print("\n Completion:")
    print(completion)
    print("\n")
    json_response = find_extract_json(completion.choices[0].message.content)

    try:
        snippets_reranked = json.loads(json_response)
        snippets_idx = snippets_reranked['snippets']
        filtered_array = [snippets[i] for i in snippets_idx]
    except Exception as e:
        print(f"Error parsing response as json: {json_response}: {e}")
        traceback.print_exc()
        filtered_array = snippets

    return filtered_array

def save_state(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_state(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except EOFError:
        return None
    return None

def read_jsonl_file(file_path):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            examples.append(json.loads(line))
    return examples

def extract_text_wrapped_in_tags(input_string):
    pattern = "##(.*?)##"
    match = re.search(pattern, input_string, re.DOTALL)  
    if match:
        extracted_text = match.group(1).replace('\n', '')
        return extracted_text
    else:
        return "ERROR"

def reorder_articles_by_snippet_sequence(relevant_article_ids, snippets):
    ordered_article_ids = []
    mentioned_article_ids = set()

    for snippet in snippets:
        document_id = snippet['document']
        if document_id in relevant_article_ids and document_id not in mentioned_article_ids:
            ordered_article_ids.append(document_id)
            mentioned_article_ids.add(document_id)

    for article_id in relevant_article_ids:
        if article_id not in mentioned_article_ids:
            ordered_article_ids.append(article_id)

    return ordered_article_ids

def get_relevant_snippets(examples, n, articles, question, model_name):
    processed_articles = []
    for article in articles:
        snippets = extract_relevant_snippets_few_shot(examples, n, article, question, model_name)
        if snippets:
            article['snippets'] = snippets
            processed_articles.append(article)
    return processed_articles

def process_question(question, query_examples, snip_extract_examples, snip_rerank_examples, n_shot, model_name):
    try:
        query_string = ""
        improved_query_string = ""
        relevant_articles_ids = []
        filtered_articles_ids = []
        reordered_articles_ids = []
        relevant_snippets = []

        question_id = question['id']
        print(f"Processing question {question_id}")
        wiki_context = ""

        completion = expand_query_few_shot(query_examples, n_shot, question['body'], model_name)
        query_string = extract_text_wrapped_in_tags(completion)
        query = create_query(query_string)

        relevant_articles = run_elasticsearch_query(query)
        if len(relevant_articles) == 0:
            improved_query_completion = refine_query_with_no_results(question['body'], query_string, model_name)
            improved_query_string = extract_text_wrapped_in_tags(improved_query_completion)
            query = create_query(improved_query_string)
            relevant_articles = run_elasticsearch_query(query)
            if len(relevant_articles) > 0:
                print("Query refinement worked")
            
        relevant_articles_ids = [article['id'] for article in relevant_articles]
        
        filtered_articles = get_relevant_snippets(snip_extract_examples, n_shot, relevant_articles, question['body'], model_name)
        filtered_articles_ids = [article['id'] for article in filtered_articles]
        relevant_snippets = [snippet for article in filtered_articles for snippet in article['snippets']]

        reranked_snippets = rerank_snippets(snip_rerank_examples, n_shot, relevant_snippets, question['body'], model_name)
        
        reordered_articles_ids = reorder_articles_by_snippet_sequence(filtered_articles_ids, reranked_snippets)

        return {
            "question_id": question["id"],
            "question_body": question["body"],
            "question_type": question["type"],
            "wiki_context": wiki_context,
            "completion": completion,
            "query": query_string,
            "improved_query": improved_query_string,
            "relevant_articles": relevant_articles_ids,
            "filtered_articles": filtered_articles_ids,
            "documents": reordered_articles_ids,
            "snippets": reranked_snippets
        }
    except Exception as e:
        print(f"Error processing question {question['id']}: {e}")
        traceback.print_exc()
        return {
            "question_id": question.get("id", "error"),
            "question_body": question.get("body", "error"),
            "question_type": question.get("type", "error"),
            "query": query_string or "error",
            "improved_query": improved_query_string or "error",
            "relevant_articles": relevant_articles_ids or [],
            "filtered_articles