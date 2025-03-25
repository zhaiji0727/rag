# %%
# %pip install Biopython openai "elasticsearch<8" python-dotenv mistralai fireworks-ai sentence_transformers
# %pip install --upgrade pandas
# %pip install websocket-client wikipedia-api wikipedia
# %pip install --upgrade fireworks-ai

# %%
from openai import OpenAI
# from fireworks.client import Fireworks
# import anthropic
import re
import os
import json
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import datetime
import pickle
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import timedelta

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

#Suppress warnings about elasticsearch certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# client_openai = OpenAI()
client_openai = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
# client_fireworks = Fireworks()
# client_anthropic = anthropic.Anthropic()

# %% [markdown]
# ## Retrieval

# %%
def escape_for_json(input_string):
    escaped_string = json.dumps(input_string)
    return escaped_string

# Load environment variables from .env file
load_dotenv()


def run_elasticsearch_query(query, index="pubmed25"):
    # Retrieve Elasticsearch details from environment variables
    # es_host = os.getenv('ELASTICSEARCH_HOST')
    # es_user = os.getenv('ELASTICSEARCH_USER')
    # es_password = os.getenv('ELASTICSEARCH_PASSWORD')

    # # Connect to Elasticsearch
    # es = Elasticsearch(
    #     [es_host],
    #     http_auth=(es_user, es_password),
    #     verify_certs=False,  # This will ignore SSL certificate validation
    #     timeout=120  # Set the timeout to 60 seconds (adjust as needed)
    # )
    es = Elasticsearch("http://109.105.34.64:9200", verify_certs=False, timeout=120)

    # Convert the query string to a dictionary
    if isinstance(query, str) and not isinstance(query, dict):
        query_dict = json.loads(query)
    else:
        query_dict = query

    print("\n running es query:")
    print(query_dict)
    print("\n")
    # Execute the query
    # response = es.search(query_dict, index=index)
    response = es.search(index=index, body=query_dict)

    # Process the response to extract the required information
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

def createQuery(query_string: str, size=50): 
    query = {
        "query": {
            "query_string": {
                "query": query_string
            }
        },
        "size": size
    }
    return query

# %% [markdown]
# ## Query Expansion

# %%
def expand_query_few_shot(df_prior, n, question:str, model:str):
    messages = generate_n_shot_examples_expansion(df_prior, n)
    # Add the user message
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
    
    if "accounts" in model:
        completion = client_fireworks.chat.completions.create(
            model=model,
            messages =messages,
            max_tokens = 4096,
            prompt_truncate_len = 27000,
            temperature=0.0 # randomness of completion
        )
        answer = completion.choices[0].message.content
    elif "claude" in model:
        system_message_content = messages.pop(0)['content']
        completion = client_anthropic.messages.create(
            model=model,
            system=system_message_content,
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
        answer = completion.content[0].text
    else:
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

def expand_query_wiki(wiki_context: str, question:str, model: str)-> str:
    # Add the user message
    user_message = {
        "role": "user",
        "content": f"""
        {wiki_context}
        Answer this question: '{question}' 
        Think step by step and write an exhaustive answer explaining your reasoning"""
    }
    messages = [
        {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."},
        user_message
    ]
    print("\nMessages Expand Query:")
    print(messages)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0, # randomness of completion
        logprobs=False,
        seed=90128538
    )
    query = f"""
        {{
            "query": {{
                "more_like_this" : {{
                "fields" : ["title", "abstract"],
                "like" : {escape_for_json(completion.choices[0].message.content)},
                "min_term_freq" : 1,
                "min_doc_freq": 1,
                "boost_terms": 1
                }}
            }},
        "size":100
        }}
    """
    return query

def generate_n_shot_examples_expansion(df, n):
    
    # Initialize the system message
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    
    # Initialize the list of messages with the system message
    messages = [system_message]
    
    
    if n< 1:
        top_entries = pd.DataFrame()
    else:
        top_entries = df.sort_values(by='f1_score', ascending=False).head(n)
    
    # Loop through each of the top n entries and add the user and assistant messages
    for _, row in top_entries.iterrows():
        question = row['question_body']
        completion = row['completion']
        
        # Replace problematic characters in question
        question = question.replace("/", "\\\\/")
        
        # Add the user message
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
        
        # Add the assistant message
        assistant_message = {
            "role": "assistant",
            "content": completion  
        }
        
        messages.extend([user_message, assistant_message])

    return messages

# %% [markdown]
# ## Query Refinement

# %%
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
    
    if "accounts" in model:
        completion = client_fireworks.chat.completions.create(
            model=model,
            messages =messages,
            max_tokens = 4096,
            prompt_truncate_len = 27000,
            temperature=0.0 # randomness of completion
        )
        answer = completion.choices[0].message.content
    elif "claude" in model:
        system_message_content = messages.pop(0)['content']
        completion = client_anthropic.messages.create(
            model=model,
            system=system_message_content,
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
        answer = completion.content[0].text
    else:
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


# %% [markdown]
# ## Snippet Extraction

# %%
def find_extract_json(text):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text, re.DOTALL)
    match = matches[0]
    match_clean = match.replace('\\', "\\\\")
    match_clean = match_clean.replace('\t', "\\t")
    return match_clean

from unicodedata import normalize
def normalize_unicode_string(s, form='NFKC'):
    normalized  = normalize('NFKD', s).encode('ascii','ignore').decode()
    return normalized


def generate_n_shot_examples_extraction(examples, n):
    """Takes the top n examples, flattens their messages into one list, and filters out messages with the role 'system'."""
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':  # Only add messages that don't have the 'system' role
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
    
    if "accounts" in model:
        completion = client_fireworks.chat.completions.create(
            model=model,
            messages =messages,
            max_tokens = 4096,
            prompt_truncate_len = 27000,
            temperature=0.0 # randomness of completion
        )
    elif "claude" in model:
        system_message_content = messages.pop(0)['content']
        completion = client_anthropic.messages.create(
            model=model,
            system=system_message_content,
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
    else:
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
    if hasattr(completion, 'choices'):
        json_response = find_extract_json(completion.choices[0].message.content)
    else:
        json_response = find_extract_json(completion.content[0].text)
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

    article_abstract = article.get('abstract') or ''  # This will use '' if 'abstract' is None or does not exist
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

# %% [markdown]
# ## Snippet Reranking

# %%
def generate_n_shot_examples_reranking(examples, n):
    """Takes the top n examples, flattens their messages into one list, and filters out messages with the role 'system'."""
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':  # Only add messages that don't have the 'system' role
                n_shot_examples.append(message)
    return n_shot_examples

def rerank_snippets(examples, n, snippets, question:str, model:str) -> str:
    numbered_snippets = [{'id': idx, 'text': snippet['text']} for idx, snippet in enumerate(snippets)]
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_reranking(examples, n)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""Given this question: '{question}' select the top 10 snippets that are most helpfull for answering this question from
                    this list of snippets, rerank them by helpfullness: ```{numbered_snippets}``` return a json array of their ids called 'snippets'"""}
    messages.append(user_message)
    print("Prompt Messages:")
    print(messages)
    
    if "accounts" in model:
        completion = client_fireworks.chat.completions.create(
            model=model,
            messages =messages,
            max_tokens = 4096,
            prompt_truncate_len = 27000,
            temperature=0.0 # randomness of completion
        )
    elif "claude" in model:
        system_message_content = messages.pop(0)['content']
        completion = client_anthropic.messages.create(
            model=model,
            system=system_message_content,
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
    else:
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
    if hasattr(completion, 'choices'):
        json_response = find_extract_json(completion.choices[0].message.content)
    else:
        json_response = find_extract_json(completion.content[0].text)
    
    try:
        snippets_reranked = json.loads(json_response)
        snippets_idx = snippets_reranked['snippets']
        filtered_array = [snippets[i] for i in snippets_idx]
    except Exception as e:
        print(f"Error parsing response as json: {json_response}: {e}")
        traceback.print_exc()
        filtered_array = snippets
        
    return filtered_array

# %% [markdown]
# ## Run

# %%
#model_name = "gpt-3.5-turbo-0125"
# model_name = "claude-3-opus-20240229"
#model_name = "accounts/fireworks/models/mixtral-8x7b-instruct"
#model_name_extract = "accounts/samyateia-49f400/models/fa6580e58ba04e52b8a16b484d23bc14"
#model_name_rerank = "accounts/samyateia-49f400/models/336b7d6963d64fe0bc0de7264b737ba8"
# model_name = "mistral"
# model_name = 'llama3-70b'
model_name = 'llama3.2:3B'

n_shot = 10

# Get the current timestamp in a sortable format
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if '/' in model_name or ':' in model_name:
    pickl_name = model_name.replace('/', '-').replace(':', '-')
else:
    pickl_name = model_name
pickl_file = f'{pickl_name}-{n_shot}-shot.pkl'

def save_state(data, file_path=pickl_file):
    """Save the current state to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_state(file_path=pickl_file):
    """Load the state from a pickle file if it exists, otherwise return None."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except EOFError:  # Handles empty pickle file scenario
        return None
    return None

def read_jsonl_file(file_path):
    """Reads a JSONL file and returns a list of examples."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            examples.append(json.loads(line))
    return examples

def extract_text_wrapped_in_tags(input_string):
    pattern = "##(.*?)##"
    match = re.search(pattern, input_string, re.DOTALL)  
    if match:
        # Remove line breaks from the matched string
        extracted_text = match.group(1).replace('\n', '')
        return extracted_text
    else:
        return "ERROR"

def reorder_articles_by_snippet_sequence(relevant_article_ids, snippets):
    ordered_article_ids = []
    mentioned_article_ids = set()

    # Add article IDs in the order they appear in the snippets
    for snippet in snippets:
        document_id = snippet['document']
        if document_id in relevant_article_ids and document_id not in mentioned_article_ids:
            ordered_article_ids.append(document_id)
            mentioned_article_ids.add(document_id)

    # Add the remaining article IDs that weren't mentioned in snippets
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

# Run specific few-shot configuration
query_examples = pd.read_csv('2024-03-26_19-24-27_claude-3-opus-20240229_11B1-10-Shot_Retrieval.csv')

snip_extract_examples_file = "Snippet_Extraction_Examples.jsonl"     
snip_extract_examples = read_jsonl_file(snip_extract_examples_file)

snip_rerank_examples_file = "Snippet_Reranking_Examples.jsonl"     
snip_rerank_examples = read_jsonl_file(snip_rerank_examples_file)


def process_question(question):
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

        #0 query expansion
        completion = expand_query_few_shot(query_examples, n_shot, question['body'], model_name)
        query_string = extract_text_wrapped_in_tags(completion)
        query = createQuery(query_string)

        relevant_articles = run_elasticsearch_query(query)
        if len(relevant_articles) == 0:
            improved_query_completion = refine_query_with_no_results(question['body'], query_string, model_name)
            improved_query_string = extract_text_wrapped_in_tags(improved_query_completion)
            query = createQuery(improved_query_string)
            relevant_articles = run_elasticsearch_query(query)
            if len(relevant_articles) > 0:
                print("query refinement worked")
            
        relevant_articles_ids = [article['id'] for article in relevant_articles]
        
        #1 snippet extraction
        filtered_articles = get_relevant_snippets(snip_extract_examples, n_shot, relevant_articles, question['body'], model_name)
        filtered_articles_ids = [article['id'] for article in filtered_articles]
        relevant_snippets = [snippet for article in filtered_articles for snippet in article['snippets']]

        #2 rerank snippets
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
            "filtered_articles": filtered_articles_ids or [],
            "documents": reordered_articles_ids[:10] if reordered_articles_ids else [],
            "snippets": relevant_snippets or []
        }

# Define columns
columns = ['question_id', 'question_body', 'question_type', 'wiki_context', 'completion', 'query', 'improved_query', 'relevant_articles', 'filtered_articles', 'documents', 'snippets']

# Initialize empty DataFrame
questions_df = pd.DataFrame(columns=columns)

# Load the input file in JSON format
# input_file_name = 'BioASQ-task11bPhaseA-testset1.json'
# input_file_name = '/home/samsung/haoquan/training13b.json'
input_file_name = '/mnt/data/dataset/BioASQ/Task12BGoldenEnriched/12B1_golden.json'


with open(input_file_name) as input_file:
    data = json.loads(input_file.read())

# Assuming 'load_state' returns a DataFrame or None
saved_df = load_state(pickl_file)

if saved_df is not None and not saved_df.empty:
    processed_ids = set(saved_df['question_id'])  # Assuming 'question_id' is your identifier
    questions_df = saved_df
else:
    processed_ids = set()

# Assuming `data["questions"]` is your list of questions to process
# Filter out questions that have already been processed
questions_to_process = [q for q in data["questions"] if q["id"] not in processed_ids]
#questions_to_process = questions_to_process[:2]

total_questions = len(questions_to_process)
processed_count = 0

start_time = time.time()
# Use ThreadPoolExecutor to process questions in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    # Dictionary to keep track of question futures
    future_to_question = {executor.submit(process_question, q): q for q in questions_to_process}
    
    for future in as_completed(future_to_question):
        question = future_to_question[future]
        processed_count += 1
        try:
            result = future.result()
            if result:
                # Append result to the DataFrame
                result_df = pd.DataFrame([result])
                questions_df = pd.concat([questions_df, result_df], ignore_index=True)
                save_state(questions_df, pickl_file)
            print(f"Progress: {processed_count}/{total_questions} questions processed ({(processed_count / total_questions) * 100:.2f}%)")
        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            traceback.print_exc()
print(f'Model name: {model_name}, n-shot: {n_shot}, Time: {timedelta(seconds=time.time() - start_time)}')

'''
# Process questions sequentially
for question in tqdm(questions_to_process):
    try:
        # Process the question
        result = process_question(question)
        if result:
            # Append result to the DataFrame
            result_df = pd.DataFrame([result])
            questions_df = pd.concat([questions_df, result_df], ignore_index=True)
            # Save the updated DataFrame to the pickle file
            save_state(questions_df, pickl_file)
    except Exception as e:
        print(f"Error processing question {question['id']}: {e}")
        traceback.print_exc()
'''

# Prefix the output file name with the timestamp
if '/' in model_name:
    model_name_pretty = model_name.split("/")[-1]
else:
    model_name_pretty = model_name
output_file_name = f"./Results/{timestamp}_{model_name_pretty}_2024AB1-Fine-Tuned-{n_shot}-Shot.csv"

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

questions_df.to_csv(output_file_name, index=False)

# After processing all questions and saving the final output:
try:
    # Check if the pickle file exists before attempting to delete it
    if os.path.exists(pickl_file):
        os.remove(pickl_file)
        print("Intermediate state pickle file deleted successfully.")
except Exception as e:
    print(f"Error deleting pickle file: {e}")
    traceback.print_exc()

# %% [markdown]
# ## Create Run File

# %%
import pandas as pd
import json



def csv_to_json(csv_filepath, json_filepath):
    empty = 0
    # Step 1: Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filepath)
    
    # Transform the DataFrame into a list of dictionaries, one per question
    questions_list = df.to_dict(orient='records')
    
    # Initialize the structure of the JSON file
    json_structure = {"questions": []}
    
    # Step 2: Transform the DataFrame into the desired JSON structure
    for item in questions_list:
        # Adjusting exact_answer format based on question_type
        if item["question_type"] in ["list", "factoid"]:
            exact_answer_format = [[]]  # For 'list' or 'factoid', it's a list of lists
        else:
            exact_answer_format = ""  # Default to an empty string
            
            
        if len(eval(item["relevant_articles"])) == 0:
            empty = empty +1
        #print(len(eval(item["relevant_articles"])))
        # Construct question_dict conditionally excluding 'exact_answer' for 'ideal' type
        question_dict = {
            "documents": eval(item["documents"])[:10],
            "snippets": eval(item["snippets"])[:10],
            "body": item["question_body"],
            "type": item["question_type"],
            "id": item["question_id"],
            "ideal_answer": ""
        }
        if item["question_type"] != "summary":
            question_dict["exact_answer"] = exact_answer_format
        
        json_structure["questions"].append(question_dict)
    
    # Step 3: Write the JSON structure to a file
    with open(json_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(json_structure, json_file, ensure_ascii=False, indent=4)
    print(empty)

# Example usage
csv_filepath = './Results/2025-03-07_10-28-18_llama3-70b_2024AB1-Fine-Tuned-1-Shot.csv'  # Update this path to your actual CSV file path
json_filepath = './Results/2025-03-07_10-28-18_llama3-70b-1-shot.json'  # Update this path to where you want to save the JSON file
csv_to_json(csv_filepath, json_filepath)




