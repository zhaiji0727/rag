def rerank_snippets(examples, n, snippets, question: str, model: str) -> str:
    numbered_snippets = [
        {"id": idx, "text": snippet["text"]} for idx, snippet in enumerate(snippets)
    ]
    system_message = {
        "role": "system",
        "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain.",
    }
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_reranking(examples, n)
    messages.extend(few_shot_examples)
    user_message = {
        "role": "user",
        "content": f"""Given this question: '{question}' select the top 10 snippets that are most helpful for answering this question from
    this list of snippets, rerank them by helpfulness: ```{numbered_snippets}``` return a json array of their ids called 'snippets'""",
    }
    messages.append(user_message)
    # print("Prompt Messages:")
    # print(messages)

    # completion = client_openai.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.0,
    #     response_format={"type": "json_object"},
    #     seed=90128538,
    # )
    # completion = get_chat_completion(
    #     model=model, messages=messages, response_format={"type": "json_object"}
    # )
    # print("\n Completion:")
    # print(completion)
    # print("\n")
    # if "deepseek" in model.lower():
    #     answer = completion.choices[0].message.content.split("</think>")[-1]
    # else:
    #     answer = completion.choices[0].message.content
    # json_response = find_extract_json(answer)
    # json_response = find_extract_json(completion.choices[0].message.content)

    # try:
    #     snippets_reranked = json.loads(json_response)
    #     snippets_idx = snippets_reranked["snippets"]
    #     filtered_array = [snippets[i] for i in snippets_idx]
    # except Exception as e:
    # print(f"Error parsing response as json: {json_response}: {e}")
    # traceback.print_exc()
    # filtered_array = snippets

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
            json_response = find_extract_json(answer)

            snippets_reranked = json.loads(json_response)
            snippets_idx = snippets_reranked["snippets"]
            filtered_array = [snippets[i] for i in snippets_idx]

            break
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
            traceback.print_exc()
            if attempt + 1 == MAX_RETRIES:
                print("⚠️ 达到最大重试次数，启用降级方案")
                return snippets
            else:
                print(f"⏳ {RETRY_INTERVAL}秒后重试...")
                time.sleep(RETRY_INTERVAL)

    return filtered_array


def save_state(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
        print(f"Saved state to: {file_path}")


def load_state(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                print(f"Loaded state from: {file_path}")
                return pickle.load(f)
    except EOFError:
        return None
    return None


def read_jsonl_file(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))
    return examples


def extract_text_wrapped_in_tags(input_string):
    pattern = "##(.*?)##"
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        extracted_text = match.group(1).replace("\n", "")
        return extracted_text
    else:
        return "ERROR"


def reorder_articles_by_snippet_sequence(relevant_article_ids, snippets):
    ordered_article_ids = []
    mentioned_article_ids = set()

    for snippet in snippets:
        document_id = snippet["document"]
        if (
            document_id in relevant_article_ids
            and document_id not in mentioned_article_ids
        ):
            ordered_article_ids.append(document_id)
            mentioned_article_ids.add(document_id)

    for article_id in relevant_article_ids:
        if article_id not in mentioned_article_ids:
            ordered_article_ids.append(article_id)

    return ordered_article_ids


def get_relevant_snippets(examples, n, articles, question, model_name):
    processed_articles = []
    # for article in tqdm(articles, desc="Extracting relevant snippets."):
    for article in articles:
        snippets = extract_relevant_snippets_few_shot(
            examples, n, article, question, model_name
        )
        if snippets:
            article["snippets"] = snippets
            processed_articles.append(article)
    return processed_articles


def encode_texts(texts):
    with _bgem_semaphore:
        return embed_model.encode(
            texts, return_dense=True, return_colbert_vecs=False, return_sparse=False
        )["dense_vecs"]


def process_question(
    question,
    query_examples,
    snip_extract_examples,
    snip_rerank_examples,
    n_shot,
    model_name,
    text_boost,
    abstract_boost,
    title_boost,
):
    try:
        query_string = ""
        improved_query_string = ""
        relevant_articles_ids = []
        filtered_articles_ids = []
        reordered_articles_ids = []
        relevant_snippets = []

        question_id = question["id"]
        print(f"Processing question {question_id}")

        if abstract_boost != 0.0 or title_boost != 0.0:
            # add rewrite part
            rewrite_completion = rewrite_original_query(question["body"], model_name)
            rewrite_query_string = extract_text_wrapped_in_tags(rewrite_completion)
            # print(f'query: {question["body"]}\nrewrite query: {rewrite_query_string}')
            
            attempt = 0
            MAX_RETRIES = 3
            while query_string == "ERROR" and attempt < MAX_RETRIES:
                # 重试时传入之前生成的random_seed
                random_seed = random.randint(0, 1000000)
                rewrite_completion = rewrite_original_query(question["body"], model_name, random_seed)
                rewrite_query_string = extract_text_wrapped_in_tags(rewrite_completion)
                attempt += 1

            # query_vector = encode_texts(question["body"])
            query_vector = encode_texts(rewrite_query_string)
            # query_vector = encode_texts(question["body"])
        else:
            query_vector = None

        wiki_context = ""

        completion = expand_query_few_shot(
            query_examples, n_shot, question["body"], model_name
        )
        query_string = extract_text_wrapped_in_tags(completion)

        attempt = 0
        MAX_RETRIES = 3
        while (query_string == "ERROR" or not validate_es_query(query_string)) and attempt < MAX_RETRIES:
            # 重试时传入之前生成的random_seed
            random_seed = random.randint(0, 1000000)
            completion = expand_query_few_shot(
                query_examples, max(0, n_shot - attempt - 1), question["body"], model_name, seed=random_seed
            )
            query_string = extract_text_wrapped_in_tags(completion)
            attempt += 1

        query = create_query(
            query_string, query_vector, text_boost, abstract_boost, title_boost
        )
        relevant_articles = run_elasticsearch_query(query)
        if len(relevant_articles) == 0:
            improved_query_completion = refine_query_with_no_results(
                question["body"], query_string, model_name
            )
            improved_query_string = extract_text_wrapped_in_tags(
                improved_query_completion
            )
            query = create_query(
                query_string, query_vector, text_boost, abstract_boost, title_boost
            )
            relevant_articles = run_elasticsearch_query(query)
            if len(relevant_articles) > 0:
                print("Query refinement worked")

        relevant_articles_ids = [article["id"] for article in relevant_articles]

        filtered_articles = get_relevant_snippets(
            snip_extract_examples,
            n_shot,
            relevant_articles,
            question["body"],
            model_name,
        )
        filtered_articles_ids = [article["id"] for article in filtered_articles]
        relevant_snippets = [
            snippet for article in filtered_articles for snippet in article["snippets"]
        ]

        reranked_snippets = rerank_snippets(
            snip_rerank_examples,
            n_shot,
            relevant_snippets,
            question["body"],
            model_name,
        )

        reordered_articles_ids = reorder_articles_by_snippet_sequence(
            filtered_articles_ids, reranked_snippets
        )

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
            "snippets": reranked_snippets,
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
            "snippets": relevant_snippets or [],
        }


def csv_to_json(csv_filepath, json_filepath):
    empty = 0
    df = pd.read_csv(csv_filepath)
    questions_list = df.to_dict(orient="records")
    json_structure = {"questions": []}

    for item in questions_list:
        if item["question_type"] in ["list", "factoid"]:
            exact_answer_format = [[]]
        else:
            exact_answer_format = ""

        if len(eval(item["relevant_articles"])) == 0:
            empty += 1

        question_dict = {
            "documents": eval(item["documents"])[:10],
            "snippets": eval(item["snippets"])[:10],
            "body": item["question_body"],
            "type": item["question_type"],
            "id": item["question_id"],
            "ideal_answer": "",
        }
        if item["question_type"] != "summary":
            question_dict["exact_answer"] = exact_answer_format

        json_structure["questions"].append(question_dict)

    with open(json_filepath, "w", encoding="utf-8") as json_file:
        json.dump(json_structure, json_file, ensure_ascii=False, indent=4)
    print(empty)


def execute_evaluation(
    text_boost,
    abstract_boost,
    title_boost,
    golden_filepath,
    json_filepath,
    csv_filepath="evaluation_results.csv",
):
    try:
        cmd = f"java -Xmx10G -cp $CLASSPATH:/home/samsung/haoquan/Evaluation-Measures-master/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseA -e 5 {golden_filepath} {json_filepath}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"命令执行失败，错误信息：{result.stderr}")
            return False

        output = result.stdout
        result_list = output.strip().split()

        if len(result_list) < 16:
            print("输出不完整，无法提取所需的数字。")
            return False

        section5_10 = result_list[5:10]
        section11_15 = result_list[10:15]
        selected_data = section5_10 + section11_15

        if len(selected_data) != 10:
            print("提取的数据长度不正确，期望10个数字。")
            return False

        headers = [
            "text boost",
            "abstract boost",
            "title boost",
            "MPrec documents",
            "MRec documents",
            "MF1 documents",
            "MAP documents",
            "GMAP documents",
            "MPrec snippets",
            "MRec snippets",
            "MF1 snippets",
            "MAP snippets",
            "GMAP snippets",
        ]
        selected_data = [text_boost, abstract_boost, title_boost] + selected_data

        with open(csv_filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if os.path.getsize(csv_filepath) == 0:
                writer.writerow(headers)
                writer.writerow(selected_data)
            else:
                writer.writerow(selected_data)

        return True

    except Exception as e:
        print(f"执行评估时发生错误：{str(e)}")
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Elasticsearch query parameters configuration"
    )

    parser.add_argument(
        "--text_boost",
        type=float,
        default=1.0,
        help="Boost value for text field (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--abstract_boost",
        type=float,
        default=0.0,
        help="Boost value for abstract field (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--title_boost",
        type=float,
        default=0.0,
        help="Boost value for title field (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        choices=["mistral", "llama3-70b", "deepseek-r1:70b-q80", "deepseek-r1:70b", "glm4:32b"],
        help="Select model version (options: mistral, llama3-70b，deepseek-r1:70b-q80, deepseek-r1:70b, glm4:32b)",
    )
    parser.add_argument(
        "--forward_port",
        type=int,
        default=11434,
        help="Local port for API forwarding (1024-65535, default: 11434)",
    )
    parser.add_argument(
        "--multiclient",
        action="store_true",
        help="Enable multi-client mode, alternating between multiple OpenAI clients",
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        help="Path to pickle file containing data from previous runs (default: None)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    text_boost = args.text_boost
    abstract_boost = args.abstract_boost
    title_boost = args.title_boost

    global enable_multiclient
    enable_multiclient = args.multiclient
    # enable_multiclient = False #######################################################################
    if enable_multiclient:
        # client_openai_1 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11435/v1",
        #     timeout=3000,
        # )
        # client_openai_2 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11436/v1",
        #     timeout=3000,
        # )
        # client_openai_3 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11437/v1",
        #     timeout=3000,
        # )
        # client_openai_4 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11438/v1",
        #     timeout=3000,
        # )
        # client_openai_5 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11439/v1",
        #     timeout=3000,
        # )
        # client_openai_6 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11440/v1",
        #     timeout=3000,
        # )
        # client_openai_7 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11441/v1",
        #     timeout=3000,
        # )
        # client_openai_8 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11442/v1",
        #     timeout=3000,
        # )
        # client_openai_9 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11443/v1",
        #     timeout=3000,
        # )
        # client_openai_10 = OpenAI(
        #     api_key="ollama",
        #     base_url=f"http://127.0.0.1:11444/v1",
        #     timeout=3000,
        # )
        # semaphore_1 = threading.Semaphore(2)
        # semaphore_2 = threading.Semaphore(2)
        # semaphore_3 = threading.Semaphore(2)
        # semaphore_4 = threading.Semaphore(2)
        # semaphore_5 = threading.Semaphore(2)
        # semaphore_6 = threading.Semaphore(2)
        # semaphore_7 = threading.Semaphore(2)
        # semaphore_8 = threading.Semaphore(2)
        # semaphore_9 = threading.Semaphore(2)
        # semaphore_10 = threading.Semaphore(2)
        # global client_rotator
        # # client_rotator = ClientRotator(
        # #     [(client_openai_1, semaphore_1), (client_openai_2, semaphore_2)]
        # # )
        # # client_rotator = ClientRotator(
        # #     [(client_openai_1, semaphore_1), (client_openai_2, semaphore_2), (client_openai_3, semaphore_3)]
        # # )
        # client_rotator = ClientRotator(
        #     [
        #         (client_openai_1, semaphore_1),
        #         (client_openai_2, semaphore_2),
        #         (client_openai_3, semaphore_3),
        #         (client_openai_4, semaphore_4),
        #         (client_openai_5, semaphore_5),
        #         (client_openai_6, semaphore_6),
        #         (client_openai_7, semaphore_7),
        #         (client_openai_8, semaphore_8),
        #         (client_openai_9, semaphore_9),
        #         (client_openai_10, semaphore_10),
        #     ]
        # )
        
        # 配置参数
        base_port = 11435  # 起始端口
        num_clients = 2    # 客户端数量
        semaphore_value = 2 # 每个信号量的初始值

        # 智能生成端口列表
        ports = range(base_port, base_port + num_clients)
        
        # 批量创建客户端
        clients = [
            OpenAI(
                api_key="ollama",
                base_url=f"http://127.0.0.1:{port}/v1",
                timeout=3000,
            )
            for port in ports
        ]
        
        # 批量创建信号量
        semaphores = [threading.Semaphore(semaphore_value) for _ in ports]
        
        # 组合客户端与信号量
        global client_rotator
        client_rotator = ClientRotator(list(zip(clients, semaphores)))
        
    else:
        forward_port = args.forward_port
        # Initialize OpenAI client
        global client_openai
        if forward_port == 11434:
            client_openai = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                timeout=3000,
            )
        else:
            client_openai = OpenAI(
                api_key="ollama",
                base_url=f"http://127.0.0.1:{str(forward_port)}/v1",
                timeout=3000,
            )

    global embed_model
    if abstract_boost != 0.0 or title_boost != 0.0:
        embed_model = BGEM3FlagModel(r"/mnt/data/haoquan/model/bge-m3")
    else:
        embed_model = None
        print("Embed model not initialized~")

    # model_name = 'llama3.2:3B'
    # model_name = "mistral"
    model_name = args.model
    n_shot = 10

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not args.pickle_file:
        pickl_name = (
            model_name.replace("/", "-").replace(":", "-")
            if "/" in model_name or ":" in model_name
            else model_name
        )
        pickl_file = f"/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/pkl_{timestamp}-{pickl_name}-{n_shot}-shot.pkl"
    else:
        pickl_file = args.pickle_file

    input_file_name = "/mnt/data/dataset/BioASQ/Task12BGoldenEnriched/12B1_golden.json"
    # input_file_name = "/home/samsung/haoquan/test.json"
    query_examples = pd.read_csv(
        # "2024-03-26_19-24-27_claude-3-opus-20240229_11B1-10-Shot_Retrieval.csv"
        "/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/2024-03-26_19-24-27_claude-3-opus-20240229_11B1-10-Shot_Retrieval.csv"
    )
    snip_extract_examples = read_jsonl_file(
        "/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/Snippet_Extraction_Examples.jsonl"
    )
    snip_rerank_examples = read_jsonl_file(
        "/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/Snippet_Reranking_Examples.jsonl"
    )

    with open(input_file_name) as input_file:
        data = json.loads(input_file.read())

    saved_df = load_state(pickl_file)
    questions_df = (
        saved_df
        if saved_df is not None and not saved_df.empty
        else pd.DataFrame(
            columns=[
                "question_id",
                "question_body",
                "question_type",
                "wiki_context",
                "completion",
                "query",
                "improved_query",
                "relevant_articles",
                "filtered_articles",
                "documents",
                "snippets",
            ]
        )
    )

    processed_ids = (
        set(questions_df["question_id"]) if not questions_df.empty else set()
    )
    questions_to_process = [
        q for q in data["questions"] if q["id"] not in processed_ids
    ]

    total_questions = len(questions_to_process)
    processed_count = 0

    start_time = time.time()

    # Parallel run
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_question = {
            executor.submit(
                process_question,
                q,
                query_examples,
                snip_extract_examples,
                snip_rerank_examples,
                n_shot,
                model_name,
                text_boost,
                abstract_boost,
                title_boost,
            ): q
            for q in questions_to_process
        }

        for future in as_completed(future_to_question):
            question = future_to_question[future]
            processed_count += 1
            try:
                result = future.result()
                if result:
                    result_df = pd.DataFrame([result])
                    questions_df = pd.concat(
                        [questions_df, result_df], ignore_index=True
                    )
                    save_state(questions_df, pickl_file)
                print(
                    f"Progress: {processed_count}/{total_questions} questions processed ({(processed_count / total_questions) * 100:.2f}%)"
                )
            except Exception as e:
                print(f"Error processing question {question['id']}: {e}")
                traceback.print_exc()

    """
    # Sequential run, debug use
    for q in questions_to_process:
        processed_count += 1
        try:
            # 直接同步调用处理函数（替代 executor.submit）
            result = process_question(
                q,
                query_examples,
                snip_extract_examples,
                snip_rerank_examples,
                n_shot,
                model_name,
                text_boost,
                abstract_boost,
                title_boost,
            )

            if result:
                # 保存结果到 DataFrame
                result_df = pd.DataFrame([result])
                questions_df = pd.concat([questions_df, result_df], ignore_index=True)
                # 每次处理完立即保存状态
                save_state(questions_df, pickl_file)

            # 打印进度
            print(
                f"Progress: {processed_count}/{total_questions} questions processed ({(processed_count / total_questions) * 100:.2f}%)"
            )

        except Exception as e:
            print(f"Error processing question {q['id']}: {e}")
            traceback.print_exc()
    """

    print(
        f"text boost: {text_boost}, abstract boost: {abstract_boost}, title boost: {title_boost}, Model name: {model_name}, n-shot: {n_shot}, Time: {timedelta(seconds=time.time() - start_time)}"
    )

    model_name_pretty = model_name.split("/")[-1] if "/" in model_name else model_name
    output_filepath = f"/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/Results/Evaluation/{timestamp}_{model_name_pretty}_12B1_PhaseA_{n_shot}-Shot_text-{str(int(round(text_boost * 100)))}_abstract-{str(int(round(abstract_boost * 100)))}_title-{str(int(round(title_boost * 100)))}.csv"
    # print(output_filepath)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    questions_df.to_csv(output_filepath, index=False)

    json_filepath = os.path.splitext(output_filepath)[0] + ".json"
    csv_to_json(output_filepath, json_filepath)

    golden_filepath = "/mnt/data/dataset/BioASQ/Task12BGoldenEnriched/12B1_golden.json"
    eval_csv_filepath = f"/home/samsung/haoquan/bioasq2024-main/02_12B/Batch1/PhaseA/Results/Evaluation/{model_name}_{n_shot}_shot_12B1_PhaseA_evaluation_results.csv"

    # 执行评估并写入CSV
    success = execute_evaluation(
        text_boost,
        abstract_boost,
        title_boost,
        golden_filepath,
        json_filepath,
        eval_csv_filepath,
    )
    if success:
        print(f'评估完成，数据已保存到"{eval_csv_filepath}"。')
    else:
        print("评估失败，请检查输入文件和配置。")

    try:
        if os.path.exists(pickl_file):
            os.remove(pickl_file)
            print("Intermediate state pickle file deleted successfully.")
    except Exception as e:
        print(f"Error deleting pickle file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
