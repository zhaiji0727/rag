import math
import pathlib
from typing import Dict, Any
import torch
from tqdm import tqdm
import pytrec_eval
import numpy as np
import csv

def eval_mrr(qrel, run, cutoff=None):  
    """  
    Compute MRR@cutoff manually.  
    """  
    mrr = 0.0  
    num_ranked_q = 0  
    results = {}  
    for qid in qrel:  
        if qid not in run:  
            continue  
        num_ranked_q += 1  
        docid_and_score = [(docid, score) for docid, score in run[qid].items()]  
        docid_and_score.sort(key=lambda x: x[1], reverse=True)  
        for i, (docid, _) in enumerate(docid_and_score):  
            rr = 0.0  
            if cutoff is None or i < cutoff:  
                if docid in qrel[qid] and qrel[qid][docid] > 0:  
                    rr = 1.0 / (i + 1)  
                    break  
        results[qid] = rr  
        mrr += rr  
    mrr /= num_ranked_q  
    results["all"] = mrr  
    return results

import torch
from tqdm import tqdm
import pytrec_eval
import numpy as np
import csv

def load_beir_qrels(qrels_file):  
    qrels = {}  
    try:  
        with open(qrels_file) as f:  
            tsvreader = csv.DictReader(f, delimiter="\t")  
            for row in tsvreader:  
                qid = row["query-id"]  
                pid = row["corpus-id"]  
                rel = int(row["score"])  
                if qid in qrels:  
                    qrels[qid][pid] = rel  
                else:  
                    qrels[qid] = {pid: rel}  
    except Exception as e:  
        print(f"Error loading qrels file: {e}")  
    return qrels 

def save_as_trec(
    rank_result: Dict[str, Dict[str, Dict[str, Any]]], output_path: str, run_id: str = "OpenMatch"
):
    """
    Save the rank result as TREC format:
    <query_id> Q0 <doc_id> <rank> <score> <run_id>
    """
    pathlib.Path("/".join(output_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for qid in rank_result:
            # sort the results by score
            sorted_results = sorted(
                rank_result[qid].items(), key=lambda x: x[1], reverse=True
            )
            for i, (doc_id, score) in enumerate(sorted_results):
                f.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, doc_id, i + 1, score, run_id))


local_image_embeddings_2x2 = np.load('embeddings/InfoVQA_corpus_embeddings_2x2.npy')
local_image_embeddings_4x4 = np.load('embeddings/InfoVQA_corpus_embeddings_4x4.npy')
image_embeddings = np.load('embeddings/InfoVQA_corpus_embeddings.npy')
query_embeddings = np.load('embeddings/InfoVQA_queries_with_instruction_embeddings.npy')
corpus_ids = np.load('embeddings/InfoVQA_corpus_corpus_ids.npy')
query_ids = np.load('embeddings/InfoVQA_queries_query_ids.npy')
qrels = load_beir_qrels('dataset/VisRAG-Ret-Test-InfoVQA/qrels/infographicsvqa-eval-qrels.tsv')


# 将numpy数组转换为PyTorch张量并移动到GPU
local_image_embeddings_tensor_2x2 = torch.tensor(local_image_embeddings_2x2).cuda()
local_image_embeddings_tensor_4x4 = torch.tensor(local_image_embeddings_4x4).cuda()
image_embeddings_tensor = torch.tensor(image_embeddings).cuda()
query_embeddings_tensor = torch.tensor(query_embeddings).cuda()


print("Testing InfoVQA...")

gamma1 = 0.7
gamma2 = 0
gamma3 = 0.3
temperature1 = 35
temperature2 = 35

run = {}

for q_idx, query in enumerate(query_embeddings_tensor):
    qid = query_ids[q_idx]
    
    scores1 = torch.einsum('ijk,k->ij', local_image_embeddings_tensor_2x2, query)
    scores2 = torch.einsum('ijk,k->ij', local_image_embeddings_tensor_4x4, query)

    scaled_scores1 = scores1 * temperature1
    scaled_scores2 = scores2 * temperature2
    alpha1 = torch.softmax(scaled_scores1, dim=1)
    alpha2 = torch.softmax(scaled_scores2, dim=1)
    
    t_list = [10 * i for i in range(1, 6)]
    scores2_np = scores2.cpu().numpy()
    np.save('scores2.npy', scores2_np)
    loaded_scores2 = np.load('scores2.npy')
    scaled_scores = [loaded_scores2 * t for t in t_list]
    alpha = [np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True) for s in scaled_scores]
    
    # unscaled_alpha2 = torch.softmax(scores2, dim=1)
    unscaled_alpha2 = np.exp(scores2_np) / np.sum(np.exp(scores2_np), axis=1, keepdims=True)
    
    # Plot the distributions
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 18))

    indices = range(1, len(unscaled_alpha2[0]) + 1, len(unscaled_alpha2[0]) // 4)

    # Plot unscaled_alpha2[10]
    plt.subplot(3, 2, 1)
    plt.bar(range(1, len(unscaled_alpha2[0]) + 1), unscaled_alpha2[10], alpha=0.5, label='unscaled_alpha2')
    plt.title('Original')
    plt.xlabel('Sub-Image')
    plt.ylabel('Attention Score')
    plt.xticks(indices)  # Set x-axis to show integer ticks
    # plt.legend()

    # Plot each element in alpha list
    for i in range(5):
        plt.subplot(3, 2, i + 2)
        plt.bar(range(1, len(alpha[i][10]) + 1), alpha[i][10], alpha=0.5, label=f'alpha[{i}]')
        plt.title(f'T = {t_list[i]}')
        plt.xlabel('Sub-Image')
        plt.ylabel('Attention Score')
        plt.xticks(indices)  # Set x-axis to show integer ticks
        # plt.legend()

    plt.tight_layout(h_pad=5.0, w_pad=2.0)
    plt.show()

    
    exit(0)
    
    local_agg1 = torch.einsum('ij,ijk->ik', alpha1, local_image_embeddings_tensor_2x2)
    local_agg2 = torch.einsum('ij,ijk->ik', alpha2, local_image_embeddings_tensor_4x4)
    

    final_fusion = gamma1 * image_embeddings_tensor + gamma2 * local_agg1 + gamma3 * local_agg2
    final_score = torch.matmul(final_fusion, query)
    
    top_k_indices = torch.argsort(final_score, descending=True)[:10]  # 取前10个
    run[qid] = {corpus_ids[idx]: float(final_score[idx].cpu().numpy()) for idx in top_k_indices}

# save_as_trec(run, 'Output/paper_results/InfoVQA/test.1.trec')    

for cutoff in [1, 3, 5, 10]:    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{cutoff}", f"recall.{cutoff}"})  
    eval_results = evaluator.evaluate(run)  
    
    for measure in sorted(eval_results[next(iter(eval_results))].keys()):  
        value = pytrec_eval.compute_aggregated_measure(  
            measure, [query_measures[measure] for query_measures in eval_results.values()]  
        )  
        print(f"{measure:25s}{'all':8s}{value:.4f}")  

    
    mrr = eval_mrr(qrels, run, cutoff)['all']  
    print(f'MRR@{cutoff}: {mrr}')  
                
    