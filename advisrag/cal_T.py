import math
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

local_image_embeddings = np.load('embeddings/InfoVQA_corpus_embeddings_4x4.npy')
image_embeddings = np.load('embeddings/InfoVQA_corpus_embeddings.npy')
query_embeddings = np.load('embeddings/InfoVQA_queries_with_instruction_embeddings.npy')
corpus_ids = np.load('embeddings/InfoVQA_corpus_corpus_ids.npy')
query_ids = np.load('embeddings/InfoVQA_queries_query_ids.npy')
qrels = load_beir_qrels('dataset/VisRAG-Ret-Test-InfoVQA/qrels/infographicsvqa-eval-qrels.tsv')

# gamma_list = [round(0.1 * i, 1) for i in range(5, 10)]
gamma_list = [0.6, 0.7, 0.8, 0.9]

local_image_embeddings_tensor = torch.tensor(local_image_embeddings).cuda()
image_embeddings_tensor = torch.tensor(image_embeddings).cuda()
query_embeddings_tensor = torch.tensor(query_embeddings).cuda()

best_mrr = 0
best_gamma = 0
best_temperature = 0

import matplotlib.pyplot as plt

mrr_results = {}

for temperature in tqdm([1] + list(range(5, 51, 5))):
    print(f'\nTesting temperature: {temperature}')
    for gamma in gamma_list:
        print(f'  gamma: {gamma}') 
        run = {} 
        for q_idx, query in enumerate(query_embeddings_tensor):
            qid = query_ids[q_idx]
            
            scores = torch.einsum('ijk,k->ij', local_image_embeddings_tensor, query)
            
            scaled_scores = scores * temperature
            alpha = torch.softmax(scaled_scores, dim=1)
            local_agg = torch.einsum('ij,ijk->ik', alpha, local_image_embeddings_tensor)
            
            final_fusion = gamma * image_embeddings_tensor + (1 - gamma) * local_agg
            final_score = torch.matmul(final_fusion, query)
            
            top_k_indices = torch.argsort(final_score, descending=True)[:10]
            run[qid] = {corpus_ids[idx]: float(final_score[idx].cpu().numpy()) for idx in top_k_indices}
        
        mrr = eval_mrr(qrels, run, 10)['all']
        print(f'    MRR@10: {mrr}')
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_gamma = gamma
            best_temperature = temperature
        
        mrr_results[(temperature, gamma)] = mrr

    print(f'\tInfoVQA Current Best MRR: {best_mrr}, Best Gamma: {best_gamma}, Best Temperature: {best_temperature}')

print(f'InfoVQA Best MRR: {best_mrr}, Best Gamma: {best_gamma}, Best Temperature: {best_temperature}')

# InfoVting the results
temperatures = sorted(set(temp for temp, _ in mrr_results.keys()))
gammas = sorted(set(gamma for _, gamma in mrr_results.keys()))

for gamma in gammas:
    mrr_values = [mrr_results[(temp, gamma)] for temp in temperatures]
    plt.plot(temperatures, mrr_values, label=f'Gamma: {gamma}')

plt.xlabel('Temperature')
plt.ylabel('MRR@10')
plt.title('InfoVQA\nMRR@10 vs Temperature for different Gamma values')
plt.legend()
plt.grid(True)
plt.show()