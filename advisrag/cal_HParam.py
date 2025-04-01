import math
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

# PlotQA 2x2 + 4x4

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


local_image_embeddings_2x2 = np.load('embeddings/PlotQA_corpus_embeddings_2x2.npy')
local_image_embeddings_4x4 = np.load('embeddings/PlotQA_corpus_embeddings_4x4.npy')
image_embeddings = np.load('embeddings/PlotQA_corpus_embeddings.npy')
query_embeddings = np.load('embeddings/PlotQA_queries_with_instruction_embeddings.npy')
corpus_ids = np.load('embeddings/PlotQA_corpus_corpus_ids.npy')
query_ids = np.load('embeddings/PlotQA_queries_query_ids.npy')
qrels = load_beir_qrels('dataset/VisRAG-Ret-Test-PlotQA/qrels/plotqa-eval-qrels.tsv')

with open('PlotQA_exploration_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Gamma1', 'Gamma2', 'Gamma3', 'Temperature1', 'Temperature2', 'Cutoff', 'MRR'])

# gamma_list = [round(0.1 * i, 1) for i in range(6, 11)]
gamma_list = []
for i in range(6, 11):
    for j in range(11 - i):
        k = 10 - i - j
        if j == 0 or k == 0:
            continue
        gamma_list.append((round(i * 0.1, 1), round(j * 0.1, 1), round(k * 0.1, 1)))
gamma_list.reverse()
# gamma_list = [round(0.80 + 0.01 * i, 2) for i in range(21)]

# 将numpy数组转换为PyTorch张量并移动到GPU
local_image_embeddings_tensor_2x2 = torch.tensor(local_image_embeddings_2x2).cuda()
local_image_embeddings_tensor_4x4 = torch.tensor(local_image_embeddings_4x4).cuda()
image_embeddings_tensor = torch.tensor(image_embeddings).cuda()
query_embeddings_tensor = torch.tensor(query_embeddings).cuda()

best_mrr = 0
print("Testing PlotQA...")
# for gamma1, gamma2, gamma3 in tqdm(gamma_list):
for gamma1, gamma2, gamma3 in gamma_list:
    print(f'\ngamma1: {gamma1}, gamma2: {gamma2}, gamma3: {gamma3}')
    run = {} 
    for temperature1 in range(10, 51, 5):
    # for temperature1 in [1]:
        for temperature2 in range(10, 51, 5):
        # for temperature2 in [30]:
            print(f'temperature1: {temperature1}, temperature2: {temperature2}')
            for q_idx, query in enumerate(query_embeddings_tensor):
                qid = query_ids[q_idx]
                
                scores1 = torch.einsum('ijk,k->ij', local_image_embeddings_tensor_2x2, query)
                scores2 = torch.einsum('ijk,k->ij', local_image_embeddings_tensor_4x4, query)

                # temperature1 = 100.0
                # temperature2 = 10.0
                scaled_scores1 = scores1 * temperature1
                scaled_scores2 = scores2 * temperature2
                alpha1 = torch.softmax(scaled_scores1, dim=1)
                alpha2 = torch.softmax(scaled_scores2, dim=1)
                
                local_agg1 = torch.einsum('ij,ijk->ik', alpha1, local_image_embeddings_tensor_2x2)
                local_agg2 = torch.einsum('ij,ijk->ik', alpha2, local_image_embeddings_tensor_4x4)
                

                final_fusion = gamma1 * image_embeddings_tensor + gamma2 * local_agg1 + gamma3 * local_agg2
                final_score = torch.matmul(final_fusion, query)
                
                top_k_indices = torch.argsort(final_score, descending=True)[:10]  # 取前10个
                run[qid] = {corpus_ids[idx]: float(final_score[idx].cpu().numpy()) for idx in top_k_indices}
            for cutoff in [3, 10]:    
                evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{cutoff}", f"recall.{cutoff}"})  
                eval_results = evaluator.evaluate(run)  
                
                for measure in sorted(eval_results[next(iter(eval_results))].keys()):  
                    value = pytrec_eval.compute_aggregated_measure(  
                        measure, [query_measures[measure] for query_measures in eval_results.values()]  
                    )  
                    print(f"{measure:25s}{'all':8s}{value:.4f}")  

                
                mrr = eval_mrr(qrels, run, cutoff)['all']  
                print(f'MRR@{cutoff}: {mrr}')  
                
                with open('PlotQA_exploration_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([gamma1, gamma2, gamma3, temperature1, temperature2, cutoff, mrr])
                    
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_gamma1 = gamma1
                    best_gamma2 = gamma2
                    best_gamma3 = gamma3
                    best_temperature1 = temperature1
                    best_temperature2 = temperature2
                print(f'Current Best MRR: {best_mrr}, Best Gamma1: {best_gamma1}, Best Gamma2: {best_gamma2}, Best Gamma3: {best_gamma3}, Best Temperature1: {best_temperature1}, Best Temperature2: {best_temperature2}')
print(f'PlotQA\nBest MRR: {best_mrr}, Best Gamma1: {best_gamma1}, Best Gamma2: {best_gamma2}, Best Gamma3: {best_gamma3}, Best Temperature1: {best_temperature1}, Best Temperature2: {best_temperature2}')
    