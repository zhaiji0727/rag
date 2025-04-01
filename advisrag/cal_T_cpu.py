import math
import numpy as np
from tqdm import tqdm
import pytrec_eval
import csv
from multiprocessing import Pool, cpu_count

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

def process_query(args):
    q_idx, query, local_image_embeddings, image_embeddings, query_ids, corpus_ids, temperature, gamma = args
    qid = query_ids[q_idx]
    scores = np.einsum('ijk,k->ij', local_image_embeddings, query)
    scaled_scores = scores * temperature
    alpha = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=1, keepdims=True)
    local_agg = np.einsum('ij,ijk->ik', alpha, local_image_embeddings)
    final_fusion = gamma * image_embeddings + (1 - gamma) * local_agg
    final_score = np.dot(final_fusion, query)
    top_k_indices = np.argsort(final_score)[::-1][:10]
    return qid, {corpus_ids[idx]: float(final_score[idx]) for idx in top_k_indices}

if __name__ == '__main__':
    local_image_embeddings = np.load('embeddings/MP_DocVQA_corpus_embeddings_5x5.npy')
    image_embeddings = np.load('embeddings/MP_DocVQA_corpus_embeddings.npy')
    query_embeddings = np.load('embeddings/MP_DocVQA_queries_with_instruction_embeddings.npy')
    corpus_ids = np.load('embeddings/MP_DocVQA_corpus_corpus_ids.npy')
    query_ids = np.load('embeddings/MP_DocVQA_queries_query_ids.npy')
    qrels = load_beir_qrels('dataset/VisRAG-Ret-Test-MP-DocVQA/qrels/docvqa_mp-eval-qrels.tsv')

    gamma_list = [round(0.1 * i, 1) for i in range(5, 10)]

    best_mrr = 0
    best_gamma = 0
    best_temperature = 0

    for temperature in tqdm([1] + list(range(5, 50, 5))):
        print(f'\nTesting temperature: {temperature}')
        for gamma in gamma_list:
            print(f'  gamma: {gamma}') 
            run = {}
            with Pool(cpu_count()) as pool:
                args = [(q_idx, query, local_image_embeddings, image_embeddings, query_ids, corpus_ids, temperature, gamma) for q_idx, query in enumerate(query_embeddings)]
                results = pool.map(process_query, args)
                for qid, result in results:
                    run[qid] = result
            
            mrr = eval_mrr(qrels, run, 10)['all']
            print(f'    MRR@10: {mrr}')
            
            if mrr > best_mrr:
                best_mrr = mrr
                best_gamma = gamma
                best_temperature = temperature
        
        print(f'\tMP_DocVQA Current Best MRR: {best_mrr}, Best Gamma: {best_gamma}, Best Temperature: {best_temperature}')

    print(f'MP_DocVQA Best MRR: {best_mrr}, Best Gamma: {best_gamma}, Best Temperature: {best_temperature}')