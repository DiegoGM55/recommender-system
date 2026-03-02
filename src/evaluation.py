import math
import numpy as np
import pandas as pd

def precision_at_k(recs, truth, k=10):
    recs_k = recs[:k]
    if k == 0: return 0.0
    hits = sum(1 for i in recs_k if i in truth)
    return hits / k

def recall_at_k(recs, truth, k=10):
    if len(truth) == 0: return 0.0
    recs_k = recs[:k]
    hits = sum(1 for i in recs_k if i in truth)
    return hits / len(truth)

def ndcg_at_k(recs, truth, k=10):
    recs_k = recs[:k]
    # DCG: relevância binária (1 se item está no truth, senão 0)
    dcg  = sum((1.0 / math.log2(idx+2)) for idx, i in enumerate(recs_k) if i in truth)
    # IDCG: melhor caso = todos relevantes no topo
    ideal_hits = min(len(truth), k)
    idcg = sum((1.0 / math.log2(i+2)) for i in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate_recommender(users, recommend_fn, truth_dict, exclude_seen_dict, k=10):
    """recommend_fn(u, K, exclude=set(...)) -> lista de itens recomendados"""
    precs, recs, ndcgs = [], [], []
    for u in users:
        truth = truth_dict.get(u, set())
        if not truth:     # só avalia quem tem item de validação
            continue
        recs_u = recommend_fn(u, K=k, exclude=exclude_seen_dict.get(u, set()))
        precs.append(precision_at_k(recs_u, truth, k))
        recs.append(recall_at_k(recs_u, truth, k))
        ndcgs.append(ndcg_at_k(recs_u, truth, k))

    return {
        "users_eval": len(precs),
        "P@{}".format(k): float(np.mean(precs)) if precs else 0.0,
        "R@{}".format(k): float(np.mean(recs)) if recs else 0.0,
        "NDCG@{}".format(k): float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

def get_results_df(results, model_name):
    """
    Converte a lista de resultados em um DataFrame e adiciona o nome do modelo.
    """
    df_results = pd.DataFrame(results)
    df_results['Model'] = model_name
    return df_results

def compare_models(df_model_A, df_model_B, model_A_name, model_B_name, metric_cols=['Precision', 'Recall', 'NDCG']):
    """
    Compara as métricas de dois modelos e imprime o resultado formatado
    com a melhoria percentual.
    """
    print(f"--- Comparação: {model_A_name} vs. {model_B_name} ---\n")
    
    # Junta os dois dataframes pela coluna 'K'
    comparison_df = pd.merge(df_model_A, df_model_B, on='K', suffixes=(f'_{model_A_name}', f'_{model_B_name}'))
    
    for k_value in comparison_df['K'].unique():
        print(f"K={k_value}\n")
        
        row = comparison_df[comparison_df['K'] == k_value].iloc[0]
        
        for metric in metric_cols:
            val_A = row[f'{metric}_{model_A_name}']
            val_B = row[f'{metric}_{model_B_name}']
            
            # Calcula a melhoria percentual (uplift)
            uplift = ((val_A - val_B) / val_B) * 100 if val_B > 0 else float('inf')
            
            # Formata a string de saída
            print(f"- {metric}@{k_value}: {val_A:.4f} vs {val_B:.4f} → {uplift:+.0f}%")
        
        print("-" * 20)