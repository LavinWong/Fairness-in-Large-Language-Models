from collections import defaultdict
import torch
from tqdm import tqdm
import pandas as pd
from medium_sized.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.data import *
from medium_sized.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.evaluate import *

def calculate_cps(model, token_ids, spans, mask_id, log_softmax, device):
    '''
    Given token ids of a sequence, return the summed log probability of
    masked shared tokens between sequence pair (CPS).
    '''
    spans = spans[1:-1]
    token_ids = token_ids.to(device)
    masked_token_ids = token_ids.repeat(len(spans), 1)
    masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id

    hidden_states = model(masked_token_ids)[0]
    token_ids = token_ids.view(-1)[spans]
    log_probs = log_softmax(hidden_states[range(hidden_states.size(0)), spans, :])
    span_log_probs = log_probs[range(hidden_states.size(0)), token_ids]
    score = torch.sum(span_log_probs).item()

    ranks = get_rank_for_gold_token(log_probs, token_ids.view(-1, 1))

    return score, ranks

def run_experiment():
    print("------------Medium-sized LMs: Intrinsic bias - Probability-based bias - CPS------------")
    tokenizer, model = load_tokenizer_and_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    bias_score_list = []
    bias_type_list = []
    dataset_list = []
    for i in range(0, 2):
        total_score = 0
        stereo_score = 0
        mask_id = tokenizer.mask_token_id
        log_softmax = torch.nn.LogSoftmax(dim=1)
        count = defaultdict(int)
        scores = defaultdict(int)
        all_ranks = []
        if i == 0:
            print("------------ CPS - Crows-Pairs Dataset------------")
            inputs = load_crows_pairs_dataset()
            dataset_name = "Crows-Pairs"
        else:
            print("------------ CPS - Stereoset Dataset------------")
            inputs = load_stereoset_dataset()
            dataset_name = "Stereoset"

        for input in tqdm(inputs):
            bias_type = input['bias_type']
            count[bias_type] += 1

            pro_sentence = input['stereotype']
            pro_token_ids = tokenizer.encode(pro_sentence, return_tensors='pt').to(device)
            anti_sentence = input['anti-stereotype']
            anti_token_ids = tokenizer.encode(anti_sentence, return_tensors='pt').to(device)

            with torch.no_grad():
                pro_spans, anti_spans = get_span(pro_token_ids[0], anti_token_ids[0], 'equal')
                pro_score, pro_ranks = calculate_cps(model, pro_token_ids, pro_spans, mask_id, log_softmax, device)
                anti_score, anti_ranks = calculate_cps(model, anti_token_ids, anti_spans, mask_id, log_softmax, device)
                pro_score = round(pro_score, 3)
                anti_score = round(anti_score, 3)
                
            all_ranks += anti_ranks
            all_ranks += pro_ranks
            total_score += 1
            if pro_score > anti_score:
                stereo_score += 1
                scores[bias_type] += 1

        bias_score = (stereo_score / total_score) * 100

        bias_score_list.append(bias_score)
        bias_type_list.append("Bias score")
        dataset_list.append(dataset_name)

        for bias_type, score in sorted(scores.items()):
            bias_score = (score / count[bias_type]) * 100
            bias_score_list.append(bias_score)
            bias_type_list.append(bias_type)
            dataset_list.append(dataset_name)
  
        all_ranks = [rank for rank in all_ranks if rank != -1]
        accuracy = sum([1 for rank in all_ranks if rank == 1]) / len(all_ranks)
        accuracy *= 100
        bias_score_list.append(accuracy)
        bias_type_list.append('Accuracy')
        dataset_list.append(dataset_name)

    df = pd.DataFrame({
        'Bias_type': bias_type_list,
        'Dataset': dataset_list,
        'Score': bias_score_list,
    })

    df.to_csv("medium_sized/intrinsic_bias/probability_based/pseudo_log_likelihood_metrics/cps/result.csv", index=False)