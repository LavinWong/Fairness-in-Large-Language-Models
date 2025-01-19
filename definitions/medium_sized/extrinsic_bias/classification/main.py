from medium_sized.extrinsic_bias.classification.data import *
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd

def get_TPR(y_pred, y_true, p2i, i2p, gender):
    
    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)
    
    for y_hat, y, g in zip(y_pred, y_true, gender):
        
        if y == y_hat:
            
            scores[i2p[y]][g] += 1
        
        prof_count_total[i2p[y]][g] += 1
    
    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []
    
    for profession, scores_dict in scores.items():
        
        good_m, good_f = scores_dict["m"], scores_dict["f"]
        prof_total_f = prof_count_total[profession]["f"]
        prof_total_m = prof_count_total[profession]["m"]
        tpr_m = (good_m) / prof_total_m
        tpr_f = (good_f) / prof_total_f
        
        tprs[profession]["m"] = tpr_m
        tprs[profession]["f"] = tpr_f
        tprs_ratio.append(0)
        tprs_change[profession] = tpr_f - tpr_m
        
    return tprs, tprs_change, np.mean(np.abs(tprs_ratio))

def rms_diff(tpr_diff):
    
    return np.sqrt(np.mean(tpr_diff**2))

def run_experiment():
    print("------------Medium-sized LMs: Extrinsic bias - Natural Language Understanding - Classification task------------")
    p2i, i2p, g2i, i2g, train, dev, test, x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression(warm_start = True, penalty = 'l2',
                            solver = "saga", multi_class = 'multinomial', fit_intercept = False,
                            verbose = 5, n_jobs = 90, random_state = 1, max_iter = 7)

    start = time.time()
    idx = np.random.rand(x_train.shape[0]) < 1.0
    clf.fit(x_train[idx], y_train[idx])
    print("time: {}".format(time.time() - start))
    accuracy = clf.score(x_test, y_test)
    
    y_pred_before = clf.predict(x_test)
    test_gender = [d["g"] for d in test]
    tprs_before, tprs_change_before, mean_ratio_before = get_TPR(y_pred_before, y_test, p2i, i2p, test_gender)

    change_vals_before = np.array(list((tprs_change_before.values())))
    gap = rms_diff(change_vals_before)

    print(f"Accuracy: {accuracy}")
    print(f"GAP: {gap}")

    res = {
        "Accuracy": accuracy,
        "GAP": gap
    }

    cont_list = [{"name": key, "value": value} for key, value in res.items()]
    df = pd.DataFrame(cont_list)
    df.to_csv("medium_sized/extrinsic_bias/classification/result.csv", index=False)