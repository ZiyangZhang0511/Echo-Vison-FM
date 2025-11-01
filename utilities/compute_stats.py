import os
import json
import argparse
import numpy as np
from scipy.stats import pearsonr

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_filepath", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    ### read json ###
    with open(args.json_filepath, "r") as f:
        json_data = json.load(f)
    # print(json_data.keys())

    prediction = 100*np.array(json_data["total_logits"]).squeeze()
    ground_truth = 100*np.array(json_data["total_targets"]).squeeze()
    # print(prediction.shape)

    # Pearson correlation
    rho = np.corrcoef(ground_truth, prediction)[0, 1]

    # Bias metrics
    mb   = np.mean(prediction - ground_truth)                  # mean bias
    mbr  = mb / np.mean(ground_truth)                    # bias ratio (optional)

    std_pred = np.std(prediction, ddof=1)


    r, p = pearsonr(ground_truth, prediction)

    print(f"r = {r:.3f},  p = {p:.3g}") 
    print(f"Pearson r  : {rho:.4f}")
    print(f"Mean bias  : {mb:.3f}")
    print(f"Bias ratio : {mbr:.3%}")
    print(f"Std (pred) : {std_pred:.3f}")


if __name__ == "__main__":
    main()