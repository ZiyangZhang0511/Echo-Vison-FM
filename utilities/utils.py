import os
import json
import random
import shutil
from pathlib import Path

from scipy import stats

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

import torch

def remove_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error occurred while removing the directory: {e}")

def compute_metrics(logits, targets, task_type:str, change_scale=False, threshold=0.5):

    if task_type == "ef_classification":
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        predictions = predictions.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        auc = roc_auc_score(targets, probabilities.cpu().detach().numpy())
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
        }

    elif task_type in ["as_classification"]:
        
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        predictions = predictions.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='macro')
        if len(torch.unique(torch.tensor(targets))) != logits.shape[-1]:
            auc = None
        else:
            auc = roc_auc_score(targets, probabilities.cpu().detach().numpy(), multi_class='ovr', average='macro')
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc,
        }

    elif task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        targets = targets.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()

        if change_scale:
            targets = np.exp(targets)
            logits = np.exp(logits)

        mae = mean_absolute_error(targets, logits)
        mse = mean_squared_error(targets, logits)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, logits)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        }


def get_the_most_recent_dir(directory:str):
    all_dirs = [d for d in Path(directory).iterdir() if d.is_dir()]
    sorted_dirs = sorted(all_dirs, key=lambda x: x.stat().st_mtime)
    return str(sorted_dirs[-1].resolve())


def test(
    accelerator,
    model,
    dataloader,
    criterion,
    mode:str,
    args,
    enable_print=True,
    is_save_results=False,
):
    total_loss = 0
    total_logits, total_targets = [], []
    for pixel_values, targets, temporal_indices in dataloader:

        if args.task_type in ["as_classification"]:
            targets = targets.to(args.device)
        else:
            targets = targets.view(-1, 1).to(args.device)
        
        with torch.no_grad():
            logits = model(pixel_values.to(args.device), temporal_indices.to(args.device))
        
        loss = criterion(logits, targets)

        total_loss += loss.item() / len(dataloader)
        total_logits.append(logits)
        total_targets.append(targets)

    total_logits = torch.cat(total_logits).detach().cpu()
    total_targets = torch.cat(total_targets).detach().cpu()

    if args.dataset_name == "echonet_dynamic" and args.task_type in ["esv_regression", "edv_regression"]:
        change_scale = True
    else:
        change_scale = False
    metrics_dict = compute_metrics(total_logits, total_targets, args.task_type, change_scale=change_scale)
    
    if enable_print:
        print(f"{mode} loss: {total_loss}, {mode} metrics: {metrics_dict}")
    
    if is_save_results:
        save_results(total_logits, total_targets, total_loss, metrics_dict, args)

    return total_loss, metrics_dict


def save_results(total_logits, total_targets, total_loss, metrics_dict, args):
    
    if args.feat_extractor == "echo_videomae":
        path = './results/main_formal'
    elif args.feat_extractor in ["vivit", "videoresnet", "vanilla_videomae"]:
        path = './results/baseline_formal'
    
    # create a dict of output data
    output_data = {
        "total_loss": float(np.array(total_loss)),  # 转换为 Python float
        "total_logits": total_logits.tolist(),  # 转换为列表
        "total_targets": total_targets.tolist(),  # 转换为列表
        "metrics": {k: float(v) if isinstance(v, np.float32) else v for k, v in metrics_dict.items()}  # 确保 metrics 中的所有值都是 Python float
    }
    
    # frame save .json filepath
    filename = f"{args.feat_extractor}_{args.dataset_name}_{args.task_type}_stff{args.stff_net_flag}_linearhead{args.linear_prediction_head}_finetune{args.finetune}.json"
    filepath = os.path.join(path, filename)
    
    os.makedirs(path, exist_ok=True)
    try:
        with open(filepath, 'w') as file:
            json.dump(output_data, file, indent=4)  
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def convert_to_float(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 numpy 数组转换为 Python 列表
    elif isinstance(obj, np.generic):
        return obj.item()  # 将 numpy 标量转换为普通 Python 数值
    elif isinstance(obj, dict):
        return {k: convert_to_float(v) for k, v in obj.items()}  # 递归处理字典
    elif isinstance(obj, list):
        return [convert_to_float(x) for x in obj]
    else:
        return obj  # 如果是其他类型，直接返回


def run_multiple_tests(
    model,
    dataloader,
    criterion,
    mode:str,
    args,
    device,
    num_runs=50,
    output_file="metrics_results.json",
    save=False,
    save_for_boxplot=False,
):
    all_metrics = []
    all_losses = []

    all_data = list(dataloader)
    # print(min(args.batch_size, len(all_data)-2), len(all_data))
    for _ in range(num_runs):
        random_batches = random.sample(all_data, min(args.batch_size, len(all_data)-2))
        
        
        loss, metrics_dict = test(None, model, random_batches, criterion, mode, args, enable_print=False)
        all_metrics.append(metrics_dict)
        all_losses.append(loss)
     
    metrics_ci = compute_metrics_ci(all_metrics, all_losses)
    
    if save:
        save_metrics_to_json(metrics_ci, output_file)

    if save_for_boxplot:
        boxplot_dir = "./results/boxplot"
        os.makedirs(boxplot_dir, exist_ok=True)
        save_jsonpath = os.path.join(
            boxplot_dir,
            f"{args.feat_extractor}_{args.dataset_name}_{args.task_type}_stff{args.stff_net_flag}_linearhead{args.linear_prediction_head}_finetune{args.finetune}.json"
        )

        all_metrics = convert_to_float(all_metrics)
        all_losses = convert_to_float(all_losses)

        # save as JSON
        with open(save_jsonpath, 'w') as f:
            json.dump({"metrics": all_metrics, "losses": all_losses}, f, indent=4)
    
    return metrics_ci

def compute_metrics_ci(all_metrics, all_losses, confidence=0.95):
    
    metrics_ci = {}
    
    # calculate ci of loss
    losses_np = np.array(all_losses)
    mean_loss = np.mean(losses_np)
    std_error_loss = np.std(losses_np, ddof=1) / np.sqrt(len(losses_np))
    t_value_loss = stats.t.ppf((1 + confidence) / 2., df=len(losses_np)-1)
    ci_loss = std_error_loss * t_value_loss
    
    metrics_ci['loss'] = {
        'mean': mean_loss,
        'ci': ci_loss,
        'ci_lower': mean_loss - ci_loss,
        'ci_upper': mean_loss + ci_loss,
    }
    
    # get name of metrics
    metric_names = set()
    for metrics in all_metrics:
        metric_names.update(metrics.keys())
    
    # loop for each metric
    for metric_name in metric_names:

        metric_list = [metrics[metric_name] for metrics in all_metrics if metrics.get(metric_name) is not None]
        
        if not metric_list:
            metrics_ci[metric_name] = {
                'mean': None,
                'ci': None,
                'ci_lower': None,
                'ci_upper': None
            }
            continue
        
        metric_values_np = np.array(metric_list)
        
        # compute ci
        mean_value = np.mean(metric_values_np)
        std_error = np.std(metric_values_np, ddof=1) / np.sqrt(len(metric_values_np))
        t_value = stats.t.ppf((1 + confidence) / 2., df=len(metric_values_np)-1)
        ci = std_error * t_value
        
        metrics_ci[metric_name] = {
            'mean': mean_value,
            'ci': ci,
            'ci_lower': mean_value - ci,
            'ci_upper': mean_value + ci,
        }
    
    return metrics_ci


def save_metrics_to_json(metrics_ci, output_file):
    
    def convert_to_float(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为 Python 列表
        elif isinstance(obj, np.generic):
            return obj.item()  # 将 numpy 标量转换为普通 Python 数值
        elif isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}  # 递归处理字典
        else:
            return obj  # 如果是其他类型，直接返回

    # convert all data
    metrics_ci_converted = convert_to_float(metrics_ci)
        
    # save as JSON
    with open(output_file, 'w') as f:
        json.dump(metrics_ci_converted, f, indent=4)
    print(f"Metrics results saved to {output_file}")