import subprocess
import os

# 定义checkpoint路径 TODO
BASE_PATH = "/home/olg7848/p32335/my_research/echo_vision/checkpoints/checkpoints_main"
BASELINE_PATH = "/home/olg7848/p32335/my_research/echo_vision/checkpoints/checkpoints_baseline"

# 特征提取器列表
FEAT_EXTRACTORS = ["vivit", "videoresnet", "vanilla_videomae", "echo_videomae"]
# 任务类型列表
TASK_TYPES = ["ef_regression", "esv_regression", "edv_regression", "ef_classification"]
# STFF标志列表
STFF_NET_FLAGS = [True, False]

# 用于存储检查点路径的字典
CHECKPOINT_PATHS = {}

def parse_subdir(subdir):
    parts = subdir.split('_')
    if len(parts) == 10:    # 对应 "echo_videomae" 的checkpoints
        feat_extractor = parts[3] + '_' + parts[4]
        task_type = parts[7] + '_' + parts[8]
        stff_flag_str = parts[9]
        
    elif len(parts) == 8:   # 对应 "vivit", "videoresnet" 的checkpoints
        feat_extractor = parts[3]
        task_type = parts[6] + '_' + parts[7]
        stff_flag_str = 'stffFalse'
    elif len(parts) == 9:   # 对应 "vanilla_videomae" 的checkpoints
        feat_extractor = parts[3] + '_' + parts[4]
        task_type = parts[7] + '_' + parts[8]
        stff_flag_str = 'stffFalse'
    else:
        return None, None, None, None

    stff_flag = (stff_flag_str == 'stffTrue')
    return feat_extractor, task_type, stff_flag

def collect_checkpoint_paths():
    paths = [BASE_PATH, BASELINE_PATH]
    for path in paths:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for subdir in subdirs:
            feat_extractor, task_type, stff_flag = parse_subdir(subdir)
            if feat_extractor and task_type and stff_flag in STFF_NET_FLAGS:
                key = f"{feat_extractor}_{task_type}_{'True' if stff_flag else 'False'}"
                CHECKPOINT_PATHS[key] = os.path.join(path, subdir)

def execute_commands():
    base_command = ["accelerate", "launch", "--config_file", "./config.yaml", "test.py"]
    for feat_extractor in FEAT_EXTRACTORS:
        for task_type in TASK_TYPES:
            for stff_net_flag in STFF_NET_FLAGS:
                checkpoint_path = CHECKPOINT_PATHS.get(f"{feat_extractor}_{task_type}_{stff_net_flag}")
                if checkpoint_path:
                    command = (
                        base_command +
                        ["--dataset_name", "echonet_dynamic"] +
                        ["--feat_extractor", feat_extractor] +
                        ["--task_type", task_type] +
                        ["--checkpoint_path", checkpoint_path] +
                        ["--num_workers", "8"]
                    )
                    if stff_net_flag:
                        command += ["--stff_net_flag"]
                    
                    env = os.environ.copy()
                    # env['HF_ENDPOINT'] = 'https://hf-mirror.com'    # TODO 可以不使用镜像网站 删除env即可
                    print("Executing command:", " ".join(command))
                    subprocess.run(command, env=env, check=True)
                else:
                    print(f"Checkpoint path not found for {feat_extractor} and {task_type} and {stff_net_flag}")

def main():
    collect_checkpoint_paths()  # 收集checkpoints对应目录映射

    # for key, path in CHECKPOINT_PATHS.items():
    #     print(f"{key}: {path}")

    execute_commands()  

if __name__ == "__main__":
    main()