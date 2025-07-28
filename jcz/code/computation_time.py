import re

def compute_avg_computation_time(log_file_path):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    pattern = r"Computation time:\s*([\d.]+)s"
    times = []

    for line in lines:
        match = re.search(pattern, line)
        if match:
            times.append(float(match.group(1)))

    if times:
        avg_time = sum(times) / len(times)
        print(f"共提取 {len(times)} 条记录")
        print(f"平均 Computation time: {avg_time:.4f} 秒")
    else:
        print("未在日志中找到 Computation time")

# 替换为你的实际日志文件路径
log_path = "/home/featurize/Bench2Drive/leaderboard/jcz_data/1_jcz_bench2drive220_uniad_speedlimit_1_4_uniad_tensorrt_prev_q_traj.log"
compute_avg_computation_time(log_path)
