import simpy
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import csv  # [新增] 用于导出CSV


# =========================
# Step 1: 读取测量数据 (保持不变)
# =========================

def load_and_process_data(filename="measurement_results.json"):
    with open(filename, 'r') as f:
        results = json.load(f)

    def cfg(name):
        return results[name]

    # 提取 High/Fast 配置数据
    high_cfgs = [cfg("config_high_simple"), cfg("config_high_comic"), cfg("config_high_complex")]
    fast_cfgs = [cfg("config_fast_simple"), cfg("config_fast_comic"), cfg("config_fast_complex")]

    high_latency_dist = []
    fast_latency_dist = []
    high_q = []
    fast_q = []

    for c in high_cfgs:
        high_latency_dist.extend(c["latency_distribution_ms"])
        high_q.append(c["quality_clip_score"])

    for c in fast_cfgs:
        fast_latency_dist.extend(c["latency_distribution_ms"])
        fast_q.append(c["quality_clip_score"])

    data_lists = {
        "high_latency": high_latency_dist,
        "fast_latency": fast_latency_dist,
        "high_quality": float(np.mean(high_q)),
        "fast_quality": float(np.mean(fast_q))
    }
    return data_lists


# ========================================================
# Autoscaler Configuration
# ========================================================
AUTOSCALER_CONFIG = {
    "MIN_GPUS": 1,
    "MAX_GPUS": 4,
    "CHECK_INTERVAL_MS": 1 * 1000,  # 每1秒检查一次
    "SCALE_DOWN_COOLDOWN_PERIODS": 60,  # 连续空闲60个周期才缩容
}


# =========================
# Step 2: Job Sampling (SJF Logic)
# =========================

def pick_job_profile_for_sjf(data_lists):
    u = random.random()
    if u < 0.5:
        # Small jobs
        base = random.choice(data_lists["fast_latency"])
        return base * 0.25, data_lists["fast_quality"] - 1.0, "small"
    elif u < 0.85:
        # Normal jobs
        return random.choice(data_lists["fast_latency"]), data_lists["fast_quality"], "normal"
    else:
        # Large jobs
        return random.choice(data_lists["high_latency"]), data_lists["high_quality"], "large"


# =========================
# Step 3: Service Logic
# =========================

# --- 1. Static SJF Logic ---
def request_source_sjf(env, lambda_rate, server, data_lists, results_collector):
    request_id = 0
    while True:
        inter_arrival_time = random.expovariate(lambda_rate)
        yield env.timeout(inter_arrival_time)
        request_id += 1
        service_time, quality, size = pick_job_profile_for_sjf(data_lists)
        env.process(gpu_service_sjf(env, server, results_collector, service_time, quality))


def gpu_service_sjf(env, server, results_collector, service_time, quality):
    arrival_time = env.now
    with server.request(priority=service_time) as req:
        yield req
        yield env.timeout(service_time)
        results_collector.append((env.now - arrival_time, quality))


# --- 2. Dynamic SJF Logic ---

def auto_scaler(env, gpu_resource, capacity_gate, usage_log):
    """
    后台自动伸缩进程。
    记录状态到 usage_log。
    """
    cfg = AUTOSCALER_CONFIG
    scale_down_counter = 0

    # 记录初始状态
    usage_log.append((env.now, capacity_gate.level))

    while True:
        yield env.timeout(cfg["CHECK_INTERVAL_MS"])

        # 负载定义：正在处理 + 排队中
        total_jobs = gpu_resource.count + len(gpu_resource.queue)
        current_cap = capacity_gate.level

        # 扩容逻辑 (Scale Up)
        if total_jobs > current_cap and current_cap < cfg["MAX_GPUS"]:
            yield capacity_gate.put(1)
            usage_log.append((env.now, capacity_gate.level))  # 记录变化
            scale_down_counter = 0

            # 缩容逻辑 (Scale Down)
        elif total_jobs < current_cap and current_cap > cfg["MIN_GPUS"]:
            scale_down_counter += 1
            if scale_down_counter >= cfg["SCALE_DOWN_COOLDOWN_PERIODS"]:
                yield capacity_gate.get(1)
                usage_log.append((env.now, capacity_gate.level))  # 记录变化
                scale_down_counter = 0
        else:
            if total_jobs >= current_cap:
                scale_down_counter = 0


def request_source_sjf_dynamic(env, lambda_rate, gpu_resource, capacity_gate, data_lists, results_collector):
    request_id = 0
    while True:
        inter_arrival_time = random.expovariate(lambda_rate)
        yield env.timeout(inter_arrival_time)
        request_id += 1
        service_time, quality, size = pick_job_profile_for_sjf(data_lists)

        env.process(gpu_service_sjf_dynamic(
            env, gpu_resource, capacity_gate, results_collector, service_time, quality
        ))


def gpu_service_sjf_dynamic(env, gpu_resource, capacity_gate, results_collector, service_time, quality):
    arrival_time = env.now

    # 1. 获取容量许可
    yield capacity_gate.get(1)

    try:
        # 2. 获取实际 GPU
        with gpu_resource.request(priority=service_time) as req:
            yield req
            # 3. 执行任务
            yield env.timeout(service_time)
    finally:
        # 4. 归还许可
        capacity_gate.put(1)

    results_collector.append((env.now - arrival_time, quality))


# =========================
# Step 4: Simulation Runner
# =========================

def run_simulation(lambda_rate, policy_type, capacity, data_lists, simulation_time):
    results_collector = []
    env = simpy.Environment()
    gpu_usage_log = []  # 存储时间序列数据
    avg_gpu_usage = 0.0

    if policy_type == "sjf":
        # 静态策略
        server = simpy.PriorityResource(env, capacity=capacity)
        env.process(request_source_sjf(env, lambda_rate, server, data_lists, results_collector))
        avg_gpu_usage = capacity
        # 对于静态，日志只有开始和结束
        gpu_usage_log = [(0, capacity), (simulation_time, capacity)]

    elif policy_type == "sjf_dynamic":
        # 动态策略
        gpu_resource = simpy.PriorityResource(env, capacity=AUTOSCALER_CONFIG["MAX_GPUS"])
        capacity_gate = simpy.Container(env,
                                        capacity=AUTOSCALER_CONFIG["MAX_GPUS"],
                                        init=AUTOSCALER_CONFIG["MIN_GPUS"])

        env.process(
            request_source_sjf_dynamic(env, lambda_rate, gpu_resource, capacity_gate, data_lists, results_collector))
        env.process(auto_scaler(env, gpu_resource, capacity_gate, gpu_usage_log))

    env.run(until=simulation_time)

    # --- 统计计算 ---
    warmup_time = simulation_time * 0.1
    stable_results = results_collector[int(len(results_collector) * 0.1):]

    if not stable_results:
        return 0, 0, 0, 0, []

    # 计算加权平均 GPU 使用数
    if policy_type == "sjf_dynamic" and gpu_usage_log:
        stable_log = [entry for entry in gpu_usage_log if entry[0] >= warmup_time]
        if not stable_log:
            stable_log = [(warmup_time, gpu_usage_log[-1][1])] if gpu_usage_log else [(warmup_time, 1)]

        weighted_sum = 0
        for i in range(len(stable_log) - 1):
            dur = stable_log[i + 1][0] - stable_log[i][0]
            weighted_sum += stable_log[i][1] * dur
        weighted_sum += stable_log[-1][1] * (simulation_time - stable_log[-1][0])
        avg_gpu_usage = weighted_sum / (simulation_time - warmup_time)

    latencies = [r[0] for r in stable_results]
    qualities = [r[1] for r in stable_results]

    p99_lat = np.percentile(latencies, 99)
    avg_lat = np.mean(latencies)
    avg_qual = np.mean(qualities)

    # 返回 log 作为第5个元素
    return p99_lat, avg_lat, avg_qual, avg_gpu_usage, gpu_usage_log


# =========================
# Step 5: Main & Plotting
# =========================

def format_ms_axis(ax):
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylim(bottom=0)


def main():
    print("Running Dynamic Autoscaler Simulation...")

    try:
        data_lists = load_and_process_data("measurement_results.json")
    except FileNotFoundError:
        print("Error: 'measurement_results.json' not found.")
        sys.exit(1)

    # 1. Main Loop for Comparison (Latency vs Lambda)
    lambda_values = np.linspace(0.00001, 0.00005, 10)
    SIMULATION_TIME = 10 * 1000 * 1000  # 10,000 seconds

    scenarios = [
        {"key": "sjf_c1", "policy": "sjf", "cap": 1, "label": "Static SJF (1 GPU)"},
        {"key": "sjf_c2", "policy": "sjf", "cap": 2, "label": "Static SJF (2 GPUs)"},
        {"key": "sjf_c4", "policy": "sjf", "cap": 4, "label": "Static SJF (4 GPUs)"},
        {"key": "dynamic", "policy": "sjf_dynamic", "cap": 4, "label": "Dynamic SJF (1-4 GPUs)"}
    ]

    results = {s["key"]: {"avg_lat": [], "avg_gpu": []} for s in scenarios}

    print("Step 1: Running Load Sweep Simulation...")
    for lam in lambda_values:
        print(f"  > Simulating Lambda = {lam:.6f}")
        for s in scenarios:
            _, avg_lat, _, avg_gpu, _ = run_simulation(
                lam, s["policy"], s["cap"], data_lists, SIMULATION_TIME
            )
            results[s["key"]]["avg_lat"].append(avg_lat)
            results[s["key"]]["avg_gpu"].append(avg_gpu)

    # 2. Generate Comparison Plots
    print("\nStep 2: Generating Comparison Plots (PNG)...")

    # --- Plot 1: Latency Comparison ---
    plt.figure(figsize=(10, 6))
    for s in scenarios[:-1]:
        plt.plot(lambda_values, results[s["key"]]["avg_lat"], marker='o', alpha=0.5, label=s["label"])

    dyn_key = "dynamic"
    plt.plot(lambda_values, results[dyn_key]["avg_lat"],
             marker='s', linewidth=3, linestyle='--', color='red', label=scenarios[-1]["label"])

    plt.xlabel(r"Arrival Rate $\lambda$ (1/ms)")
    plt.ylabel("Average Latency (ms)")
    plt.title("Latency Comparison: Static vs Dynamic SJF")
    plt.grid(True, linestyle='--', alpha=0.6)
    format_ms_axis(plt.gca())
    plt.legend()
    plt.savefig("NEW_sjf_static_vs_dynamic_latency.png")

    # --- Plot 2: Average GPU Usage ---
    plt.figure(figsize=(10, 6))
    for s in scenarios[:-1]:
        plt.plot(lambda_values, results[s["key"]]["avg_gpu"], linestyle=':', alpha=0.6, label=s["label"])

    plt.plot(lambda_values, results[dyn_key]["avg_gpu"],
             marker='s', linewidth=3, linestyle='--', color='purple', label="Dynamic Usage (Avg)")

    plt.xlabel(r"Arrival Rate $\lambda$ (1/ms)")
    plt.ylabel("Average Active GPUs")
    plt.title("Resource Efficiency: Dynamic Scaling GPU Usage")
    plt.yticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("NEW_sjf_dynamic_gpu_usage.png")

    # ========================================================
    # Step 6: [MODIFIED] Export Time-Series to CSV
    # ========================================================
    print("\nStep 3: Exporting Time-Series Data to CSV (GPU Usage vs Time)...")

    # 选取一个高负载点 (Lambda=4.0e-5) 来展示频繁的扩缩容细节
    target_lambda = 0.00004
    print(f"  > Running detailed trace for Lambda = {target_lambda:.6f} ...")

    # 运行一次单独的仿真，获取详细的 usage_log
    _, _, _, _, usage_log = run_simulation(
        target_lambda, "sjf_dynamic", 4, data_lists, SIMULATION_TIME
    )

    csv_filename = "gpu_usage_time_series.csv"

    try:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["time_sec", "active_gpus"])

            if usage_log:
                for entry in usage_log:
                    time_ms = entry[0]
                    level = entry[1]
                    # 转换时间单位：毫秒 -> 秒，并保留3位小数
                    writer.writerow([f"{time_ms / 1000.0:.3f}", level])

                # 确保最后时刻的状态也被写入
                writer.writerow([f"{SIMULATION_TIME / 1000.0:.3f}", usage_log[-1][1]])

        print(f"  > Successfully saved: {csv_filename}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

    print("\nAll simulations, plots, and CSV export complete.")


if __name__ == "__main__":
    main()