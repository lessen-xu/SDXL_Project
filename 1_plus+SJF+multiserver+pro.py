import simpy
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import csv

# =========================
# Step 1: Reading Data
# =========================

def load_and_process_data(filename="measurement_results.json"):
    """
    Extract from measurement_results.json:
    - high_latency: Delay distribution for the high-quality mode (combining simple / comic / complex)
    - fast_latency: Delay distribution for the fast mode (combining simple / comic / complex)
    - high_quality: Average CLIP score for high quality
    - fast_quality: Average CLIP score for fast mode
    """
    with open(filename, 'r') as f:
        results = json.load(f)

    def cfg(name):
        return results[name]

    # high quality
    high_cfgs = [
        cfg("config_high_simple"),
        cfg("config_high_comic"),
        cfg("config_high_complex"),
    ]
    # fast mode
    fast_cfgs = [
        cfg("config_fast_simple"),
        cfg("config_fast_comic"),
        cfg("config_fast_complex"),
    ]

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

    high_quality_score = float(np.mean(high_q))
    fast_quality_score = float(np.mean(fast_q))

    print(f"Data loaded: high_dist (n={len(high_latency_dist)}), fast_dist (n={len(fast_latency_dist)})")
    print(f"Avg High Time: {np.mean(high_latency_dist):.0f} ms | Avg Clip Score: {high_quality_score:.2f}")
    print(f"Avg Fast Time: {np.mean(fast_latency_dist):.0f} ms | Avg Clip Score: {fast_quality_score:.2f}")
    print()

    data_lists = {
        "high_latency": high_latency_dist,
        "fast_latency": fast_latency_dist,
        "high_quality": high_quality_score,
        "fast_quality": fast_quality_score
    }

    return data_lists


# =========================
#  Sample SJF and job
# =========================

def pick_job_profile_for_sjf(data_lists):
    """
    Sampling function for "job size" used by SJF.

    We synthesize three types of tasks:
    - small: Shorter than normal fast, simulating 64x64 icons
    - normal: Normal fast task
    - large: High-quality heavy task

    Returns:
        service_time_ms: Sampled service time (also used as priority for PriorityResource)
        quality_score: Quality score
        size_label: 'small' | 'normal' | 'large'
    """
    u = random.random()

    if u < 0.5:
        # 50% small task: Reduce the fast delay to 1/4.
        base = random.choice(data_lists["fast_latency"])
        service_time_ms = base * 0.25
        quality_score = data_lists["fast_quality"] - 1.0
        size_label = "small"
    elif u < 0.85:
        # 35% general fast task
        service_time_ms = random.choice(data_lists["fast_latency"])
        quality_score = data_lists["fast_quality"]
        size_label = "normal"
    else:
        # 15% huge task: Use high delay
        service_time_ms = random.choice(data_lists["high_latency"])
        quality_score = data_lists["high_quality"]
        size_label = "large"

    return service_time_ms, quality_score, size_label


# =========================
# Step 3: Arrival & Service
# =========================

def request_source(env, lambda_rate, base_policy, server, data_lists, results_collector):
    """
    Arrival process corresponding to high / fast / smart.
    lambda_rate: 1/ms (because our time unit is ms)
    """
    request_id = 0
    while True:
        inter_arrival_time = random.expovariate(lambda_rate)
        yield env.timeout(inter_arrival_time)
        request_id += 1

        env.process(
            gpu_service(
                env,
                f"Req-{request_id}",
                base_policy,
                server,
                data_lists,
                results_collector
            )
        )


def gpu_service(env, req_id, base_policy, server, data_lists, results_collector):
    """
    FCFS service process, corresponding to base_policy in {high, fast, smart}.
    """
    arrival_time = env.now

    with server.request() as req:
        yield req

        service_time = 0.0
        quality_score = 0.0

        # Smart policy: Decide high / fast based on queue length
        if base_policy == "smart":
            if len(server.queue) > 5:
                service_time = random.choice(data_lists["fast_latency"])
                quality_score = data_lists["fast_quality"]
            else:
                service_time = random.choice(data_lists["high_latency"])
                quality_score = data_lists["high_quality"]

        elif base_policy == "high":
            service_time = random.choice(data_lists["high_latency"])
            quality_score = data_lists["high_quality"]

        elif base_policy == "fast":
            service_time = random.choice(data_lists["fast_latency"])
            quality_score = data_lists["fast_quality"]

        else:
            raise ValueError(f"Unknown base_policy in gpu_service: {base_policy}")

        yield env.timeout(service_time)

        total_latency = env.now - arrival_time
        results_collector.append((total_latency, quality_score))


def request_source_sjf(env, lambda_rate, server, data_lists, results_collector):
    """
    Arrival process for SJF policy: Each job samples its own service_time and quality upon arrival,
    using service_time as priority.
    """
    request_id = 0
    while True:
        inter_arrival_time = random.expovariate(lambda_rate)
        yield env.timeout(inter_arrival_time)
        request_id += 1

        service_time_ms, quality_score, size_label = pick_job_profile_for_sjf(data_lists)

        env.process(
            gpu_service_sjf(
                env,
                f"SJF-Req-{request_id}",
                server,
                results_collector,
                service_time_ms,
                quality_score,
                size_label,
            )
        )


def gpu_service_sjf(env, req_id, server, results_collector,
                    service_time_ms, quality_score, size_label):
    """
    SJF service process:
    - Uses PriorityResource
    - priority = service_time_ms (shorter jobs are served earlier)
    """
    arrival_time = env.now

    with server.request(priority=service_time_ms) as req:
        yield req

        yield env.timeout(service_time_ms)

        total_latency = env.now - arrival_time
        results_collector.append((total_latency, quality_score))


# =========================
# Step 4: Single Simulation Run (with multi-GPU support)
# =========================

def run_simulation(lambda_rate, base_policy, capacity, data_lists, simulation_time):
    """
    Run one complete simulation.

    Parameters:
        lambda_rate: Arrival rate (unit: 1/ms)
        base_policy: 'high' / 'fast' / 'smart' / 'sjf'
        capacity: Number of GPUs / servers
        data_lists: Measurement data
        simulation_time: Total simulation time (ms)

    Returns:
        p99_latency_ms
        avg_latency_ms
        avg_quality
        throughput_req_per_sec
    """
    results_collector = []
    env = simpy.Environment()

    if base_policy == "sjf":
        server = simpy.PriorityResource(env, capacity=capacity)
        env.process(request_source_sjf(env, lambda_rate, server, data_lists, results_collector))
    else:
        server = simpy.Resource(env, capacity=capacity)
        env.process(request_source(env, lambda_rate, base_policy, server, data_lists, results_collector))

    env.run(until=simulation_time)

    warmup_count = len(results_collector) // 10
    stable_results = results_collector[warmup_count:]

    if not stable_results or len(stable_results) < 50:
        print(f"Warning: Not enough data: lambda={lambda_rate}, policy={base_policy}, cap={capacity}, "
              f"n={len(stable_results)}")
        return 0.0, 0.0, 0.0, 0.0

    stable_sim_time_ms = simulation_time * 0.9

    latencies = np.array([res[0] for res in stable_results], dtype=float)
    qualities = np.array([res[1] for res in stable_results], dtype=float)

    p99_latency = float(np.percentile(latencies, 99))
    avg_latency = float(np.mean(latencies))
    avg_quality = float(np.mean(qualities))

    throughput_req_per_sec = len(stable_results) / (stable_sim_time_ms / 1000.0)

    return p99_latency, avg_latency, avg_quality, throughput_req_per_sec


# =========================
# Step 5: Main Program + Plotting +CSV
# =========================

def format_ms_axis(ax):
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylim(bottom=0)


def main():
    print("Running multi-policy, multi-server simulation (Approach B: policy key encodes capacity).")

    json_file = "measurement_results.json"

    try:
        data_lists = load_and_process_data(json_file)
    except FileNotFoundError:
        print(f"Error: '{json_file}' not found.")
        print("Please make sure the real data file is in the same directory as this script.")
        sys.exit(1)

    # We use the previous lambda (unit: 1/ms), approximately 0.00001~0.00005
    lambda_values = np.linspace(0.00001, 0.00005, 10)
    SIMULATION_TIME = 5 * 1000 * 1000  # 5e6 ms ≈ 5000 seconds

    print(f"\n--- Simulation Settings ---")
    print(f"Simulation Time per run: {SIMULATION_TIME / 1000:.0f} seconds")
    print(f"Lambda values (1/ms): {lambda_values}")
    print()

    # base policies (without GPU count)
    base_policies = {
        "high":  "Static High Quality",
        "fast":  "Static Fast",
        "smart": "Smart Adaptive",
        "sjf":   "SJF (Small Jobs First)",
    }

    # GPU counts (capacity) we are interested in
    capacities = [1, 2, 4]

    # Approach B: Combine "base_policy + capacity" into policy_key
    # e.g.: fast_c1 / fast_c2 / fast_c4 / sjf_c1 / sjf_c4 ...
    policies_to_run = {}
    for base_key, base_name in base_policies.items():
        for cap in capacities:
            key = f"{base_key}_c{cap}"
            human_name = f"{base_name}, {cap} GPU(s)"
            policies_to_run[key] = {
                "base_policy": base_key,
                "capacity": cap,
                "label": human_name
            }

    # Prepare to store results
    plot_p99 = {key: [] for key in policies_to_run}
    plot_avg = {key: [] for key in policies_to_run}
    plot_thr = {key: [] for key in policies_to_run}
    plot_qual = {key: [] for key in policies_to_run}

    print("--- Running simulations for all policies (B: base_policy + capacity) ---")
    for lam in lambda_values:
        print(f"\nLambda = {lam:.6f}")
        for policy_key, meta in policies_to_run.items():
            base_policy = meta["base_policy"]
            capacity = meta["capacity"]

            p99_lat, avg_lat, avg_qual, throughput = run_simulation(
                lam, base_policy, capacity, data_lists, SIMULATION_TIME
            )

            plot_p99[policy_key].append(p99_lat)
            plot_avg[policy_key].append(avg_lat)
            plot_thr[policy_key].append(throughput)
            plot_qual[policy_key].append(avg_qual)

            print(
                f"  - {meta['label']:<32}: "
                f"P99={p99_lat:,.0f}ms | Avg={avg_lat:,.0f}ms | "
                f"Thr={throughput:.4f} req/s | Q={avg_qual:.2f}"
            )

    print("\n--- Simulation Complete ---\n")


    print("--- Saving results to CSV ---")
    csv_filename = "simulation_results_flat.csv"

    # Prepare data for CSV writing
    header = [
        "lambda",
        "policy_key",
        "base_policy",
        "capacity",
        "p99_latency_ms",
        "avg_latency_ms",
        "throughput_req_s",
        "avg_quality"
    ]
    all_rows = [header]

    # Iterate over lambda_values (as x-axis)
    for i, lam in enumerate(lambda_values):
        # Iterate over each policy (policy_key)
        for policy_key, meta in policies_to_run.items():
            base_policy = meta["base_policy"]
            capacity = meta["capacity"]

            # Extract data for the corresponding lambda (index i) from the results dictionary
            row_data = [
                lam,
                policy_key,
                base_policy,
                capacity,
                plot_p99[policy_key][i],
                plot_avg[policy_key][i],
                plot_thr[policy_key][i],
                plot_qual[policy_key][i]
            ]
            all_rows.append(row_data)

    # Write to CSV file
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_rows)
        print(f"Successfully saved flat results to {csv_filename}")
    except IOError as e:
        print(f"Error: Could not write to CSV file {csv_filename}. Reason: {e}")
    except PermissionError:
        print(f"Error: Permission denied. Could not write to {csv_filename}.")
        print("Please check if the file is open in another program (like Excel).")
    print()  # Add a blank line

    # ========= Plotting: Too many lines are messy, so we plot grouped by "base_policy" =========

    # 1. For each base_policy, plot three lines (capacity=1/2/4) in one figure (P99 vs lambda)
    for base_key, base_name in base_policies.items():
        plt.figure(figsize=(9, 6))
        for cap in capacities:
            key = f"{base_key}_c{cap}"
            label = f"{base_name}, {cap} GPU(s)"
            plt.plot(lambda_values, plot_p99[key], marker='o', label=label)

        plt.xlabel("Arrival Rate $\\lambda$ (1/ms, higher = heavier load)")
        plt.ylabel("P99 Latency (ms)")
        plt.title(f"P99 Latency vs Load for {base_name} (different GPU counts)")
        plt.grid(True, linestyle='--', alpha=0.6)
        format_ms_axis(plt.gca())
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"policy_{base_key}_P99_vs_lambda_by_capacity.png")

    # 2. Plot a dedicated "Avg Latency vs lambda" multi-GPU graph for SJF (can be used directly in a paper)
    plt.figure(figsize=(9, 6))
    for cap in capacities:
        key = f"sjf_c{cap}"
        label = f"SJF, {cap} GPU(s)"
        plt.plot(lambda_values, plot_avg[key], marker='o', label=label)
    plt.xlabel("Arrival Rate $\\lambda$ (1/ms)")
    plt.ylabel("Average Latency (ms)")
    plt.title("SJF: Average Latency vs Load (different GPU counts)")
    plt.grid(True, linestyle='--', alpha=0.6)
    format_ms_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig("sjf_Avg_vs_lambda_by_capacity.png")

    # 3. Fix one lambda point, see how Avg Latency / Throughput changes with capacity
    lambda_mid = lambda_values[len(lambda_values) // 2]

    def find_index_of_lambda(target):
        # Since it's linearly generated, just find the closest index here
        arr = np.array(lambda_values)
        return int(np.argmin(np.abs(arr - target)))

    idx_mid = find_index_of_lambda(lambda_mid)

    # (a) Average latency vs capacity
    plt.figure(figsize=(9, 6))
    for base_key, base_name in base_policies.items():
        avg_list = [plot_avg[f"{base_key}_c{cap}"][idx_mid] for cap in capacities]
        plt.plot(capacities, avg_list, marker='o', label=base_name)
    plt.xlabel("Number of GPUs (capacity)")
    plt.ylabel("Average Latency (ms)")
    plt.title(f"Average Latency vs GPU Count (lambda={lambda_values[idx_mid]:.5f})")
    plt.grid(True, linestyle='--', alpha=0.6)
    format_ms_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig("capacity_AvgLatency_vs_capacity.png")

    # (b) Throughput vs capacity
    plt.figure(figsize=(9, 6))
    for base_key, base_name in base_policies.items():
        thr_list = [plot_thr[f"{base_key}_c{cap}"][idx_mid] for cap in capacities]
        plt.plot(capacities, thr_list, marker='o', label=base_name)
    plt.xlabel("Number of GPUs (capacity)")
    plt.ylabel("Throughput (req/s)")
    plt.title(f"Throughput vs GPU Count (lambda={lambda_values[idx_mid]:.5f})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("capacity_Throughput_vs_capacity.png")

    plt.show()


if __name__ == "__main__":
    try:
        import simpy  # noqa: F401
    except ImportError:
        print("Error: 'simpy' library not found. Please install with: pip install simpy")
        sys.exit(1)

    main()