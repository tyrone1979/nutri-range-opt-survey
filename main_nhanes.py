import os
import time
from nutriswap_nhanes import *

if __name__ == '__main__':
    print("=" * 70)
    print("NutriSwap-GA v4: 超越 RVEA")
    print("=" * 70)

    paths = {
        'nhanes': ['./data/nhanes.csv', '/mnt/agents/upload/nhanes.csv'],
        'food_type': ['./data/food_type.csv', '/mnt/agents/upload/food_type.csv'],
        'target': ['./data/target.csv', '/mnt/agents/upload/target.csv'],
    }
    found = {}
    for k, cs in paths.items():
        for p in cs:
            if os.path.exists(p):
                found[k] = p
                break
    if not all(k in found for k in ['nhanes', 'food_type', 'target']):
        print("Error: Missing data files!")
        exit(1)

    matrix, meal_tags, nutrients, all_targets = load_data(
        found['nhanes'], found['food_type'], found['target'])
    n_foods = len(matrix)
    n_targets = len(all_targets)
    print(f"\nTotal targets: {n_targets}")

    NUM_TARGETS_TO_TEST = min(5, n_targets)
    print(f"\nTesting first {NUM_TARGETS_TO_TEST} targets...")

    all_targets_results = []

    for target_idx in range(NUM_TARGETS_TO_TEST):
        target_info = all_targets[target_idx]
        ta, ci = target_info['target_array'], target_info['col_idx']

        print(f"\n{'=' * 70}")
        print(f"Target {target_idx + 1}/{NUM_TARGETS_TO_TEST} (idx={target_info['idx']})")
        print(f"Nutrients: {target_info['nutrients']}")
        print(f"Ranges: {ta.tolist()}")

        all_valid = prefilter(matrix, ta, ci)
        print(f"Pre-filtered: {len(all_valid)}/{n_foods} foods")

        if len(all_valid) < MIN_FOODS:
            print("  Not enough valid foods, skipping")
            continue

        max_categories = compute_max_categories(all_valid)
        print(f"  Max food categories: {max_categories}")

        b_count = np.sum(meal_tags[all_valid] == 0)
        l_count = np.sum(meal_tags[all_valid] == 1)
        d_count = np.sum(meal_tags[all_valid] == 2)
        print(f"  b/l/d: {b_count}/{l_count}/{d_count}")

        if b_count == 0 or l_count == 0 or d_count == 0:
            print("  Missing meal types, skipping")
            continue

        def _rand_ind():
            ind = random_individual(all_valid, meal_tags)
            if ind is None:
                ind = np.random.choice(all_valid, MIN_FOODS, replace=False)
            return ind

        # 安全调用
        fc = 0
        for _ in range(100):
            ind = _rand_ind()
            if ind is not None and len(ind) > 0:
                try:
                    _, _, feasible, _ = evaluate_single(ind, matrix, ta, ci, meal_tags, max_categories)
                    if feasible:
                        fc += 1
                except Exception:
                    pass
        print(f"Random feasibility: {fc}%")

        print(f"\nRunning algorithms (pop={POP_SIZE}, gen={N_GENERATIONS})")
        print(f"{'Algorithm':<15} {'FFG':>6} {'FHR':>6} {'FSC':>6} {'BestFit':>8} {'Diversity':>9} {'N_Foods':>7} {'Evals':>8} {'Time(s)':>8}")
        print(f"{'-' * 84}")

        algorithms = {
            'NutriSwap': lambda: NutriSwapGA(matrix, meal_tags, all_valid, max_categories, debug=True).run(ta, ci),
            'BMOEA': lambda: BMOEA(matrix, meal_tags, all_valid, max_categories, debug=False).run(ta, ci),
            'ACMO-Diet': lambda: ACMODiet(matrix, meal_tags, all_valid, max_categories, debug=False).run(ta, ci),
        }

        if PYMOO_AVAILABLE:
            pymoo_algos = ['RVEA']
            for name in pymoo_algos:
                algorithms[name] = lambda n=name: [run_pymoo_algorithm(
                    n, matrix, ta, ci, meal_tags, all_valid, max_categories,
                    pop_size=POP_SIZE, n_generations=N_GENERATIONS, debug=False)]

        target_results = {}
        for name, algo_fn in algorithms.items():
            t0 = time.time()
            try:
                result = algo_fn()
                if result is None or len(result) == 0:
                    print(f"{name:<15} ERROR: No result returned")
                    continue
                r = result[0]
                elapsed = time.time() - t0
                target_results[name] = r
                diversity_display = f"{r.get('diversity', 0):8.3f}"
                n_foods_display = f"{r.get('avg_n_foods', 0):6.1f}"
                print(f"{name:<15} {r['ffg']:6.0f} {r['fhr'] * 100:5.0f}% {r['fsc']:5.0f} {r['best_fitness']:8.3f} {diversity_display:>9} {n_foods_display:>7} {r['n_evals']:7.0f} {elapsed:7.1f}")
            except Exception as e:
                print(f"{name:<15} ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nSummary for Target {target_idx + 1}:")
        for name, r in target_results.items():
            print(f"  {name:<12}: FFG={r['ffg']:5.0f} FHR={r['fhr'] * 100:5.0f}% BestFit={r['best_fitness']:.3f}")

        all_targets_results.append({
            'target_idx': target_idx + 1,
            'target_nutrients': target_info['nutrients'],
            'results': target_results
        })

        print(f"{'=' * 70}")

    if len(all_targets_results) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY ACROSS ALL TARGETS")
        print(f"{'=' * 70}")

        algo_avg = {}
        for target_result in all_targets_results:
            for name, r in target_result['results'].items():
                if name not in algo_avg:
                    algo_avg[name] = {'ffg': [], 'fsc': [], 'best_fitness': [], 'diversity': [], 'avg_n_foods': []}
                algo_avg[name]['ffg'].append(r['ffg'])
                algo_avg[name]['fsc'].append(r['fsc'])
                algo_avg[name]['best_fitness'].append(r['best_fitness'])
                algo_avg[name]['diversity'].append(r.get('diversity', 0))
                algo_avg[name]['avg_n_foods'].append(r.get('avg_n_foods', 0))

        print(f"\n{'Algorithm':<15} {'Avg FFG':>10} {'Avg FSC':>10} {'Avg BestFit':>12} {'Avg Diversity':>14} {'Avg N_Foods':>12}")
        print(f"{'-' * 73}")
        for name in sorted(algo_avg.keys()):
            avg_ffg = np.mean(algo_avg[name]['ffg'])
            avg_fsc = np.mean(algo_avg[name]['fsc'])
            avg_fit = np.mean(algo_avg[name]['best_fitness'])
            avg_div = np.mean(algo_avg[name]['diversity'])
            avg_nf = np.mean(algo_avg[name]['avg_n_foods'])
            print(f"{name:<15} {avg_ffg:10.1f} {avg_fsc:10.1f} {avg_fit:12.3f} {avg_div:14.3f} {avg_nf:12.1f}")

        print(f"{'=' * 70}")

    print("\nAll targets completed!")