"""
Baseline 算法对比模块
包含三种基线方法:
1. RandomSearch: 纯随机搜索
2. GreedySearch: 贪心局部搜索
3. SimulatedAnnealing: 模拟退火
"""

import numpy as np
import copy


class RandomSearch:
    """基线1: 纯随机搜索 - 随机生成大量方案，取最优"""

    def __init__(self, ga_engine, target_ranges):
        self.ga = ga_engine
        self.target_ranges = target_ranges

    def search(self, n_trials=1000, breakfast_size=3, lunch_size=3, dinner_size=3):
        """随机生成n_trials个方案，返回最佳"""
        best_meal = None
        best_fitness = -1

        for _ in range(n_trials):
            meal = self.ga.create_random_daily_meal(breakfast_size, lunch_size, dinner_size)
            fitness, _, _ = self.ga.evaluate_meal(meal, self.target_ranges)

            if fitness > best_fitness:
                best_fitness = fitness
                best_meal = meal

        return best_meal, best_fitness


class GreedySearch:
    """基线2: 贪心局部搜索 - 从一个随机解出发，逐步替换为更优的相似食物"""

    def __init__(self, ga_engine, target_ranges):
        self.ga = ga_engine
        self.target_ranges = target_ranges

    def search(self, n_iterations=100, breakfast_size=3, lunch_size=3, dinner_size=3):
        """贪心搜索: 每次替换一个食物为更优的相似食物"""
        # 初始随机解
        meal = self.ga.create_random_daily_meal(breakfast_size, lunch_size, dinner_size)
        fitness, _, _ = self.ga.evaluate_meal(meal, self.target_ranges)

        for _ in range(n_iterations):
            improved = False

            # 遍历所有餐次的所有食物
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                for i in range(len(meal[meal_type])):
                    food_id = meal[meal_type][i]
                    similar_foods = self.ga.similar_foods.get(food_id, [])

                    # 尝试每个相似食物
                    for sim_food_id, _ in similar_foods[:5]:
                        new_meal = copy.deepcopy(meal)
                        new_meal[meal_type][i] = sim_food_id
                        new_meal['total_nutrients'] = self.ga._calculate_total_nutrients(new_meal)

                        new_fitness, _, _ = self.ga.evaluate_meal(new_meal, self.target_ranges)

                        if new_fitness > fitness:
                            meal = new_meal
                            fitness = new_fitness
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if not improved:
                break  # 局部最优，停止

        return meal, fitness


class SimulatedAnnealing:
    """基线3: 模拟退火 - 带温度控制的随机搜索"""

    def __init__(self, ga_engine, target_ranges):
        self.ga = ga_engine
        self.target_ranges = target_ranges

    def search(self, n_iterations=1000, initial_temp=1.0, cooling_rate=0.995,
               breakfast_size=3, lunch_size=3, dinner_size=3):
        """模拟退火搜索"""
        # 初始解
        current_meal = self.ga.create_random_daily_meal(breakfast_size, lunch_size, dinner_size)
        current_fitness, _, _ = self.ga.evaluate_meal(current_meal, self.target_ranges)

        best_meal = copy.deepcopy(current_meal)
        best_fitness = current_fitness

        temp = initial_temp

        for _ in range(n_iterations):
            # 生成邻居: 随机替换一个食物为相似食物
            new_meal = copy.deepcopy(current_meal)
            meal_type = np.random.choice(['breakfast', 'lunch', 'dinner'])
            pos = np.random.randint(0, len(new_meal[meal_type]))
            food_id = new_meal[meal_type][pos]

            similar_foods = self.ga.similar_foods.get(food_id, [])
            if similar_foods:
                new_food_id = np.random.choice([f for f, _ in similar_foods[:5]])
                new_meal[meal_type][pos] = new_food_id
                new_meal['total_nutrients'] = self.ga._calculate_total_nutrients(new_meal)

                new_fitness, _, _ = self.ga.evaluate_meal(new_meal, self.target_ranges)

                # 接受准则
                delta = new_fitness - current_fitness
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    current_meal = new_meal
                    current_fitness = new_fitness

                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_meal = copy.deepcopy(current_meal)

            temp *= cooling_rate

        return best_meal, best_fitness
