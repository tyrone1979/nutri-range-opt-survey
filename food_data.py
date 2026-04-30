"""
NutriSwap-GA 食物数据库模块
包含食物营养素数据、名称映射、餐次分类和目标范围
"""

import numpy as np

# ==================== 食物营养素数据库 ====================
# 5维营养素: [热量(kcal), 蛋白质(g), 脂肪(g), 碳水(g), 纤维(g)]
FOOD_DATABASE = {
    # 早餐食物
    'coffee_black': np.array([5, 0.3, 0, 0.5, 0]),
    'coffee_latte': np.array([120, 8, 5, 12, 3]),
    'sandwich_egg': np.array([350, 18, 15, 40, 5]),
    'sandwich_ham': np.array([400, 22, 18, 42, 4]),
    'cereal_whole': np.array([250, 8, 3, 50, 8]),
    'cereal_sugary': np.array([300, 5, 2, 60, 2]),
    'toast_butter': np.array([200, 5, 10, 25, 2]),
    'yogurt_fruit': np.array([180, 12, 5, 25, 4]),
    'pancakes': np.array([450, 10, 15, 65, 3]),
    'oatmeal': np.array([300, 10, 5, 55, 10]),

    # 午餐食物
    'rice_white': np.array([200, 4, 0.5, 45, 1]),
    'rice_brown': np.array([220, 5, 2, 45, 4]),
    'chicken_grilled': np.array([250, 35, 10, 0, 0]),
    'chicken_fried': np.array([400, 30, 22, 15, 1]),
    'salad_green': np.array([80, 3, 5, 8, 5]),
    'salad_cesar': np.array([350, 15, 25, 20, 4]),
    'soup_vegetable': np.array([150, 5, 3, 25, 6]),
    'soup_cream': np.array([300, 8, 20, 25, 3]),
    'pasta_tomato': np.array([400, 12, 8, 70, 5]),
    'fish_grilled': np.array([200, 30, 8, 5, 1]),

    # 晚餐食物
    'steak': np.array([450, 40, 28, 0, 0]),
    'pasta_cream': np.array([500, 15, 25, 60, 3]),
    'pizza_slice': np.array([300, 12, 12, 35, 2]),
    'stir_fry_veg': np.array([250, 10, 8, 35, 8]),
    'sushi_roll': np.array([200, 8, 2, 38, 2]),
    'curry_rice': np.array([450, 18, 15, 60, 5]),
    'salmon_grilled': np.array([350, 35, 20, 0, 0]),
    'tacos': np.array([400, 20, 18, 40, 6]),
    'soup_noodle': np.array([350, 15, 8, 55, 4]),
    'roast_chicken': np.array([400, 38, 22, 5, 1])
}

# 食物名称映射
FOOD_NAMES = {
    'coffee_black': '黑咖啡', 'coffee_latte': '拿铁',
    'sandwich_egg': '鸡蛋三明治', 'sandwich_ham': '火腿三明治',
    'cereal_whole': '全麦麦片', 'cereal_sugary': '甜麦片',
    'toast_butter': '黄油吐司', 'yogurt_fruit': '水果酸奶',
    'pancakes': '煎饼', 'oatmeal': '燕麦粥',
    'rice_white': '白米饭', 'rice_brown': '糙米饭',
    'chicken_grilled': '烤鸡胸', 'chicken_fried': '炸鸡',
    'salad_green': '蔬菜沙拉', 'salad_cesar': '凯撒沙拉',
    'soup_vegetable': '蔬菜汤', 'soup_cream': '奶油汤',
    'pasta_tomato': '番茄意面', 'fish_grilled': '烤鱼',
    'steak': '牛排', 'pasta_cream': '奶油意面',
    'pizza_slice': '披萨', 'stir_fry_veg': '炒蔬菜',
    'sushi_roll': '寿司卷', 'curry_rice': '咖喱饭',
    'salmon_grilled': '烤三文鱼', 'tacos': '墨西哥卷',
    'soup_noodle': '汤面', 'roast_chicken': '烤鸡'
}

# 餐次分类
MEAL_CATEGORIES = {
    'breakfast': ['coffee_black', 'coffee_latte', 'sandwich_egg', 'sandwich_ham', 
                  'cereal_whole', 'cereal_sugary', 'toast_butter', 'yogurt_fruit', 
                  'pancakes', 'oatmeal'],
    'lunch': ['rice_white', 'rice_brown', 'chicken_grilled', 'chicken_fried',
              'salad_green', 'salad_cesar', 'soup_vegetable', 'soup_cream',
              'pasta_tomato', 'fish_grilled'],
    'dinner': ['steak', 'pasta_cream', 'pizza_slice', 'stir_fry_veg',
               'sushi_roll', 'curry_rice', 'salmon_grilled', 'tacos',
               'soup_noodle', 'roast_chicken']
}

# 营养素名称
NUTRIENT_NAMES = ['热量(kcal)', '蛋白质(g)', '脂肪(g)', '碳水(g)', '纤维(g)']

# 目标营养素范围 (全天)
TARGET_RANGES = {
    0: (1500, 2000),  # 热量
    1: (80, 120),     # 蛋白质
    2: (40, 70),      # 脂肪
    3: (200, 300),    # 碳水
    4: (20, 40)       # 纤维
}

# 宽松目标 (用于对比实验)
TARGET_RANGES_LOOSE = {
    0: (1400, 2200), 1: (70, 130), 2: (30, 80), 3: (180, 320), 4: (15, 45)
}

# 严格目标
TARGET_RANGES_STRICT = {
    0: (1600, 1900), 1: (90, 110), 2: (50, 60), 3: (220, 280), 4: (25, 35)
}
