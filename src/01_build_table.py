"""
01_build_table.py

This script constructs the modeling dataset from raw Olympic medal data.

Input:
    olympic_medals.csv

Output:
    summer_model_table.csv

Description:
    The script filters Summer Olympic Games data and aggregates medal counts
    by country and year. The resulting table serves as the input dataset for
    subsequent machine learning modeling.
"""



import os
import pandas as pd

# ===== 找到数据路径 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDALS_FILE = os.path.join(BASE_DIR, "olympic_medals.csv")

print("Reading:", MEDALS_FILE)

# ===== 读取数据 =====
medals = pd.read_csv(MEDALS_FILE)

# ===== 只保留 Summer Olympics =====
medals["season"] = medals["slug_game"].str.extract(r"([a-zA-Z]+-\d{4})")

# 简单规则：冬奥 games 数量明显较少
winter_games = [
    "beijing-2022", "pyeongchang-2018", "sochi-2014",
    "vancouver-2010", "turin-2006", "salt-lake-city-2002",
    "nagano-1998", "lillehammer-1994"
]

medals = medals[~medals["slug_game"].isin(winter_games)]


print("Loaded medals:", medals.shape)

# ===== 从 slug_game 提取年份 =====
medals["year"] = medals["slug_game"].str.extract(r"(\d{4})").astype(int)

# ===== 聚合：国家 × 年份 奖牌总数 =====
agg = (
    medals
    .groupby(["year", "country_3_letter_code"])
    .size()
    .reset_index(name="total_medals")
)

print("\nPreview:")
print(agg.head())

print("\nYears:", agg["year"].min(), "-", agg["year"].max())
print("Countries:", agg["country_3_letter_code"].nunique())
print("Rows:", len(agg))

# ===== 保存建模表 =====
OUTPUT_FILE = os.path.join(BASE_DIR, "summer_model_table.csv")
agg.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:", OUTPUT_FILE)
