# -*- coding: utf-8 -*-
"""
项目：美国分县每日空气质量（AQI）数据分析 —— 数据清洗 + EDA 探索性分析
数据集：daily_aqi_by_county_2025.csv（EPA 2025 年分县每日 AQI，约 32.4 万行）
运行环境：pandas >= 1.5
说明：本脚本可直接运行，所有清洗规则均基于 EPA 官方数据规范（AQI 有效区间 0–500，
      Category 与 AQI 的对应关系采用 EPA 官方分级断点）。
"""

import pandas as pd
import numpy as np

# ============================================================
# 0. 读取数据
# ============================================================
RAW_PATH = "D:\chrome\downloads\daily_aqi_by_county_2025\daily_aqi_by_county_2025.csv"  # 按需修改为你的实际路径

# dtype=str 先全部按字符串读入，避免 County Code 的 leading zero（如 "003"）被吃掉
df = pd.read_csv(RAW_PATH, dtype=str)

print("=" * 60)
print(f"原始数据形状: {df.shape}")
print(f"原始列名: {list(df.columns)}")

# ============================================================
# 1. 列名标准化（清洗前置步骤）
# ============================================================
# 去掉首尾空格、转小写、空格替换为下划线，保证后续代码列名稳定
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ============================================================
# 2. 去重
# ============================================================
# 2.1 全列完全重复的行：通常是抓取/导出过程中的重复写入
n_full_dup = df.duplicated().sum()
print(f"\n[去重] 完全重复行数: {n_full_dup}")
df = df.drop_duplicates()

# 2.2 业务主键重复：同一州 + 县 + 日期应只有一条记录
#     若存在重复，保留“参与监测站点数最多”的一条（数据可信度更高）
biz_keys = ["state_name", "county_name", "date"]
n_biz_dup = df.duplicated(subset=biz_keys).sum()
print(f"[去重] 业务主键重复行数(州+县+日期): {n_biz_dup}")
df = (df.sort_values("number_of_sites_reporting", ascending=False)
        .drop_duplicates(subset=biz_keys, keep="first")
        .sort_index())

# ============================================================
# 3. 缺失值处理
# ============================================================
print("\n[缺失值] 各列缺失数量：")
print(df.isna().sum())

# 3.1 关键字段（日期、AQI）缺失的行无法用于时序/指标分析，直接删除
key_cols = ["date", "aqi"]
df = df.dropna(subset=key_cols)

# 3.2 分类字段缺失：单独归为 "Unknown"，避免误删整行导致样本偏差
cat_cols = ["category", "defining_parameter", "defining_site"]
for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

# 3.3 站点数缺失：按 0 处理并新增缺失标记列（保留信息，供后续过滤低覆盖样本）
df["sites_reporting_missing"] = df["number_of_sites_reporting"].isna()
df["number_of_sites_reporting"] = df["number_of_sites_reporting"].fillna("0")

# ============================================================
# 4. 字段类型转换
# ============================================================
# 4.1 日期 -> datetime，并派生年/月/星期/季节，为时序分析做准备
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.day_name()
season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["season"] = df["month"].map(season_map)

# 4.2 AQI、站点数 -> 数值型（errors='coerce' 把脏值转 NaN 而非报错中断）
df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
df["number_of_sites_reporting"] = pd.to_numeric(df["number_of_sites_reporting"], errors="coerce")

# 4.3 代码字段保持字符串（County Code 带前导零，绝不能转数值）
df["state_code"] = df["state_code"].astype(str).str.zfill(2)
df["county_code"] = df["county_code"].astype(str).str.zfill(3)

# 4.4 低基数分类字段转 category 类型：省内存 + 分组更快
df["category"] = df["category"].astype("category")
df["defining_parameter"] = df["defining_parameter"].astype("category")
df["season"] = df["season"].astype("category")

# ============================================================
# 5. 异常值处理
# ============================================================
# 5.1 AQI 有效区间 [0, 500]（EPA 规范，500 为爆表上限）。
#     区间外置为 NaN 而不是删除整行——其他字段（如污染物类型）仍有分析价值。
invalid_aqi = ~df["aqi"].between(0, 500)
print(f"\n[异常值] AQI 超出 [0,500] 的记录数: {invalid_aqi.sum()}")
df.loc[invalid_aqi, "aqi"] = np.nan

# 5.2 监测站点数应 >= 1；0 或负数说明当天无有效站点，标记为低可信度样本
df["low_coverage"] = df["number_of_sites_reporting"] < 1

# 5.3 逻辑一致性校验：Category 与 AQI 是否匹配 EPA 官方断点
#     不匹配说明源数据分类有瑕疵，打标记便于作品集展示"数据审计"能力
epa_bins = [-np.inf, 50, 100, 150, 200, 300, 500]
epa_labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
              "Unhealthy", "Very Unhealthy", "Hazardous"]
df["category_expected"] = pd.cut(df["aqi"], bins=epa_bins, labels=epa_labels)
mismatch = df["category"].astype(str) != df["category_expected"].astype(str)
print(f"[异常值] Category 与 AQI 断点不匹配的记录数: {mismatch.sum()}")
df["category_mismatch"] = mismatch  # 作品集亮点：主动暴露并量化数据质量问题

# 5.4 异常值处理后，再清理一次因 coerce 产生的 NaN 关键行
df = df.dropna(subset=["aqi"])

# ============================================================
# 6. 保存清洗结果
# ============================================================
CLEAN_PATH = "aqi_clean.csv"
df.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
print(f"\n[输出] 清洗后数据: {df.shape} -> {CLEAN_PATH}")

# ============================================================
# 7. EDA —— 探索性分析
# ============================================================
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# ---------- 7.1 描述统计：整体空气质量处于什么水平 ----------
print("\n" + "=" * 60)
print("7.1 描述统计（AQI 整体分布）")
print(df["aqi"].describe())
print("\n空气质量等级占比：")
print(df["category"].value_counts(normalize=True).round(4) * 100)

# 业务含义：AQI 均值/中位数反映全国整体空气水平；
# 污染日（AQI>100）占比直接回答"全年有多少天空气不健康"。

# ---------- 7.2 时间维度：空气质量的季节性规律 ----------
print("\n" + "=" * 60)
print("7.2 分月空气质量（均值 AQI + 污染日占比）")
df["polluted_day"] = df["aqi"] > 100  # EPA：AQI>100 即对敏感人群不健康
monthly = df.groupby("month").agg(
    avg_aqi=("aqi", "mean"),
    polluted_day_rate=("polluted_day", "mean"),
    records=("aqi", "size"),
).round(3)
print(monthly)

print("\n分季节对比：")
print(df.groupby("season").agg(avg_aqi=("aqi", "mean"),
                               max_aqi=("aqi", "max"),
                               polluted_day_rate=("polluted_day", "mean")).round(3))

# 业务含义：识别污染高发月份/季节（如夏季臭氧、冬季颗粒物），
# 为公众健康预警和政府治理排期提供依据。

# ---------- 7.3 空间维度：哪些州/县污染最严重 ----------
print("\n" + "=" * 60)
print("7.3 各州空气质量排名（按年均 AQI，仅统计记录数>=100 的州，保证样本量）")
state_stats = df.groupby("state_name").agg(
    avg_aqi=("aqi", "mean"),
    polluted_day_rate=("polluted_day", "mean"),
    counties=("county_name", "nunique"),
    records=("aqi", "size"),
).query("records >= 100").round(3)
print(state_stats.sort_values("avg_aqi", ascending=False).head(15))

print("\n污染最严重的 15 个县：")
county_stats = df.groupby(["state_name", "county_name"]).agg(
    avg_aqi=("aqi", "mean"),
    polluted_day_rate=("polluted_day", "mean"),
    records=("aqi", "size"),
).query("records >= 100").round(3)
print(county_stats.sort_values("avg_aqi", ascending=False).head(15))

# 业务含义：定位污染热点地区，是环境政策资源投放（限排、监测加密）的决策依据。

# ---------- 7.4 污染物维度：谁在主导污染 ----------
print("\n" + "=" * 60)
print("7.4 首要污染物分布（整体 + 分季节）")
print(df["defining_parameter"].value_counts(normalize=True).round(4) * 100)

pollutant_season = pd.crosstab(df["defining_parameter"],
                               df["season"], normalize="index").round(4) * 100
print("\n各污染物在不同季节的占比(%)：")
print(pollutant_season)

# 业务含义：PM2.5 主导 -> 工业/燃煤治理；Ozone 主导 -> 机动车与光化学烟雾治理；
# 季节性变化揭示污染源的季节切换（取暖季颗粒物升高、夏季臭氧升高）。

# ---------- 7.5 数据可信度维度：监测覆盖度 ----------
print("\n" + "=" * 60)
print("7.5 监测覆盖情况")
print(f"低覆盖样本占比: {df['low_coverage'].mean():.4%}")
print(f"Category 与断点不一致占比: {df['category_mismatch'].mean():.4%}")

# 业务含义：向下游读者说明数据质量边界——结论建立在多高质量的样本之上，
# 是数据分析师专业性的体现（作品集加分项）。

print("\n全部完成。建议下一步：用 matplotlib/seaborn 将 7.2-7.4 结果可视化。")
