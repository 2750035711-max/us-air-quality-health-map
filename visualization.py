# -*- coding: utf-8 -*-
"""
项目：美国空气质量健康风险地图 —— 可视化脚本
输入：data_cleaning_eda.py 产出的 aqi_clean.csv
依赖：pandas、matplotlib、seaborn
输出：3 张 PNG 图表（300dpi，可直接放入作品集）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 全局样式 ----------
sns.set_style("whitegrid")          # 先设 seaborn 样式，避免其覆盖字体设置

# 自动检测本机可用中文字体（解决"中文显示为方框"问题）
# 按优先级尝试常见中文字体，找到的第一个生效
from matplotlib import font_manager

CN_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun",
                      "DengXian", "KaiTi", "Noto Sans CJK SC",
                      "WenQuanYi Micro Hei"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
matched = [f for f in CN_FONT_CANDIDATES if f in available_fonts]

if matched:
    plt.rcParams["font.sans-serif"] = matched + ["DejaVu Sans"]
    print(f"[字体] 使用中文字体: {matched[0]}")
else:
    # 都没找到：打印全部已注册字体名，便于排查；图表标签需改为英文
    print("[字体] 未找到中文字体！已注册字体列表：")
    print(sorted(available_fonts))
    print("[字体] 请把图表文字改为英文，或安装任一中文字体后重跑")

plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
FIG_DIR = "figs/"                   # 图片输出目录

import os
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 读取清洗后数据 ----------
df = pd.read_csv("aqi_clean.csv", parse_dates=["date"])
df["polluted_day"] = df["aqi"] > 100  # EPA：AQI>100 对敏感人群不健康

# 统一配色：AQI 越高颜色越"警示"，沿用 EPA 等级色系
C_GOOD, C_MID, C_BAD = "#2E8B57", "#F4A460", "#C0392B"

# ============================================================
# 图 1：月度 AQI 趋势折线图（均值 + 污染日占比，双轴）
# ============================================================
monthly = df.groupby("month").agg(
    avg_aqi=("aqi", "mean"),
    polluted_day_rate=("polluted_day", "mean"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5.5))

# 左轴：月均 AQI 折线 + 季节均值参考线
ax1.plot(monthly["month"], monthly["avg_aqi"], marker="o", lw=2.5,
         color=C_MID, label="月均 AQI")
ax1.axhline(df["aqi"].mean(), ls="--", lw=1.2, color="gray",
            label=f"全年均值 {df['aqi'].mean():.1f}")
ax1.set_xlabel("月份")
ax1.set_ylabel("月均 AQI", color=C_MID)
ax1.set_xticks(range(1, 13))
ax1.set_ylim(30, 50)
ax1.tick_params(axis="y", labelcolor=C_MID)

# 右轴：污染日占比柱状图
ax2 = ax1.twinx()
ax2.bar(monthly["month"], monthly["polluted_day_rate"] * 100, alpha=0.35,
        color=C_BAD, label="污染日占比(%)")
ax2.set_ylabel("污染日占比 (%)", color=C_BAD)
ax2.tick_params(axis="y", labelcolor=C_BAD)

# 夏季高发期标注
ax1.annotate("夏季高发期\n(均值45.5 / 污染日2.2%)",
             xy=(7, monthly.loc[monthly["month"] == 7, "avg_aqi"].iloc[0]),
             xytext=(9, 46.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", color="gray"))

# 合并两轴图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

plt.title("2025 年美国分县日均 AQI：月度变化趋势", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig1_monthly_aqi_trend.png", dpi=300)
plt.close()
print(f"[图1] 已保存 {FIG_DIR}fig1_monthly_aqi_trend.png")

# ============================================================
# 图 2：各州年均 AQI 排名（横向条形图，Top15 污染最重 + 最清洁对照）
# ============================================================
state_stats = df.groupby("state_name").agg(
    avg_aqi=("aqi", "mean"),
    polluted_day_rate=("polluted_day", "mean"),
    records=("aqi", "size"),
).query("records >= 100")  # 样本量门槛，防止小样本误导

# 剔除非美国本土口径（数据审计发现：EPA 含美墨边境墨西哥一侧站点）
state_stats_us = state_stats.drop(index="Country Of Mexico", errors="ignore")

top15 = state_stats_us.nlargest(15, "avg_aqi").iloc[::-1]  # 反转，最大值画在最上方

fig, ax = plt.subplots(figsize=(10, 7))
colors = [C_BAD if v > 50 else C_MID for v in top15["avg_aqi"]]
ax.barh(top15.index, top15["avg_aqi"], color=colors, alpha=0.85)

# 条上标注数值 + 污染日占比
for i, (name, row) in enumerate(top15.iterrows()):
    ax.text(row["avg_aqi"] + 0.3, i,
            f"{row['avg_aqi']:.1f}  (污染日 {row['polluted_day_rate']:.1%})",
            va="center", fontsize=9)

ax.axvline(50, ls="--", lw=1.2, color=C_BAD)   # EPA Moderate 上限
ax.text(50.3, len(top15) - 0.5, "AQI=50 (Moderate上限)",
       fontsize=9, color=C_BAD, va="top")

ax.set_xlabel("年均 AQI")
ax.set_xlim(0, top15["avg_aqi"].max() * 1.35)
plt.title("2025 年美国各州年均 AQI 排名 Top15（已剔除墨西哥站点，n≥100）",
          fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig2_state_ranking.png", dpi=300)
plt.close()
print(f"[图2] 已保存 {FIG_DIR}fig2_state_ranking.png")

# ============================================================
# 图 3：首要污染物 × 季节 热力图
# ============================================================
# 行=首要污染物，列=季节，值=该污染物在此季节的记录占比
pollutant_season = pd.crosstab(df["defining_parameter"],
                               df["season"], normalize="index") * 100
# 仅保留占比 >=1% 的主要污染物，避免噪声稀释色阶
major = df["defining_parameter"].value_counts(normalize=True)
major = major[major >= 0.005].index.tolist()  # 保留占比>=0.5%的污染物（含 NO2 冬季亮点）
pollutant_season = pollutant_season.loc[major,
                                        ["Spring", "Summer", "Fall", "Winter"]]

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.heatmap(pollutant_season, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "记录占比 (%)"}, ax=ax)

ax.set_xlabel("季节")
ax.set_ylabel("首要污染物")
plt.title("2025 年首要污染物季节分布热力图（行内百分比，%）",
          fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIG_DIR + "fig3_pollutant_season_heatmap.png", dpi=300)
plt.close()
print(f"[图3] 已保存 {FIG_DIR}fig3_pollutant_season_heatmap.png")

print("\n全部图表生成完成。")
