import pandas as pd
from scipy.stats import kruskal

# ==========================================================
# 0. 讀取三份資料（自行修改成你的檔案路徑）
# ==========================================================
user = pd.read_csv("./archive/Info_UserData.csv")
print("已讀取使用者資料")
content = pd.read_csv("./archive/Info_Content.csv")
print("已讀取內容資料")
log = pd.read_csv("./archive/Log_Problem.csv")
print("已讀取使用紀錄資料")

# ==========================================================
# 1. 基礎資料合併
# ==========================================================
# log_Problem 有 uuid & ucid，兩邊合併
df = log.merge(user, on="uuid", how="left") \
        .merge(content, on="ucid", how="left")

print("原始資料筆數：", len(df))

# ==========================================================
# 2. 預處理條件
# ----------------------------------------------------------
# (1) 年級限制：1~9
df = df[(df["user_grade"] >= 1) & (df["user_grade"] <= 9)]
# (2) badges_cnt >= 1
df = df[df["badges_cnt"] >= 1]
# (3) is_self_coach = False
# 若以 0/1 表示，可改為 df["is_self_coach"] == 0
df = df[df["is_self_coach"] == False]
# (4) learning_stage：只保留 elementary & junior
df=df[(df["learning_stage"]=="elementary")|(df["learning_stage"]=="junior")]  # (4) 只保留基礎階段
df = df[
    (df["gender"].isna()) |
    (df["gender"] == "NULL") |
    (df["gender"] == "male") |
    (df["gender"] == "female")
]
print("預處理後筆數：", len(df))

# ==========================================================
# 3. 建立行為指標
# ----------------------------------------------------------
# 計算每位學生的 accuracy（正確率）
df["correct"] = df["is_correct"].astype(int)

student_group = df.groupby("uuid").agg({
    "points": "max",             # 每位學生總積分
    "badges_cnt": "max",
    "user_city": "max",
    "learning_stage": "max",
    "correct": "mean",           # 平均正確率
})
print(student_group.head())
student_group.rename(columns={"correct": "accuracy"}, inplace=True)

print("彙整後的學生資料筆數：", len(student_group))

# ==========================================================
# 4. 地區差異分析（使用 Kruskal-Wallis）
# ----------------------------------------------------------
def kruskal_by_city(df, column):
    """計算不同城市在 column 上是否有差異"""
    city_groups = [
        group[column].dropna().values
        for _, group in df.groupby("user_city")
        if len(group[column].dropna()) > 1  # 避免城市資料量太少
    ]
    stat, p = kruskal(*city_groups)
    return stat, p

# 要檢定的變數
targets = ["points", "badges_cnt", "accuracy"]

print("\n===== Kruskal-Wallis 城市差異檢定 =====")
for col in targets:
    stat, p = kruskal_by_city(student_group, col)
    print(f"{col}: H = {stat:.3f}, p = {p:.25f}")
# 第二處分析
import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency


# ==========================================================
# 3. Tableau 的 COUNTD(Date(timestamp))
# ==========================================================

def tableau_parse_timestamp(col):
    """
    100% 模擬 Tableau 的 timestamp parsing 行為：
    - 10 位 Unix 秒
    - 13 位 Unix 毫秒
    - 其他一律用 pandas 自動 parse（不強制 coerce）
    """
    col = col.astype(str)
    mask_digit = col.str.isdigit()

    time10 = pd.to_datetime(col[mask_digit & (col.str.len()==10)].astype(int), unit='s', errors='ignore')
    time13 = pd.to_datetime(col[mask_digit & (col.str.len()==13)].astype(int), unit='ms', errors='ignore')
    time_other = pd.to_datetime(col[~mask_digit], errors='ignore')

    out = pd.concat([time10, time13, time_other]).sort_index()
    return out

df["ts_clean"] = tableau_parse_timestamp(df["timestamp_TW"])
df["date_only"] = df["ts_clean"].dt.strftime("%Y-%m-%d")   # Tableau 的 DATE() 會存成字串格式

# Tableau: {FIXED Uuid: COUNTD(Date)}
active_days = df.groupby("uuid")["date_only"].nunique()

# ==========================================================
# 4. Tableau 的 total_hours
# ==========================================================
total_hours = df.groupby("uuid")["total_sec_taken"].sum() / 3600

# ==========================================================
# 5. Tableau 的教師支援
# ==========================================================
teacher_support = (df.groupby("uuid")["has_teacher_cnt"].max() > 0).astype(int)

# ==========================================================
# 6. 統整基本欄位
# ==========================================================
stu = df.groupby("uuid").agg({
    "points": "max",
    "badges_cnt": "max",
    "user_city": "max",
    "learning_stage": "max",
    "is_correct": "mean"
}).rename(columns={"is_correct":"accuracy"})

stu["active_days"] = active_days
stu["total_hours"] = total_hours
stu["teacher_support"] = teacher_support

stu = stu.dropna(subset=["active_days", "total_hours"])

# ==========================================================
# 7. 完全依照 Tableau 的 Z-score（FIXED + AVG + STDEVP）
# ==========================================================

def tableau_z(series):
    mean = series.mean()
    std = series.std(ddof=0)   # Tableau = population stdev
    return (series - mean) / std

stu["std_days"] = tableau_z(stu["active_days"])
stu["std_hours"] = tableau_z(stu["total_hours"])
stu["std_points"] = tableau_z(stu["points"])

# ==========================================================
# 8. 100% 模擬 Tableau 的行為分類判斷
# ==========================================================
stu["burst_type"] = ((stu["std_days"] < 0.2) & (stu["std_points"] > 0)).astype(int)
stu["stable_type"] = ((stu["std_hours"] > 0) & (stu["std_days"] > 0)).astype(int)
stu["selflearn_type"] = ((stu["std_hours"] > 0) & (stu["std_days"] > 0) & (stu["teacher_support"]==0)).astype(int)
stu["passive_type"] = ((stu["std_hours"] < 0) & (stu["std_points"] < 0)).astype(int)

print("\n=== 行為類型人數（與 Tableau 一致） ===")
for b in ["stable_type","burst_type","selflearn_type","passive_type"]:
    print(b, ":", stu[b].sum())


print("\n===== 第三處：地區 × 行為模式（卡方檢定）=====")

behavior_cols = ["burst_type", "stable_type", "selflearn_type", "passive_type"]

for b in behavior_cols:
    print(f"\n=== 行為模式：{b} ===")

    # 注意！！！stu 才是有行為欄位的 DataFrame
    crosstab = pd.crosstab(stu["user_city"], stu[b])

    chi2, p, dof, expected = chi2_contingency(crosstab)

    print("Chi-square =", chi2)
    print("df =", dof)
    print("p-value =", p)
