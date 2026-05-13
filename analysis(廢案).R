# 均一資料集清洗與迴歸分析腳本
# 依照實作要求.md 撰寫

# 階段一：環境準備與資料載入
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("vroom")) install.packages("vroom")

library(tidyverse)
library(vroom)

# 定義檔案路徑
LOG_PATH <- "Log_Problem.csv"
USER_PATH <- "Info_UserData.csv"
DATA_ZIP <- "junyi_data.zip"
# 資料下載網址 (來自實作要求.md)
DOWNLOAD_URL <- "https://www.kaggle.com/api/v1/datasets/download/junyiacademy/learning-activity-public-dataset-by-junyi-academy"

# --- 自動檢查與下載邏輯 ---
if (!file.exists(LOG_PATH) | !file.exists(USER_PATH)) {
  message("系統：檢測到缺少資料檔案 (Log_Problem.csv 或 Info_UserData.csv)")
  
  if (!file.exists(DATA_ZIP)) {
    message("系統：正在下載資料集壓縮檔 (約 1.5 GB)，請耐心等候...")
    
    # 將逾時時間增加到 3600 秒 (1 小時)
    old_timeout <- getOption("timeout")
    options(timeout = 3600)
    
    # 嘗試下載
    tryCatch({
      # 針對 Windows 系統，method 使用 "libcurl" 處理大型 HTTPS 下載較穩定
      download.file(DOWNLOAD_URL, destfile = DATA_ZIP, mode = "wb", method = "libcurl")
    }, error = function(e) {
      options(timeout = old_timeout) # 發生錯誤時還原設定
      stop("下載失敗！可能是網路不穩或 Kaggle 連結變動。\n",
           "建議解決方案：\n",
           "1. 手動下載檔案：", DOWNLOAD_URL, "\n",
           "2. 將檔案命名為 junyi_data.zip 並放在此資料夾：", getwd(), "\n")
    })
    
    options(timeout = old_timeout) # 下載成功後還原設定
  }
  
  message("系統：正在解壓資料集...")
  unzip(DATA_ZIP)
  
  # 檢查解壓後是否產生正確檔案 (有時 Kaggle 會包在子目錄)
  if (!file.exists(LOG_PATH)) {
    # 搜尋目錄下所有的 csv 並嘗試移動/重新命名 (這是一個保險做法)
    all_csvs <- list.files(pattern = "*.csv", recursive = TRUE)
    log_src <- all_csvs[grep("Log_Problem", all_csvs)][1]
    user_src <- all_csvs[grep("Info_UserData", all_csvs)][1]
    
    if (!is.na(log_src)) file.rename(log_src, LOG_PATH)
    if (!is.na(user_src)) file.rename(user_src, USER_PATH)
  }
  message("系統：資料準備就緒。")
}

message("正在讀取資料... 使用 vroom 以提升效率")

# 1. 載入核心資料表：僅讀取必要欄位以節省記憶體
log_problem <- vroom(LOG_PATH, 
                     col_select = c(uuid, used_hint_cnt, total_sec_taken, is_correct),
                     show_col_types = FALSE)

user_info <- vroom(USER_PATH, 
                   col_select = c(uuid), 
                   show_col_types = FALSE)

message(sprintf("原始資料讀取完成：Log_Problem 共 %d 筆紀錄，使用者名單共 %d 人。", 
                nrow(log_problem), nrow(user_info)))

#----------------------------------------------------#
# 階段二：資料清洗與特徵工程
message("正在進行資料清洗與特徵聚合...")

student_stats <- log_problem %>%
  # 移除含有缺失值的紀錄
  filter(!is.na(uuid), !is.na(used_hint_cnt), !is.na(total_sec_taken), !is.na(is_correct)) %>%
  group_by(uuid) %>%
  summarize(
    avg_hint = mean(used_hint_cnt, na.rm = TRUE),
    avg_time = mean(total_sec_taken, na.rm = TRUE),
    accuracy = sum(is_correct == 1 | is_correct == TRUE) / n(),
    total_tasks = n()
  )

count_after_agg <- nrow(student_stats)
message(sprintf("初步聚合完成：共有 %d 位學生的統計資料。", count_after_agg))

student_stats <- student_stats %>%
  # 為了統計穩定性，過濾掉作答次數太少的學生 (例如至少做過 5 題)
  filter(total_tasks >= 5) %>%
  inner_join(user_info, by = "uuid")

count_after_filter <- nrow(student_stats)
message(sprintf("初步篩選完成 (作答數 >= 5 並與使用者名單對應)：剩下 %d 位學生 (篩選掉 %d 位)。", 
                count_after_filter, count_after_agg - count_after_filter))


# 處理極端值 (Outliers Handling)
message("正在處理極端值 (Outliers)...")

remove_outliers <- function(df, col) {
  Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  # 由於提示與時間不會是負數，將下限與 0 取大值
  df %>% filter(!!sym(col) >= max(0, lower) & !!sym(col) <= upper)
}

# 針對平均提示次數與平均時間進行過濾
original_count <- nrow(student_stats)
student_stats <- student_stats %>%
  remove_outliers("avg_hint") %>%
  remove_outliers("avg_time")

new_count <- nrow(student_stats)
message(sprintf("極端值過濾完成：\n  - 過濾前：%d 位學生\n  - 過濾後：%d 位學生\n  - 已移除：%d 筆紀錄 (約佔 %.2f%%)", 
                original_count, new_count, original_count - new_count, 
                (original_count - new_count) / original_count * 100))


# ------------------------------------------------------#
# 階段三：建立統計模型
message("正在建立多元線性迴歸模型...")

# 正確率 ~ 平均提示次數 + 平均解題時間
model <- lm(accuracy ~ avg_hint + avg_time, data = student_stats)

# 輸出報表
cat("\n--- 迴歸分析報表 ---\n")
print(summary(model))

# 階段四：視覺化
message("正在產出視覺化圖表...")

plot_accuracy_hint <- ggplot(student_stats, aes(x = avg_hint, y = accuracy)) +
  geom_bin2d(bins = 50) + # 資料量大時，使用 bin2d 或 hex 比 geom_point 更清晰
  scale_fill_gradient(low = "lightblue", high = "red") +
  geom_smooth(method = "lm", color = "yellow", linetype = "dashed") +
  labs(title = "平均提示次數與答題正確率之關係",
       subtitle = "黃線為線性迴歸趨勢線",
       x = "平均使用提示次數 (avg_hint)",
       y = "答題正確率 (accuracy)",
       fill = "學生人數") +
  theme_minimal()

# 儲存圖表
ggsave("analysis_result.png", plot_accuracy_hint, width = 10, height = 7)
message("圖表已儲存為 analysis_result.png")

cat("\n分析完成！\n")
