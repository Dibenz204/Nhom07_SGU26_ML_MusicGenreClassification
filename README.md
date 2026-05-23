# 🎵 Music Genre Classification

> Báo cáo học phần **Máy học (Machine Learning)** — Khoa Toán - Ứng dụng, Trường Đại học Sài Gòn  
> Tháng 5 năm 2026

---

## Nhóm thực hiện

| Họ tên | MSSV | Phụ trách |
|---|---|---|
| Nguyễn Đình Phong | 3122480044 | Tiền xử lý, xây dựng mô hình, tổng hợp báo cáo, slide/poster |
| Trương Thị Ngọc Nhi | 3122480039 | Phân tích đa biến |
| Võ Văn Truyền Vũ | 3122480080 | Phân tích đơn biến |
| Lê Tấn Nhật Minh | 3122480032 | Xác định tính chất dữ liệu |

**Giảng viên hướng dẫn:** TS. Đỗ Như Tài

---

## Tổng quan dự án

Dự án xây dựng hệ thống phân loại tự động các bài hát vào **11 thể loại âm nhạc** (Class 0–10) dựa trên các đặc trưng vật lý của sóng âm. Việc tự động hóa gán nhãn thể loại là nền tảng cho các hệ thống gợi ý nhạc (Recommendation Systems) trên các nền tảng phát nhạc trực tuyến.

### Mục tiêu chiến lược

- **Thấu hiểu đặc trưng âm thanh** — giải mã mối quan hệ và sức mạnh phân tách lớp của các thuộc tính vật lý
- **Triệt tiêu sai lệch hệ thống** — làm sạch lỗi logic, xử lý bất thường và dữ liệu khuyết
- **Tối ưu hóa kiến trúc dữ liệu** — thiết kế biến phái sinh và chọn giải pháp phù hợp để xử lý mất cân bằng lớp

---

## Dataset

| Thông số | Giá trị |
|---|---|
| Số bài hát (dòng) | 14,396 |
| Số đặc trưng (cột) | 18 (14 features + 1 nhãn) |
| Số thể loại nhạc | 11 (Class 0–10) |
| Dữ liệu trùng lặp | 0 |
| Tổng ô thiếu dữ liệu | 5,483 |

### Phân phối lớp (mất cân bằng nghiêm trọng)

```
Class 10  ████████████████████████████  3,959  (27.5%)  ← Rock / Metal / Punk
Class 6   ███████████████              2,069  (14.4%)  ← Hip-Hop / Rap
Class 9   ███████████████              2,019  (14.0%)  ← Pop / Latin / Reggae
Class 8   ██████████                   1,483  (10.3%)
Class 5   ████████                     1,157  ( 8.0%)  ← EDM / Electronic
Class 1   ███████                      1,098  ( 7.6%)  ← Alternative / Indie
Class 2   ███████                      1,018  ( 7.1%)  ← Blues / R&B
Class 0   ███                            500  ( 3.5%)  ← Acoustic / Classical
Class 7   ███                            461  ( 3.2%)  ← Jazz
Class 3   ██                             322  ( 2.2%)  ← Classical / Symphony
Class 4   ██                             310  ( 2.2%)  ← Country / Live Folk
```

>  Chênh lệch **12.8×** giữa Class 10 và Class 4 — cần chiến lược xử lý mất cân bằng.

### Các features sử dụng

| Feature | Kiểu | Ý nghĩa |
|---|---|---|
| `danceability` | float [0,1] | Khả năng nhảy theo nhịp |
| `energy` | float [0,1] | Cường độ và mức độ sôi động |
| `loudness` | float (dBFS) | Âm lượng tổng thể (luôn ≤ 0) |
| `speechiness` | float [0,1] | Tỉ lệ lời nói/rap trong bài |
| `acousticness` | float [0,1] | Xác suất là nhạc acoustic |
| `instrumentalness` | float [0,1] | Xác suất không có giọng hát |
| `liveness` | float [0,1] | Xác suất thu âm trực tiếp (live) |
| `valence` | float [0,1] | Sắc thái cảm xúc (buồn → vui) |
| `tempo` | float (BPM) | Tốc độ nhịp |
| `duration_in min/ms` | float | Thời lượng bài hát |
| `key` | int [0,11] | Khóa nhạc (Pitch Class) |
| `mode` | int {0,1} | Điệu thức (0=thứ, 1=trưởng) |
| `time_signature` | int | Số nhịp mỗi ô nhịp |
| `Popularity` | float [0,100] | Độ phổ biến trên nền tảng |

---

## Phân tích dữ liệu

### Missing Values

| Cột | Số lượng thiếu | Tỉ lệ | Mức độ |
|---|---|---|---|
| `instrumentalness` | 3,541 | 24.60% | Nghiêm trọng |
| `key` | 1,609 | 11.18% | Trung bình |
| `Popularity` | 333 | 2.31% | Nhẹ |

### Các lỗi logic phát hiện

| Điều kiện | Số dòng vi phạm | Nhận xét |
|---|---|---|
| `loudness > 0` dBFS | 7 | Bất hợp lệ về vật lý âm thanh số |
| `duration < 30,000ms` | 2,075 | Nghi ngờ lỗi đơn vị hoặc preview |
| `energy > 0.8` và `acousticness > 0.8` | 14 | Mâu thuẫn logic âm thanh |
| `instrumentalness > 0.8` và `speechiness > 0.5` | 2 | Mâu thuẫn logic |
| `duration > 600,000ms` (>10 phút) | 62 | Hợp lệ — nhạc cổ điển, live |

### Mức độ quan trọng các features (F-score)

```
acousticness      ████████████████████  13.1%
speechiness       ████████████████      10.4%
energy            ████████████████      10.1%
Popularity        █████████████          8.9%
instrumentalness  █████████████          8.5%
loudness          █████████████          8.4%
valence           ████████████           8.1%
danceability      ████████████           8.0%
duration_ms       ██████████             7.5%
liveness          ████████               7.1%
tempo             ███                    2.0%
```

> Chỉ cần 11 features đầu để giải thích **90%** khả năng ra quyết định của mô hình.

### Tương quan nổi bật

| Cặp features | Hệ số | Ý nghĩa |
|---|---|---|
| `energy` ↔ `loudness` | **+0.77** | Bài sôi động → âm lượng lớn |
| `energy` ↔ `acousticness` | **−0.75** | Rock/EDM vs Acoustic/Classical |
| `loudness` ↔ `acousticness` | **−0.61** | Nhạc mộc thường nhỏ hơn |
| `danceability` ↔ `valence` | **+0.44** | Bài dễ nhảy → cảm xúc tích cực |

---

## Tiền xử lý dữ liệu

### Pipeline xử lý tập Train

```
1. Đồng nhất đơn vị duration
   └─ Nếu giá trị < 100 → nhân × 60,000 (quy về milliseconds)
   └─ 14.4% dữ liệu bị lỗi trộn lẫn phút/milliseconds

2. Sửa giá trị vô lý
   └─ loudness > 0: đổi dấu thành giá trị âm (7 dòng)

3. Điền khuyết (Missing Value Imputation)
   └─ Popularity     → Median = 44.0
   └─ key            → Mode   = 7
   └─ instrumentalness → Median = 0.003990

4. Xử lý ngoại lai — Winsorization [1%–99%]
   └─ loudness:     [−22.64, −2.14] dBFS
   └─ duration_ms:  [112,844 – 526,782] ms  (~1.9–8.8 phút)
   └─ tempo:        [70.98 – 193.98] BPM

5. Cyclical Encoding cho key (xử lý tính tuần hoàn nhạc lý)
   └─ key_sin = sin(2π × k / 12)
   └─ key_cos = cos(2π × k / 12)

6. Loại bỏ cột không dùng
   └─ Id, Artist Name, Track Name
```

### Tham số lưu lại để áp dụng cho Validation/Test

```json
{
  "pop_median":      44.0,
  "key_mode":        7,
  "instr_median":    0.00399,
  "loudness_lo":    -22.636,
  "loudness_hi":    -2.135,
  "duration_ms_lo":  112843.8,
  "duration_ms_hi":  526781.9,
  "tempo_lo":        70.978,
  "tempo_hi":        193.975
}
```

> Toàn bộ tham số được tính từ tập Train và áp dụng cố định cho Validation/Test để tránh **Data Leakage**.

---

## Mô hình

### 1. Random Forest (Bagging)

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced"
)
```

| Tập | Accuracy | Macro F1 |
|---|---|---|
| Train | 91.68% | 0.9315 |
| Validation | 45.24% | 0.4459 |

**F1 theo class nổi bật:**

| Class | F1 (Val) | Ghi chú |
|---|---|---|
| Class 7 | 0.79 | Cao nhất — Jazz dễ nhận diện |
| Class 5 | 0.72 | EDM có đặc trưng rõ ràng |
| Class 1 | 0.006 | Thấp nhất — bị nhầm sang Class 6, 10 |

---

### 2. XGBoost (Boosting)

```python
XGBClassifier(
    n_estimators=1000,    # early stopping → tối ưu tại 214 cây
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=...,
    reg_lambda=...,
    early_stopping_rounds=50
)
```

| Tập | Accuracy | Macro F1 |
|---|---|---|
| Train | 83.52% | 0.8657 |
| Validation | 43.40% | 0.4546 |

---

### 3. Stacking Ensemble (RF + XGBoost)

Kiến trúc 2 tầng: Random Forest (Bagging) + XGBoost (Boosting) làm base models, **Logistic Regression** làm meta-model.

| Tập | Accuracy | Macro F1 |
|---|---|---|
| Validation | 42.92% | 0.4535 |

**Thay đổi F1 so với mô hình đơn tốt nhất:**

| Class | RF | XGB | Stacking | Winner |
|---|---|---|---|---|
| Class 1 | 0.018 | 0.061 | **0.215** | Stacking |
| Class 5 | 0.708 | 0.726 | **0.729** | Stacking |
| Class 7 | 0.766 | 0.792 | **0.796** | Stacking |
| Class 10 | 0.397 | **0.416** | 0.341 | Base |

> Stacking cải thiện đáng kể các lớp khó (Class 1: +0.154) nhưng đánh đổi hiệu năng trên Class 10.

---

## Kết quả Kaggle

| Submission | Public Score | Private Score | Chênh lệch |
|---|---|---|---|
| **Random Forest** | **0.46785** | **0.46481** | **0.00304** ✅ |
| XGBoost + ID | 0.45992 | 0.44074 | 0.01918 |
| XGBoost | 0.45952 | 0.42962 | 0.02990 ⚠️ |
| Stacking RF+XGB | 0.45396 | 0.44444 | 0.00952 |
| Baseline (sample) | 0.50317 | 0.48055 | — |

### Phân tích

- **Random Forest** đạt **private score cao nhất (0.4648)** và ổn định nhất (Δ = 0.003) nhờ cơ chế Bagging giảm phương sai
- **XGBoost** có public score tốt nhưng chênh lệch lớn (Δ = 0.030) — dấu hiệu nhạy cảm với phân phối dữ liệu
- **Stacking** ổn định thứ hai (Δ = 0.010) nhưng chưa vượt được RF trên private set
- Validation set tự tạo chưa đại diện đủ cho private test: XGBoost tốt nhất trên val nhưng RF thắng trên Kaggle

---

## Cấu trúc project

```
├── Data/
│   ├── train.csv
│   ├── test.csv
│   └── Data-processed/
│       ├── train.csv              # 11,516 mẫu
│       ├── validation.csv         #  2,880 mẫu
│       └── train_params.json      # tham số tiền xử lý
├── Notebooks/
│   ├── 1_data_profiling.ipynb
│   ├── 2_univariate_analysis.ipynb
│   ├── 3_multivariate_analysis.ipynb
│   ├── 4_preprocessing.ipynb
│   └── 5_modeling.ipynb
├── Submissions/
│   ├── submission_fixed.csv       # Random Forest  → private 0.4648
│   ├── submission_xgb.csv         # XGBoost        → private 0.4296
│   ├── submission_fixed_XGB_ID.csv
│   └── submission_ensemble.csv    # Stacking       → private 0.4444
└── README.md
```

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![matplotlib](https://img.shields.io/badge/matplotlib-3.x-11557c)
![seaborn](https://img.shields.io/badge/seaborn-0.13-4c72b0)

**Kỹ thuật sử dụng:** PCA · KMeans · SMOTE · Winsorization · Cyclical Encoding · GridSearchCV · Stacking Ensemble

---

## Kết luận

1. **Random Forest là mô hình tốt nhất** cho bài toán này — private score 0.4648, ổn định nhất giữa public và private set
2. **Acousticness, speechiness và energy** là 3 features quan trọng nhất để phân loại thể loại nhạc
3. **Mất cân bằng lớp** (Class 10 gấp 12.8× Class 4) là thách thức lớn nhất — `class_weight=balanced` giúp một phần nhưng chưa đủ
4. **Stacking Ensemble** cải thiện đáng kể Class 1 (F1: 0.006 → 0.215) nhưng tổng thể chưa vượt RF
5. Sự khác biệt giữa validation set tự tạo và Kaggle private test nhấn mạnh tầm quan trọng của **validation set đại diện**

---

*Đại học Sài Gòn · Khoa Toán - Ứng dụng · Tháng 5/2026*
