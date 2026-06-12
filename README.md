# 🔬 Phát Hiện Bệnh Võng Mạc Do Tiểu Đường

> Đồ án PBL4 – Trí Tuệ Nhân Tạo | Khoa Điện Tử - Viễn Thông | ĐH Bách Khoa Đà Nẵng

**Sinh viên thực hiện:** Trịnh Minh Việt – Trần Lê Long Vũ  
**GVHD:** TS. Trần Thị Minh Hạnh  
**Lớp:** 22KTMT1

---

## 📋 Tổng Quan

Hệ thống phân loại mức độ bệnh võng mạc tiểu đường (Diabetic Retinopathy) từ ảnh đáy mắt (fundus) sử dụng các mô hình học sâu. Dự án huấn luyện và so sánh nhiều kiến trúc và chiến lược transfer learning trên các tập dữ liệu khác nhau.

### Bài toán
- **Dữ liệu 1 nhãn:** Phân loại 5 mức độ DR (Grade 0–4) trên tập APTOS 2019
- **Dữ liệu 2 nhãn:** Phân loại đồng thời DR (4 mức) và DME - Diabetic Macular Edema (3 mức) trên tập Messidor / IDRiD

---

## 🗂️ Tập Dữ Liệu

| Tập dữ liệu | Số ảnh | Nhãn | Nguồn |
|---|---|---|---|
| APTOS 2019 | 3.662 | DR Grade 0–4 | Kaggle |
| Messidor | 1.200 | DR Grade 0–3 + DME risk | Messidor Project |
| IDRiD | 516 (413 train / 103 test) | DR Grade 0–4 + DME 0–2 | IEEE DataPort |

**Phân bố APTOS 2019:**
| Grade | Số lượng |
|---|---|
| 0 (No DR) | 1.805 |
| 1 (Mild) | 370 |
| 2 (Moderate) | 999 |
| 3 (Severe) | 193 |
| 4 (Proliferative) | 295 |

---

## 🏗️ Kiến Trúc Mô Hình

### Mô hình 1 — ResNet50 + Transfer Learning (APTOS 2019)

```
Input 224×224×3
    ↓
ResNet50 backbone (ImageNet pretrained)
  ├─ Conv1: 7×7, stride 2 → 112×112×64
  ├─ MaxPool: 3×3, stride 2 → 56×56×64
  ├─ Conv2_x: 3 blocks → 56×56×256
  ├─ Conv3_x: 4 blocks → 28×28×512
  ├─ Conv4_x: 6 blocks → 14×14×1024
  └─ Conv5_x: 3 blocks → 7×7×2048 ★ (unfreeze phase 2)
    ↓
GlobalAveragePooling2D → 2048
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(5, Softmax) → Grade 0–4
```

**Chiến lược huấn luyện 2 giai đoạn:**
- **Phase 1** (10 epochs, lr=1e-4): Frozen toàn bộ backbone, chỉ train classifier head
- **Phase 2** (20 epochs, lr=1e-5): Unfreeze 40 layers cuối của ResNet50

**Hàm mất mát:** `sparse_categorical_crossentropy` + `class_weight='balanced'`

---

### Mô hình 2 — MultiTaskModel ResNet152 (Messidor / IDRiD)

```
Input 224×224×3
    ↓
ResNet152 backbone (checkpoint từ APTOS)
  ├─ Conv1 + BN + ReLU + MaxPool → 56×56×64
  ├─ Layer1 [frozen]: 3 blocks → 56×56×256
  ├─ Layer2 [unfrozen]: 8 blocks → 28×28×512
  ├─ Layer3 [unfrozen]: 36 blocks → 14×14×1024
  └─ Layer4 [unfrozen]: 3 blocks → 7×7×2048
    ↓
AdaptiveAvgPool2d(1×1) + Flatten → 2048
    ↙              ↘
head_dr           head_dme
Linear(2048→4)    Linear(2048→3)
DR Grade 0–3      DME Risk 0–2
```

**Chiến lược:** Transfer learning từ checkpoint APTOS → fine-tune Layer2/3/4, frozen Layer1 + Conv1  
**Hàm mất mát:** `loss = loss_dr + loss_dme` (CrossEntropyLoss cho từng nhánh)

---

### Mô hình 3 — CANet (Cross-task Attention Network)

Kiến trúc attention-based cho bài toán multi-task DR + DME:
- **Backbone:** ResNet50
- **Disease-specific Attention Module:** Channel-wise + Spatial-wise attention cho từng bệnh
- **Disease-dependent Attention Module:** Khai thác tương quan DR ↔ DME
- **Hàm mất mát:** `L = L_DR + L_DME + λ(L_DR' + L_DME')`, λ=0.25

---

## 📊 Kết Quả

### ResNet50 trên APTOS 2019

| Cấu hình | Validation Accuracy |
|---|---|
| Fine-tuning 40 layers | 77.05% |
| Fine-tuning 25 layers | 76.00% |
| Huấn luyện tiếp từ checkpoint | **98%** |

### ResNet50/152 trên Messidor (10-fold cross-validation)

| Mô hình | DR Acc | DR AUC | DME Acc | DME AUC | Joint Acc |
|---|---|---|---|---|---|
| ResNet50 pretrain | 70.50% | 0.8308 | 90.50% | 0.9324 | 64.92% |
| ResNet152 pretrain | 71.08% | 0.8363 | 90.08% | 0.9334 | 65.08% |

---

## ⚙️ Tiền Xử Lý Dữ Liệu

- Resize ảnh về **224×224** pixels
- Chuẩn hóa bằng `preprocess_input` (ResNet mean ImageNet)
- **Data augmentation:** xoay ±15°, lật ngang, zoom ±10%
- Chia train/validation: **80% / 20%**
- Xử lý mất cân bằng: `class_weight='balanced'`

---

## 🛠️ Công Nghệ Sử Dụng

- **Framework:** TensorFlow / Keras, PyTorch
- **Môi trường:** Google Colab (GPU)
- **Ngôn ngữ:** Python 3
- **Thư viện chính:** `tensorflow`, `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---

## 📁 Cấu Trúc Thư Mục

```
├── resnet50.ipynb                        # ResNet50 fine-tuning trên APTOS 2019 (1 nhãn)
├── checkpoint_resnet152.ipynb            # Huấn luyện ResNet152, lưu checkpoint từ APTOS
├── mutil-task_checkpoint_resnet152.ipynb # MultiTask ResNet152 transfer từ checkpoint APTOS
├── CANet_ResNet50.ipynb                  # CANet với backbone ResNet50
├── CANet_ResNet152.ipynb                 # CANet với backbone ResNet152
├── CANet_IDRiD.ipynb                     # CANet huấn luyện trên tập IDRiD
└── README.md
```

---

## 🚀 Hướng Dẫn Chạy

### 1. Cài đặt thư viện
```bash
pip install tensorflow torch torchvision scikit-learn pandas numpy matplotlib
```

### 2. Chuẩn bị dữ liệu
Tải dữ liệu APTOS 2019 từ [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) và đặt vào thư mục `data/aptos2019/`.

### 3. Huấn luyện mô hình
```bash
# Chạy trên Google Colab hoặc môi trường có GPU
jupyter notebook notebooks/modelpbl4.ipynb
```

---

## 📚 Tài Liệu Tham Khảo

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
2. Li, X., et al. (2019). *CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading*. IEEE TMI.
3. APTOS 2019 Blindness Detection. Kaggle Competition.
4. Messidor Project. https://www.adcis.net/en/third-party/messidor/

---

## 📝 License

Dự án phục vụ mục đích học thuật — PBL4, Đại học Bách Khoa Đà Nẵng, 2026.
