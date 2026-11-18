# Thông tin học viên

- Họ và tên: Hà Tiến Trịnh
- MSHV: 2A202603154
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: HaTienTrinh
- Repo bài nộp (URL): https://github.com/HaTienTrinh/K5-L23-Day08-2A202603154-HaTienTrinh-AI-Assisted-Labeling
- Commit chốt bài (hash): <<>>
- Máy: Windows 11 Home, GPU NVIDIA GeForce RTX 3060 Laptop 6 GB (CUDA 12.4, torch 2.6.0)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 73.6% / 7.9% / 10.5% (393 / 42 / 56; mẫu số: 534 box máy); added: 14.9% (77; mẫu số: 518 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 71.0% / 83.0% (TP 44, FP 18, FN 9; mẫu số recall: 53 đối tượng gold, 4 ảnh; lạc quan vì ngưỡng chọn trên chính gold)
- Bắt bẫy: 9/12 = 75.0% (KTC 95%: 47–91%)
- Thời gian: gán tay 6.25 phút/ảnh (25 phút / 4 ảnh); duyệt có pre-label 4.58 phút/ảnh (110 phút / 24 ảnh)
- Lỗi QC: 40.0% theo ảnh (2/5; cận trên 95% Wilson 77%), 2.6% theo đối tượng (3/114)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 40% > ngưỡng 2%, lỗi 'thua' lặp lại (3/3 lỗi), bắt bẫy 75% < 80%
