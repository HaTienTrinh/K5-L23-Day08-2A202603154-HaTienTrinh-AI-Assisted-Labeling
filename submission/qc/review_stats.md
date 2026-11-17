# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 534; đối tượng sau duyệt: 518; ngưỡng accepted: IoU >= 0.9
- Đã **loại 24 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 393 | 393/534 = 73.6% | 534 box máy |
| edited (sửa) | 42 | 42/534 = 7.9% | 534 box máy |
| deleted (xoá) | 56 | 56/534 = 10.5% | 534 box máy |
| added (vẽ thêm) | 77 | 77/518 = 14.9% | 518 đối tượng sau duyệt |

Trong số deleted + added, **13 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 64.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 217 | 11 | 13 | 34 |
| bicycle | 6 | 0 | 2 | 4 |
| motorbike | 109 | 22 | 10 | 24 |
| car | 45 | 2 | 4 | 2 |
| bus | 3 | 0 | 0 | 5 |
| truck | 1 | 0 | 4 | 4 |
| three_wheeler | 12 | 7 | 23 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 53 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 44 | 18 | 9 | 71.0% (KTC 95%: 59–81%) | 83.0% (KTC 95%: 71–91%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 9/12 = 75.0% (KTC 95%: 47–91%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 3/3
  - extra: 2/3
  - loose: 3/3
  - missing: 1/3
  - lọt: img_033.jpg (missing), img_035.jpg (extra), img_005.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
