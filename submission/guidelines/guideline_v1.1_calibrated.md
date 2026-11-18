# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603154 - Hà Tiến Trịnh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc **gần camera**: từng xe hay `ignore_region`? | Xe nhìn thấy **≥ 50%** (thường là xe đầu / cuối dãy) gán từng chiếc. Phần giữa dãy chồng chéo, không tách được từng xe → **một** `ignore_region` bao phần đó. Không phủ `ignore_region` lên xe đã thấy ≥ 50%. Bổ sung mục 4.1.6 (v1.0 chỉ nói vật < 20 px ở xa) | img_006 | CAL-05 |
| 3 | Hàng nhô ra: box / mask xe có gồm hàng không? | Hàng **nằm trong hoặc đè lên thùng xe** (thanh long, rổ trên ba gác) thuộc box **và** mask của xe. Hàng nhô ra khỏi xe > 20% chiều dài xe: **không** tính (giữ mục 4.1.5). Áp dụng cho cả 4.1 (box) và 4.2 (mask) | img_003 | CAL-04 |
| 4 | Người ngồi trong / sau xe bị che: ngưỡng "≥ 20%" có hợp lý không? | **Giữ** ngưỡng 20% diện tích của mục 4.1.2. Làm rõ bằng ảnh mốc: người ngồi sau xe máy img_036 (box 34×39 px) là ca sát ngưỡng; không chắc → chọn theo ước lượng tốt nhất và tick `can_review` | img_036 | CAL-03 |

**Chưa chốt** (chưa có ca được ghi nhận trong lô 28 ảnh này để làm ảnh ví dụ, giữ nguyên là câu hỏi mở cho vòng sau):

- Câu 1: xe đạp điện / xe máy điện nhỏ là `bicycle` hay `motorbike`.
- Câu 5: kính chắn gió / cửa sổ xe có thuộc mask xe không.
- Câu 6: box máy chồng nhau quá nhiều trong ảnh đêm: gộp hay giữ.
- Câu 7: mức `conf` tối thiểu để chấp nhận box máy không cần sửa (CAL-02 chỉ đổi ngưỡng **lọc** pre-label, không phải ngưỡng chấp nhận).

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_013, img_002 | Tuk-tuk có mui, có người ngồi | YOLO26: gọi tuk-tuk là `car` (img_013, 0.41) / `truck` (img_002), có khi bỏ sót hẳn (tuk-tuk lớn img_002) | Sửa thành `three_wheeler`; lớp này lấy từ YOLOE, không từ YOLO26 | Không đổi quy tắc (mục 2 v1.0 đã rõ: xe ba bánh → `three_wheeler`); thêm img_013 vào bảng ca khó mục 5 làm ảnh ví dụ (CAL-01) |
| img_011, img_003 | Box `three_wheeler` điểm thấp không có xe ba bánh thật | YOLOE (prompt tuk-tuk / cyclo / cargo tricycle): gọi xe tải là `three_wheeler` (img_011, 0.28); box thừa #1 (0.20) và #26 ở img_003 lọt qua QC | Nâng ngưỡng lọc `three_wheeler` từ 0.15 lên **0.3** (gold: precision 83%, recall 71%), chấp nhận sót nhiều hơn | Không đổi quy tắc gán; ghi vào recipe / ngưỡng: `--thr "motorbike=0.25,three_wheeler=0.3"` (CAL-02) |
| img_036 | Người ngồi sau xe máy, bị che nhiều | — (ca `can_review` khi gán tay gold) | Giữ ngưỡng 20%, dùng ảnh này làm mốc | Mục 4.1.2: thêm ảnh mốc img_036 (CAL-03) |
| img_003 | Ba gác chở thanh long, rổ; mask SAM gồm cả hàng | SAM 2.1 (click): mask tự gồm hàng trong thùng | Hàng trong thùng thuộc box và mask xe | Mục 4.1.5 và 4.2 (CAL-04) |
| img_006 | Dãy xe máy đỗ dày gần camera | — (ca phát hiện khi QC: vùng `ignore_region` #71) | Xe ≥ 50% gán riêng, phần chồng chéo bên trong 1 `ignore_region` | Mục 4.1.6 và bảng ca khó mục 5 (CAL-05) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 2 (dãy xe đỗ dày gần camera), 3 (hàng trên xe thuộc box / mask nếu nằm trong thùng), 4 (giữ 20%, thêm ảnh mốc); thêm ảnh ví dụ tuk-tuk bị máy gọi car / truck; recipe: ngưỡng `three_wheeler` 0.15 → 0.3. Câu 1, 5, 6, 7 chưa chốt | CAL-01, CAL-02, CAL-03, CAL-04, CAL-05 |
