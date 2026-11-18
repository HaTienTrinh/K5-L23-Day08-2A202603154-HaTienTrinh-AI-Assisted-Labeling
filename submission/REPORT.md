# Báo cáo Day 08 — 2A202603154 - Hà Tiến Trịnh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | motorbike 0.25; còn lại mặc định 0.3 (person, car: không ngưỡng ≤ 0.5 nào đạt recall 90%; bicycle, bus, truck: < 5 vật trong gold) | r02, r03, r06; `qc/threshold_sweep.md` (motorbike: P 92% / R 92% tại 0.25, 12 vật gold) <<lý do của bạn>> |
| three_wheeler | YOLOE-26s-seg (`yoloe-26s-seg.pt`), prompt: tuk-tuk, cyclo, cargo tricycle | 0.15 khi chạy Part 4 (sweep: P 78% / R 100%, 7 vật gold); v1.1 đổi thành 0.3 (CAL-02) | r04, r07 <<lý do của bạn>> |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "motorbike=0.25,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (ra 623 box trên 28 ảnh)
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`; r05: 2.69 s/ảnh trên RTX 3060). Lý do: <<>>
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? <<>>
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: <<>>

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 393 | 73.6% | 534 box máy |
| edited | 42 | 7.9% | 534 box máy |
| deleted | 56 | 10.5% | 534 box máy |
| added | 77 | 14.9% | 518 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** 71.0% (KTC 95%: 59–81%) / 83.0% (KTC 95%: 71–91%); TP 44, FP 18, FN 9 (mẫu số: 53 đối tượng gold, 4 ảnh)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 9/12 = 75.0% (KTC 95%: 47–91%); loại nào lọt: missing 2/3 (img_005, img_033), extra 1/3 (img_035); bắt đủ wrong_class 3/3, loose 3/3; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 6.25 phút/ảnh (25 phút / 4 ảnh) vs duyệt 4.58 phút/ảnh (110 phút / 24 ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk, 4 người) | SAM 2.1 base (`sam2.1_b.pt`) | 4 lần thử; bản cuối 1 dương + 9 âm + box | Không gồm người (mỗi người 1 click âm; người lái cần thêm click ở mặt, chân, bàn chân). Không có hàng hoá | Lần 1 click dương rơi vào người → mask ăn sang người lái. Lần 3 thêm click dương ở đèn pha → mask kéo cả xe máy đỗ phía sau; bỏ click đó, thêm 1 click âm vào xe máy. Bản cuối: đèn pha trước chưa trọn, còn vài đốm ở tay người lái, vài lỗ nhỏ ở ghế | `tuktuk_img013.jpg` |
| img_003 (ba gác chở hàng, trẻ em + người ngồi) | SAM 2.1 base | 3 lần thử; bản cuối 1 dương + 9 âm + box | Không gồm người (cần click âm riêng cho thân, chân bé gái và tay người đàn ông). **Có gồm hàng** (thanh long, rổ) | Lần 2 mask còn dính bé gái và tay người đàn ông. Mask cuối 2 polygon (mui tách khỏi thân xe) | `bagac_img003.jpg` |
| img_010 (xích lô, nhìn từ sau) | SAM 2.1 base | 3 lần thử; bản cuối 1 dương + 5 âm + box | Không gồm người đạp | Lần 1 click dương rơi vào người đạp. Lần 2 mask còn loang lổ trên áo khoác người đạp. Mask cuối 2 polygon | `xichlo_img010.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc "người là vật riêng" (mục 5, ca xích lô) giúp quyết định đặt click âm lên người; quy tắc lỗ thủng (4.2.2) áp dụng được cho khung lan can tuk-tuk và xích lô. Guideline **chưa nói** hàng chở trên xe có thuộc mask không (câu hỏi mở 3): mask ba gác img_003 đang gồm hàng → ca cần chốt ở v1.1. Guideline cũng không nói mask được phép nhiều polygon hay không (img_003, img_010 ra 2 polygon).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: 5 (`calibration_decisions.csv`: CAL-01, CAL-02 `loi_may`; CAL-03 `can_review`; CAL-04, CAL-05 `cau_hoi_mo`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu hỏi mở 2 (dãy xe đỗ dày gần camera, img_006), 3 (hàng trong thùng thuộc box / mask xe, img_003), 4 (giữ 20%, ảnh mốc img_036); ảnh ví dụ tuk-tuk bị gọi car / truck (img_013); recipe three_wheeler 0.15 → 0.3. Câu 1, 5, 6, 7 chưa chốt
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42: img_003, img_006, img_014, img_031, img_035; 114 đối tượng); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: 3/114 = 2.6% · theo **ảnh**: 2/5 = 40.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: thừa 3/3 lỗi (img_003 #1, #26 three_wheeler; img_035 #539 car trên cửa sổ tầng 2); không có lỗi loại khác trong mẫu
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 2/5 ảnh lỗi → cận trên 95% (Wilson) 77%. <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: Rework**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11 Home, RTX 3060 Laptop 6 GB (CUDA 12.4, torch 2.6.0), CVAT v2.81.0 Docker. YOLOE lần đầu tự cài CLIP từ git (r04 nạp 33 s). <<vướng mắc khác nếu có>>
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy toàn bộ trên GPU.
- Ca `can_review` khi gán gold (Part 1): `img_036.jpg`, box `person` [948, 341, 34×39 px]: người ngồi phía sau bên phải xe máy, bị che nhiều, không chắc còn thấy ≥ 20% (guideline v1.0 mục 4.1.2, câu hỏi mở 4) → nguyên liệu calibration Part 6.
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
