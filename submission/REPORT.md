# Báo cáo Day 08 — 2A202603154 - Hà Tiến Trịnh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | motorbike 0.25; còn lại mặc định 0.3 (person, car: không ngưỡng ≤ 0.5 nào đạt recall 90%; bicycle, bus, truck: < 5 vật trong gold) | r02, r03, r06; `qc/threshold_sweep.md` (motorbike: P 92% / R 92% tại 0.25, 12 vật gold). YOLO26m chạy nhanh (0.75 s/ảnh, r02) và bắt tốt người / xe máy ở phố đông. Không dùng ngưỡng chung 0.5 vì r03 mất 35% box (person 131→88, motorbike 55→32). Motorbike 0.25 là điểm F1 cao nhất (11/12 đúng, 1 thừa). Person: recall tối đa chỉ 79% (19/24, tại 0.1–0.15) mà precision tụt còn 58–63%, nên giữ 0.3 (P 68% / R 71%). Car: recall đứng ở 7/8 từ 0.1 đến 0.85, đổi ngưỡng chỉ đổi precision |
| three_wheeler | YOLOE-26s-seg (`yoloe-26s-seg.pt`), prompt: tuk-tuk, cyclo, cargo tricycle | 0.15 khi chạy Part 4 (sweep: P 78% / R 100%, 7 vật gold); v1.1 đổi thành 0.3 (CAL-02) | r04, r07. YOLO26 không có lớp này: tuk-tuk img_002 bị gọi `truck` hoặc bỏ sót hẳn (r02). YOLOE gõ chữ tìm được xe ba bánh mà YOLO26 sót (img_002 conf 0.93, img_032 conf 0.92). Chọn 0.15 vì là ngưỡng cao nhất còn giữ đủ 7/7 vật gold; lên 0.2 đã mất 1 vật (6/7) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "motorbike=0.25,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (ra 623 box trên 28 ảnh)
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`; r05: 2.69 s/ảnh trên RTX 3060). Lý do: là bản mặc định / khuyến nghị trong `docs/MODELS.md`, chạy được trên GPU 6 GB, nhận cả click dương / âm lẫn box nên sửa mask được bằng thêm click (`sam2_points.py`). Chỉ thử bản này, không so với SAM 1 hay SAM 3. Từ r05 rút ra: box→mask hàng loạt không ổn (4 mask vỡ 2 mảnh ở img_013, tỉ lệ mask/box chỉ 0.35–0.8, chậm ~3.6× YOLO26), nên chỉ dùng SAM theo kiểu click cho 3 vật ở Part 5
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Có hai lỗi kiểu này. (1) YOLO26 gọi tuk-tuk là `truck` (img_002) hoặc `car` (img_013, conf 0.41; r05), và gọi xe buýt img_013 là `car` với conf 0.72: box đúng chỗ nhưng sai lớp, tăng ngưỡng chỉ làm mất box chứ không đổi lớp. (2) YOLOE gọi xe tải là `three_wheeler`: img_011 (0.28, r04) và img_034 (0.85, lúc review phải đổi thành `truck`); ở img_003 hai box `three_wheeler` 0.61 và 0.25 thật ra là 2 xe tải ở xa (review đổi thành `truck`). Cách sửa: lấy `three_wheeler` chỉ từ YOLOE (CAL-01), khi review bắt buộc đi bước 3 (sai nhãn) của checklist và đổi lớp bằng tay; trong `review_stats.md` có 13 trường hợp đổi lớp như vậy
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm cao mà sai: img_034 `three_wheeler` 0.85 thực ra là xe tải; img_013 `car` 0.72 thực ra là xe buýt; img_003 có 3 box `three_wheeler` (0.90, 0.80, 0.21) chồng lên cùng một chiếc ba gác, merge không gộp được vì 2 box sau chỉ ôm một phần xe (giữ 1 box, sửa lại cho ôm đủ). Điểm thấp mà đúng: img_006 xe máy #63 conf 0.28 (ảnh này QC không có lỗi); img_003 #27 `three_wheeler` 0.18 sát mép trái (QC không đánh lỗi). Ngược lại, ngay trong img_003 hai box `three_wheeler` 0.20 (#1) và 0.16 (#26) là box thừa: cùng một khoảng điểm có cả đúng lẫn sai, nên điểm không phải xác suất đúng
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng chọn và chấm trên cùng 4 ảnh, 53 vật. Nhiều lớp quá ít vật để kết luận (bicycle 1, bus 1, truck 0 vật gold). Three_wheeler chỉ 7 vật, ngưỡng 0.15 khác 0.2 đúng 1 vật. Trên batch, three_wheeler bị xoá 23 box so với 12 box giữ nguyên: dấu hiệu precision 78% trên gold không giữ được ngoài gold

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 393 | 73.6% | 534 box máy |
| edited | 42 | 7.9% | 534 box máy |
| deleted | 56 | 10.5% | 534 box máy |
| added | 77 | 14.9% | 518 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? **three_wheeler**: 42 box máy, chỉ 12 accepted (28.6%), 23 deleted (54.8%), 7 edited. Nguyên nhân: ngưỡng 0.15 thấp nên lọt nhiều box rác (img_003 có 8 box `three_wheeler` máy, cuối cùng giữ 4); YOLOE gọi xe tải là xe ba bánh (img_034, img_003); và nhiều box chỉ ôm một phần xe (img_003). Truck cũng tệ (1/5 accepted, 4/5 deleted) nhưng ít box. Person thì ngược lại: accepted 217/241 = 90.0% box máy, nhưng `added` cao nhất (34): máy sót người nhỏ / bị che. Motorbike bị sửa nhiều nhất (edited 22/141 = 15.6%). Trừ 13 ca đổi lớp, số vật máy sót thật ≈ 64
- So với gold: precision / recall của **máy** 71.0% (KTC 95%: 59–81%) / 83.0% (KTC 95%: 71–91%); TP 44, FP 18, FN 9 (mẫu số: 53 đối tượng gold, 4 ảnh)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 9/12 = 75.0% (KTC 95%: 47–91%); loại nào lọt: missing 2/3 (img_005, img_033), extra 1/3 (img_035); bắt đủ wrong_class 3/3, loose 3/3; vì sao (bạn đã bỏ bước nào của checklist)? Cả hai bẫy `missing` lọt đều nằm **sát mép phải** ảnh và **nằm cạnh một vật đã có box**: img_033 xe máy (x 922–1000 / 1024 px) ngay dưới người #475 đã được giữ; img_005 người (x 726–767 / 768 px) cạnh xe máy #46. Tức là em chỉ kiểm vật quanh box có sẵn rồi qua ảnh, bỏ bước 0 (ẩn nhãn, quét đếm trước) và làm hời hợt bước 1 (bỏ sót, nhất là mép ảnh). Bẫy `extra` lọt ở img_035 là box `car` nằm trên cửa sổ tầng 2: bước 2 (thừa) bị bỏ qua vì box trông "hợp lý" (conf hiển thị 0.49). Chính box này sau đó QC bắt được (#539)
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Bắt đủ 6/6 bẫy `wrong_class` + `loose` nhưng chỉ 3/6 bẫy `missing` + `extra`: khi có box thì em có nhìn kỹ box đó, nhưng chỗ không có box thì không nhìn (lỗi **bỏ qua**), và box thừa trông hợp lý thì giữ (lỗi **làm theo**). Accepted 73.6% vì vậy không chứng minh máy tốt: một phần là box em giữ mà không kiểm đủ. Bằng chứng thêm: cả 3 lỗi trong mẫu QC đều là box có sẵn mà em giữ nguyên không sửa: 2 box máy (img_003 #1, #26) và 1 box bẫy `extra` (img_035 #539, chính là bẫy đã lọt). Bẫy lọt qua lúc duyệt và chỉ bị bắt ở QC — đây là neo nhãn, đúng kiểu thiên kiến tự động hóa

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 6.25 phút/ảnh (25 phút / 4 ảnh) vs duyệt 4.58 phút/ảnh (110 phút / 24 ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): theo ảnh, duyệt nhanh hơn 1.67 phút/ảnh (≈ 27%). Nhưng ảnh batch đông hơn (555 đối tượng / 24 ảnh ≈ 23 vật/ảnh so với 63 / 4 ≈ 16 vật/ảnh gold; cả hai số đều tính cả `ignore_region`: gold 53 vật + 10, batch 530 vật + 25, nên 63 ở đây khác mẫu số 53 của mục 2), nên tính theo vật: gán tay 25/63 ≈ 0.40 phút/vật, duyệt 110/555 ≈ 0.20 phút/vật (≈ nhanh gấp đôi). Hai thiên lệch ngược chiều: đã quen cảnh sau khi gán gold làm duyệt nhanh hơn thực tế; còn thời gian duyệt bao gồm cả việc sửa 12 bẫy nhân tạo. Ngoài ra duyệt nhanh một phần vì bỏ bước 0 (mục 3), tức là tốc độ này đi kèm 3/12 bẫy lọt
- Kết luận: **có, nhưng chỉ cho person / motorbike / car**; three_wheeler (và truck) không nên tin pre-label. Dựa trên: accepted theo lớp person 217/241 = 90.0%, car 45/51 = 88.2%, motorbike 109/141 = 77.3%, trong khi three_wheeler 12/42 = 28.6% (dưới mốc 50% của `CALIBRATION_QC.md`: sửa lâu hơn vẽ mới), truck 1/5; thời gian duyệt ≈ 0.20 so với 0.40 phút/vật. Nhưng máy vẫn sót: recall trên gold 83.0% (44/53), added 77/518 = 14.9%, nên pre-label chỉ đáng dùng khi người duyệt thật sự quét tìm vật sót
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Ẩn nhãn máy, quét và đếm vật theo lớp ở 100% ảnh trước khi hiện nhãn; ghi số đếm của từng ảnh vào ghi chú và so với số đối tượng cuối (kiểm bằng log: 24/24 ảnh có số đếm).
  2. Trước khi chuyển ảnh, quét riêng 4 dải mép ảnh và vùng sát cạnh mỗi người / xe đã có box (đúng chỗ 2 bẫy `missing` lọt). Mục tiêu đo được: lần sau bắt đủ 3/3 bẫy `missing` và tỉ lệ bắt bẫy ≥ 10/12.
  3. Mọi box `three_wheeler` / `truck` và mọi box có conf < 0.3 phải zoom kiểm, không được accept bằng một lần nhìn; nghỉ 3–5 phút sau mỗi 10 ảnh và ghi dòng `break` vào `time_log.csv`. Kiểm bằng QC: 0 lỗi `thua` ở các box này trong mẫu QC vòng sau.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk, 4 người) | SAM 2.1 base (`sam2.1_b.pt`) | 4 lần thử; bản cuối 1 dương + 9 âm + box | Không gồm người (mỗi người 1 click âm; người lái cần thêm click ở mặt, chân, bàn chân). Không có hàng hoá | Lần 1 click dương rơi vào người → mask ăn sang người lái. Lần 3 thêm click dương ở đèn pha → mask kéo cả xe máy đỗ phía sau; bỏ click đó, thêm 1 click âm vào xe máy. Bản cuối: đèn pha trước chưa trọn, còn vài đốm ở tay người lái, vài lỗ nhỏ ở ghế | `tuktuk_img013.jpg` |
| img_003 (ba gác chở hàng, trẻ em + người ngồi) | SAM 2.1 base | 3 lần thử; bản cuối 1 dương + 9 âm + box | Không gồm người (cần click âm riêng cho thân, chân bé gái và tay người đàn ông). **Có gồm hàng** (thanh long, rổ) | Lần 2 mask còn dính bé gái và tay người đàn ông. Mask cuối 2 polygon (mui tách khỏi thân xe) | `bagac_img003.jpg` |
| img_010 (xích lô, nhìn từ sau) | SAM 2.1 base | 3 lần thử; bản cuối 1 dương + 5 âm + box | Không gồm người đạp | Lần 1 click dương rơi vào người đạp. Lần 2 mask còn loang lổ trên áo khoác người đạp. Mask cuối 2 polygon | `xichlo_img010.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc "người là vật riêng" (mục 5, ca xích lô) giúp quyết định đặt click âm lên người; quy tắc lỗ thủng (4.2.2) áp dụng được cho khung lan can tuk-tuk và xích lô. Guideline **chưa nói** hàng chở trên xe có thuộc mask không (câu hỏi mở 3): mask ba gác img_003 đang gồm hàng → ca cần chốt ở v1.1. Guideline cũng không nói mask được phép nhiều polygon hay không (img_003, img_010 ra 2 polygon).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: không thử; chỉ làm mask offline bằng `sam2_points.py`.

## 6. Calibration (Part 6)

- Số quyết định: 5 (`calibration_decisions.csv`: CAL-01, CAL-02 `loi_may`; CAL-03 `can_review`; CAL-04, CAL-05 `cau_hoi_mo`); quan trọng nhất (dẫn `CAL-xx`): **CAL-02** (nâng ngưỡng `three_wheeler` 0.15 → 0.3). Đây là lỗi lặp duy nhất QC thấy: 2/3 lỗi QC là box `three_wheeler` thừa ở img_003 với conf 0.20 và 0.16, đều dưới 0.3, và three_wheeler có deleted 23 > accepted 12. Đánh đổi: trên gold recall giảm 100% → 71% (5/7), precision tăng 78% → 83%; ở img_003 box #27 (0.18) QC coi là đúng cũng sẽ bị lọc mất
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: chốt câu hỏi mở 2 (dãy xe đỗ dày gần camera, img_006), 3 (hàng trong thùng thuộc box / mask xe, img_003), 4 (giữ 20%, ảnh mốc img_036); ảnh ví dụ tuk-tuk bị gọi car / truck (img_013); recipe three_wheeler 0.15 → 0.3. Câu 1, 5, 6, 7 chưa chốt
- Nguyên nhân gốc: tách theo ca. **Máy** sai: CAL-01 (tuk-tuk → car / truck), CAL-02 (box `three_wheeler` thừa) — không cần sửa chữ guideline, chỉ thêm ảnh ví dụ và đổi recipe. **Guideline mơ hồ**: CAL-04 (hàng trên xe), CAL-05 (dãy xe đỗ dày gần camera), CAL-03 (ngưỡng 20% khó ước lượng, giữ quy tắc nhưng thêm ảnh mốc) — các ca này mới vào v1.1 thành quy tắc. Còn 3 lỗi QC là lỗi **của em** khi duyệt (giữ nguyên 2 box máy sai và 1 box bẫy `extra`), không phải do guideline: v1.0 đã đủ rõ để xoá chúng

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42: img_003, img_006, img_014, img_031, img_035; 114 đối tượng); người kiểm: Trịnh - 2A202603154 (**hạn chế**: chính em vừa là người duyệt batch, nên dễ bỏ sót cùng chỗ như lúc duyệt).
- Lỗi theo **đối tượng**: 3/114 = 2.6% · theo **ảnh**: 2/5 = 40.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: thừa 3/3 lỗi (img_003 #1, #26 three_wheeler; img_035 #539 car trên cửa sổ tầng 2); không có lỗi loại khác trong mẫu. Lưu ý: #539 là bẫy `extra` lọt lúc duyệt (bbox [361, 64, 115, 68], conf 0.49 trùng `fake_box` trong `traps_key.json`), không phải box máy. Bỏ lỗi nhân tạo này ra thì còn 2/114 = 1.8% đối tượng lỗi đến từ box máy, nhưng theo ảnh vẫn 1/5 = 20% > 2%, nên quyết định không đổi
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): 2/5 ảnh lỗi → cận trên 95% (Wilson) 77%. Tức là với 5 ảnh, tỉ lệ ảnh lỗi thật có thể từ thấp đến 77%; ngưỡng 2% không thể chứng minh được dù mẫu sạch hết (0/5 vẫn cho cận trên 95% Wilson ≈ 43%; quy tắc 3 cho 3/5 = 60% nhưng chỉ là xấp xỉ khi mẫu lớn). Mẫu cũng có 0 ảnh `can_review` vì khi duyệt batch em không tick `can_review` box nào, nên QC chỉ là 5 ảnh ngẫu nhiên. Tỉ lệ theo đối tượng 2.6% (3/114) trông thấp nhưng 3 lỗi dồn vào 2 ảnh và cùng một loại, nên đây là lỗi hệ thống chứ không phải ngẫu nhiên
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): script đếm được số lỗi theo đối tượng / ảnh, tính cận trên Wilson, accepted / added, bắt bẫy và kiểm file hợp lệ (`validate_labels.py`). Script **không** biết box #539 ở img_035 là một chiếc ô tô "ma" trên cửa sổ: phải nhìn ảnh mới thấy; cũng không kiểm được box có ôm sát không, vật cả máy lẫn em cùng sót ngoài 4 ảnh gold, và em có thật sự quét trước (bước 0) hay không — chỉ đoán gián tiếp qua bẫy lọt

**Quyết định: Rework**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): không Hold vì guideline đã đủ để xử lý các lỗi tìm thấy (3 lỗi đều là box thừa mà v1.0 đã cho xoá; câu hỏi mở 2, 3, 4 đã chốt ở v1.1) và dữ liệu không hỏng. Rework vì: ảnh lỗi 2/5 = 40.0% > ngưỡng 2%; lỗi lặp lại (`thua` 3/3 lỗi, 2 lỗi cùng lớp `three_wheeler` conf thấp); bắt bẫy 9/12 = 75.0% < 80% nên lô duyệt chưa đáng tin. Không Release được
Nếu Rework / Hold, việc cần làm tiếp: (1) xoá 3 box lỗi (img_003 #1, #26; img_035 #539), export `labels/qc_fixed.json`; (2) duyệt lại cả 24 ảnh theo 3 quy tắc ở mục 4, ưu tiên mép ảnh và mọi box `three_wheeler` conf < 0.3, sửa luôn 2 bẫy `missing` còn sót (img_033, img_005); (3) chạy lại `build_release.py`, `qc_sample.py --force`, `qc_report.py` và ghi kết quả vòng 2 vào báo cáo; (4) lần sau tạo pre-label bằng `--thr "motorbike=0.25,three_wheeler=0.3"` theo v1.1

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không):
  - Claude Code
  - Việc AI làm: chạy lệnh script theo hướng dẫn, xử lý lỗi cài đặt, tạo commit.
    Sau khi viết xong báo cáo, em nhờ AI rà soát: đối chiếu từng con số với `review_stats.md`,
    `threshold_sweep.csv`, `qc_report_auto.md`, `time_log.csv`, `traps_key.json` và file nhãn.
  - Không dùng AI để: gán nhãn gold, duyệt batch trong CVAT, chọn quyết định Rework.
    AI chỉ đọc file JSON / CSV / MD / chạy lệnh script theo hướng dẫn.
  - Cách kiểm tra: tự mở `batch_reviewed.json` và `traps_key.json`, thấy bbox và conf 0.49 của #539
    trùng `fake_box`; xem giờ lưu file `gold_manual.json` (11:49) và `work/smoke` (17:54);
    đếm lại lớp `ignore_region` trong `gold_manual.json`; tự tính lại Wilson cho 0/5.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không: 4 ảnh gold (img_001, img_012, img_020, img_036) gán tay trong CVAT; file `gold_manual.json` lưu lúc 11:49 ngày 2026-11-17, còn lần chạy model đầu tiên (smoke r01, `work/smoke`, ảnh img_011 không thuộc gold) lúc 17:54 cùng ngày và các run Part 2 từ 18:52. (Do không đọc kỹ yêu cầu cần commit theo từng giai đoạn nên commit CP0–CP3 được tạo cùng lúc 19:26, nên thứ tự commit này không dùng làm bằng chứng).
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11 Home, RTX 3060 Laptop 6 GB (CUDA 12.4, torch 2.6.0), CVAT v2.81.0 Docker. YOLOE lần đầu tự cài CLIP từ git (r04 nạp 33 s). Khi review ảnh dày, CVAT trên trình duyệt bị giật (RAM máy còn trống ~1.3/15.7 GB); tắt hiển thị text của nhãn và đóng bớt ứng dụng để giảm tải.
- Dùng cache `data/cache/` thay vì tự chạy? Không, tự chạy toàn bộ trên GPU.
- Ca `can_review` khi gán gold (Part 1): `img_036.jpg`, box `person` [948, 341, 34×39 px]: người ngồi phía sau bên phải xe máy, bị che nhiều, không chắc còn thấy ≥ 20% (guideline v1.0 mục 4.1.2, câu hỏi mở 4) → nguyên liệu calibration Part 6.
- Lỗi / bug script đã gặp (nếu có): không gặp bug script.
- Một điều bạn làm khác lần sau: bấm giờ và ghi số đếm bước 0 cho từng ảnh ngay khi duyệt, và tick `can_review` cho ca không chắc ngay trong lúc duyệt batch (lần này 0 box), để QC có ca khó mà kiểm thay vì chỉ có ảnh ngẫu nhiên. Với car, thử ngưỡng riêng cao hơn (sweep: 0.75 cho P 88% / R 88% so với P 50% ở 0.3) thay vì để mặc định.
