# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đặng Quốc Hiệp
- MSSV: 2A202602755
- Email: dangquochiep2908@gmail.com
- Link repo (fork): https://github.com/QuocHiep123/K4-Track4-Day23-DangQuocHiep-2A202602755-Sensor-Fusion.git
- Commit hash nộp (`git rev-parse HEAD`): 3c1b2f7d68981ff6e433b6d2e54febb0724aa5f6

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, frames [0, 198], segment training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, seed 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision = 0.9700934579439252 (97.01%), recall = 0.7004048582995951 (70.04%), tp = 519, fp = 16, fn = 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.15032268781360134, matches = 502, sum_sq_err = 11.343649056695735, ghost_track_frames = 0, missed_gt_frames = 239, mean_confirmed_tracks = 2.522613065326633
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.1358667883353908, matches = 502, sum_sq_err = 9.266811654632091, ghost_track_frames = 0, missed_gt_frames = 239, mean_confirmed_tracks = 2.522613065326633
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Fused tracking đạt RMSE = 0.1359 m, cải thiện độ chính xác rõ rệt so với Lidar-only RMSE = 0.1503 m (giảm ~9.6% sai số vị trí RMSE và giảm 18.3% tổng bình phương sai số sum_sq_err từ 11.34 xuống 9.27). Đồng thời cả hai chế độ đều duy trì 502/502 matches, 0 ghost tracks, 239 missed GT frames và trung bình 2.52 confirmed tracks/frame. Điều này chứng minh việc bổ sung camera 2D thông qua phép chiếu phối cảnh và cập nhật EKF tuần tự đã tinh chỉnh ước lượng vị trí đối tượng chính xác hơn đáng kể mà không làm mất liên kết hay sinh ra vệt ảo (ghost track).

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
- Đo Lidar: Vector đo $z \in \mathbb{R}^3$ là tọa độ vị trí 3D trực tiếp $[x, y, z]^T$ trong hệ quy chiếu xe (vehicle frame). Ma trận quan sát là tuyến tính $H_{lidar} = [I_{3 \times 3} \mid 0_{3 \times 3}]$ kích thước $3 \times 6$. Ma trận hiệp phương sai sai số đo $R_{lidar} \in \mathbb{R}^{3 \times 3}$ có đơn vị mét bình phương ($m^2$), phản ánh độ chính xác trực tiếp trong không gian Euclid 3D.
- Đo Camera: Vector đo $z \in \mathbb{R}^2$ là tọa độ pixel 2D $[c_x, c_y]^T$ trên mặt phẳng ảnh camera FRONT. Ma trận hiệp phương sai $R_{cam} \in \mathbb{R}^{2 \times 2}$ có đơn vị pixel bình phương ($px^2$). Mô hình đo là hàm phi tuyến $h(x)$ (phép chiếu phối cảnh pinhole kết hợp ma trận biến đổi Extrinsics và Intrinsics), do đó cần tính ma trận Jacobian $H_{cam} = \frac{\partial h}{\partial x} \in \mathbb{R}^{2 \times 6}$ tại trạng thái dự đoán để thực hiện cập nhật EKF.

2. Vì sao cần gating Mahalanobis trước khi gán?
- Gating Mahalanobis sử dụng khoảng cách chuẩn hóa $d_M = \sqrt{\gamma^T S^{-1} \gamma}$ và ngưỡng kiểm định phân phối $\chi^2$:
  - Lọc bỏ các cặp đo đạc - track không hợp lý về mặt thống kê xác suất (outliers, clutter, false alarms, hoặc các đối tượng ở quá xa nhau vượt quá độ bất định của bộ lọc).
  - Thu hẹp bài toán gán liên kết (association cost matrix): các phần tử nằm ngoài gate được gán chi phí vô cùng (INF), ngăn chặn việc gán sai chéo (cross-assignment) giữa các phương tiện đi sát nhau.
  - Khác với khoảng cách Euclid ($L_2$) bỏ qua độ bất định và hướng biến thiên, khoảng cách Mahalanobis chuẩn hóa sai số innovation $\gamma$ theo ma trận hiệp phương sai $S = H P H^T + R$, giúp đánh giá chính xác độ tương đồng theo elip phân bố xác suất của từng track.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
- Pipeline là **track-then-fuse** (cụ thể là centralized sequential measurement-to-track fusion):
  - Trong fuse-then-track: các cảm biến Lidar và Camera được gom/dung hợp dữ liệu thô (raw point cloud + RGB image) hoặc dung hợp ở mức feature/detection trước, tạo ra danh sách detection chung rồi mới đưa vào tracker để cập nhật vết.
  - Trong track-then-fuse: Hệ thống duy trì các track độc lập. Ở mỗi chu kỳ, Kalman Filter thực hiện dự đoán (predict) trên các track hiện có. Sau đó, các phép đo Lidar được liên kết và cập nhật trước vào track; tiếp theo các phép đo Camera được chiếu và liên kết vào chính các track này để cập nhật EKF tuần tự (sequential measurement update).
  - Bằng chứng trên mã nguồn và log: Trong `run_lab.py`, trong mỗi frame:
    1) Lặp `for track in manager.track_list: KF.predict(track)`,
    2) Gọi `assoc.associate_and_update(manager, observations, KF, lidar_sensor)` với phép đo lidar 3D để cập nhật và quản lý vòng đời track,
    3) Gọi `assoc.associate_and_update(manager, observations, KF, camera_sensor)` với phép đo camera 2D để tinh chỉnh thêm trạng thái cho các confirmed tracks nằm trong Camera FOV.
    Trên log `grade_run.log`, số lượng `valid_gt` và các track `confirmed` đồng nhất ở cả 2 mode, và camera chỉ cải thiện độ chính xác vị trí `sum_sq_err` của các track sẵn có thay vì tạo track mới.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
- Innovation (residual) là $\gamma = z - h(\hat{x})$ (sai số giữa vị trí đo trên ảnh và vị trí dự đoán chiếu từ trạng thái 3D của track).
- Khi camera bị lệch calibration (lệch extrinsic hoặc intrinsic):
  - Innovation $\gamma$ sẽ xuất hiện sai số hệ thống (systematic bias / offset), khiến kỳ vọng sai số không còn là nhiễu trắng trung bình bằng 0 ($E[\gamma] \neq 0$).
  - Khoảng cách Mahalanobis $d_M = \sqrt{\gamma^T S^{-1} \gamma}$ của các cặp đúng sẽ tăng cao vượt ngưỡng $\chi^2$ gate. Phép đo camera sẽ liên tục bị từ chối liên kết (missed associations / gating rejection).
  - Nếu gate quá lỏng hoặc bị ép cập nhật, residual sai lệch hệ thống sẽ kéo trạng thái ước lượng vị trí 3D của xe lệch xa khỏi thực tế, làm tăng đột biến sai số RMSE hoặc dẫn tới hiện tượng phân kỳ bộ lọc (filter divergence).

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
- Cần tham số `sensor` tường minh khi observations rỗng (`[]`):
  - Khi không có observation nào, nếu `sensor.name == "lidar"`, hệ thống hiểu rằng toàn bộ các track đang theo dõi đều không nhận được phản xạ lidar (mặc dù lidar quét 360 độ). Do đó bắt buộc phải gọi `track.handle_miss()`, giảm track score và xét xóa track (`delete_track`) nếu miss liên tiếp.
  - Ngược lại, nếu `sensor.name == "camera"`, việc không có quan sát camera chỉ đơn giản là không có xe nào trong tầm nhìn hẹp phía trước (FOV) hoặc camera bị che khuất tạm thời. Hệ thống tuyệt đối không được phạt điểm (không giảm score) hay xóa track của xe.
- Vì sao Lidar quyết định score/init/delete còn Camera chỉ EKF update:
  - Lidar cung cấp trực tiếp tọa độ 3D $[x, y, z]$ với độ sâu và độ bao phủ toàn cảnh 360 độ quanh xe, cung cấp thông tin đầy đủ và đáng tin cậy để khởi tạo trạng thái 3D ban đầu của track (`init_track_state_from_meas`) và kiểm soát toàn bộ vòng đời tồn tại của đối tượng.
  - Camera đơn 2D bị mất chiều sâu (độ sâu bị suy biến), không thể khởi tạo một track 3D hoàn chỉnh nếu không có cảm biến đo khoảng cách hoặc mô hình giả định 3D. Ngoài ra camera chỉ quan sát một góc giới hạn (Front camera ~50°), nên không thể đại diện cho toàn bộ không gian xung quanh xe. Tuy nhiên độ phân giải góc của camera rất cao, cực kỳ hiệu quả để cập nhật EKF (refine bearing and lateral position).

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
- Điều kiện xác nhận (`confirmed`):
  - Track mới khởi tạo ở trạng thái `initialized` với `score = score_init` (0.5).
  - Khi được liên kết thành công với đo đạc Lidar, điểm số được tăng theo công thức Bayes/exponential update: `score = 1.0 - (1.0 - score) * (1.0 - p_hit)`.
  - Khi `score >= score_thresh` (0.79 hoặc 0.8), track chính thức chuyển sang trạng thái `confirmed`.
- Giữ confirmed sau miss:
  - Khi một track đã ở trạng thái `confirmed` bị miss (không liên kết được với phép đo Lidar ở frame hiện tại), trạng thái của nó vẫn được duy trì là `confirmed` (không bị giáng cấp về `initialized`), trong khi `score` bị suy giảm dần: `score = score * (1.0 - p_miss)`. Điều này giúp track duy trì sự ổn định, vượt qua hiện tượng che khuất tạm thời (temporary occlusion).
- Điều kiện xóa track (`should_delete_track`):
  - Track `initialized`: bị xóa ngay khi `score < score_thresh` (ngay lần miss đầu tiên) hoặc `misses > 0`.
  - Track `confirmed`: bị xóa khi `score < score_thresh` (sau nhiều lần miss liên tiếp khiến score tụt xuống dưới ngưỡng) HOẶC số lần miss liên tiếp vượt quá ngưỡng tối đa cho phép (`track.misses > max_misses`, mặc định 3 frame).

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI (Claude 3.7 Sonnet / Gemini 2.5 Flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích codebase, hoàn thiện công thức ma trận EKF (Part E), phép chiếu pinhole 2D (Part G), ma trận chi phí Mahalanobis (Part F), logic vòng đời track (Part H), và giải thích các câu hỏi lý thuyết.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ test suite `pytest student/tests` (đạt 128/128 test passed), chạy thực tế pipeline Waymo trên 198 frame (`fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0`), kiểm tra tính hợp lệ của log qua `tools/check_submission.py`, và đối chiếu công thức toán học với slide lý thuyết bài giảng.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
