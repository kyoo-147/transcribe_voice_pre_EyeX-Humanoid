# Server local cho mô hình whisper openai
Dự án này cải thiện thời gian thực để khởi chạy server nhận đầu vào giọng nói và chuyển chúng thành văn bản

## Cài đặt
```bash
 bash scripts/setup.sh
```

```bash
 pip install whisper-live
```

## Bắt đầu
Máy chủ hỗ trợ hai phần phụ trợ `faster_whisper` và `tensorrt`. Nếu chạy chương trình phụ trợ `tensorrt`, hãy làm theo [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md)

### Chạy máy chủ
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) backend
```bash
python3 run_server.py --port 9090 \
                      --backend faster_whisper
  
# running with custom model
python3 run_server.py --port 9090 \
                      --backend faster_whisper
                      -fw "/path/to/custom/faster/whisper/model"
```

- Chương trình phụ trợ TensorRT. Hiện tại, chúng tôi khuyên bạn chỉ nên sử dụng thiết lập docker cho TensorRT. Theo dõi [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) và hoạt động như mong đợi. Đảm bảo xây dựng Công cụ TensorRT của bạn trước khi chạy máy chủ với chương trình phụ trợ TensorRT.
```bash
# Run English only model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small_en

# Run Multilingual model
python3 run_server.py -p 9090 \
                      -b tensorrt \
                      -trt /home/TensorRT-LLM/examples/whisper/whisper_small \
                      -m
```


### Chạy máy khách
- Đang khởi tạo ứng dụng khách:
```python
from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="small",
  use_vad=False,
)
```
Nó kết nối với máy chủ chạy trên localhost tại cổng 9090. Sử dụng mô hình đa ngôn ngữ, ngôn ngữ phiên âm sẽ tự động được phát hiện. Bạn cũng có thể sử dụng tùy chọn ngôn ngữ để chỉ định ngôn ngữ đích cho bản phiên âm, trong trường hợp này là tiếng Anh ("en"). Tùy chọn dịch phải được đặt thành `Đúng` nếu chúng tôi muốn dịch từ ngôn ngữ nguồn sang tiếng Anh và `Sai` nếu chúng tôi muốn phiên âm bằng ngôn ngữ nguồn.
- Phiên âm một tập tin âm thanh:
```python
client("tests/jfk.wav")
```

- Để chép lại từ micrô:
```python
client()
```

- Để phiên âm từ luồng HLS:
```python
client(hls_url="http://as-hls-ww-live.akamaized.net/pool_904/live/ww/bbc_1xtra/bbc_1xtra.isml/bbc_1xtra-audio%3d96000.norewind.m3u8") 
```

## Tiện ích mở rộng trình duyệt
- Chạy máy chủ với chương trình phụ trợ mong muốn của bạn như được hiển thị [tại đây](https://github.com/collabora/WhisperLive?tab=readme-ov-file#running-the-server).
- Phiên âm âm thanh trực tiếp từ trình duyệt của bạn bằng tiện ích mở rộng Chrome hoặc Firefox của chúng tôi. Tham khảo [Phiên âm-Chrome âm thanh](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Chrome#readme) và [Phiên âm âm thanh-Firefox](https://github.com/collabora/whisper-live/tree/main/Audio-Transcription-Firefox#readme) để biết hướng dẫn thiết lập.

## Máy chủ Whisper Live trong Docker
- GPU
  - Faster-Whisper
  ```bash
  docker run -it --gpus all -p 9090:9090 ghcr.io/collabora/whisperlive-gpu:latest
  ```

- TenorRT. Theo dõi [TensorRT_whisper readme](https://github.com/collabora/WhisperLive/blob/main/TensorRT_whisper.md) để thiết lập docker và sử dụng chương trình phụ trợ TensorRT. Chúng tôi cung cấp hình ảnh docker dựng sẵn có TensorRT-LLM được tạo sẵn và sẵn sàng sử dụng.

- CPU
```bash
docker run -it -p 9090:9090 ghcr.io/collabora/whisperlive-cpu:latest
```
**Lưu ý**: Theo mặc định, chúng tôi sử dụng kích thước mô hình "nhỏ". Để tạo hình ảnh docker cho kích thước mô hình khác, hãy thay đổi kích thước trong server.py rồi tạo hình ảnh docker.

## Công việc tương lai
- [ ] Thêm bản dịch sang các ngôn ngữ khác ở đầu bản phiên âm.
- [x] Phần phụ trợ TensorRT cho Whisper.
