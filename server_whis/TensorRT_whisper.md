# Whisper-TensorRT
Chúng tôi mới chỉ thử nghiệm phần phụ trợ TensorRT trong docker, vì vậy, chúng tôi khuyên dùng docker để thiết lập phần phụ trợ TensorRT mượt mà.Chúng tôi mới chỉ thử nghiệm phần phụ trợ TensorRT trong docker, vì vậy, chúng tôi khuyên dùng docker để thiết lập phần phụ trợ TensorRT mượt mà.

## Cài đặt
- Cài đặt [docker](https://docs.docker.com/engine/install/)
- Cài đặt [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Sao chép kho lưu trữ.
```bash
git clone https://github.com/collabora/WhisperLive.git
cd WhisperLive
```

- Kéo hình ảnh docker TensorRT-LLM mà chúng tôi đã tạo sẵn cho chương trình phụ trợ WhisperLive TensorRT.
```bash
docker pull ghcr.io/collabora/whisperbot-base:latest
```

- Tiếp theo chúng ta chạy docker image và mount WhisperLive repo vào thư mục `/home` của container.
```bash
docker run -it --gpus all --shm-size=8g \
       --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
       -p 9090:9090 -v /path/to/WhisperLive:/home/WhisperLive \
       ghcr.io/collabora/whisperbot-base:latest
```

- Hãy chắc chắn để kiểm tra cài đặt.
```bashvarious
# export ENV=${ENV:-/etc/shinit_v2} 
# source $ENV
python -c "import torch; import tensorrt; import tensorrt_llm"
```
**LƯU Ý**: Bỏ ghi chú và cập nhật đường dẫn thư viện nếu quá trình nhập không thành công.

## Công cụ Whisper TensorRT 
- Chúng tôi xây dựng công cụ TensorRT đa ngôn ngữ `small.en` và `small`. Tập lệnh ghi lại đường dẫn của thư mục bằng công cụ Whisper TensorRT. Chúng tôi cần model_path để chạy máy chủ.
```bash
# convert small.en
bash scripts/build_whisper_tensorrt.sh /root/TensorRT-LLM-examples small.en

# convert small multilingual model
bash scripts/build_whisper_tensorrt.sh /root/TensorRT-LLM-examples small
```

## Chạy máy chủ WhisperLive với phần cuối TensorRT
```bash
cd /home/WhisperLive

# Install requirements
apt update && bash scripts/setup.sh
pip install -r requirements/server.txt

# Required to create mel spectogram
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

# Run English only model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "path/to/whisper_trt/from/build/step"

# Run Multilingual model
python3 run_server.py --port 9090 \
                      --backend tensorrt \
                      --trt_model_path "path/to/whisper_trt/from/build/step" \
                      --trt_multilingual
```
