# Nhập các thư viện cần thiết
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# Hàm xửlý chính
def main():
    parser = argparse.ArgumentParser()
    # Khai báo các giá trị biến điều khiển cho chương trình
    parser.add_argument("--model", default="base", help="Mô hình sử dụng",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Không sử dụng mô hình ngôn ngữ anh.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Mức tần số để mic phát hiện.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="Thời gian ghi âm thực tế tính bằng giây.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="Có bao nhiêu khoảng trống giữa các bản ghi trước khi chúng tôi "
                             "coi đó là một dòng mới trong phiên âm.", type=float)
    # Dành cho linux
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Tên micrô mặc định cho SpeechRecognition. "
                                 "Chạy cái này với 'list' để xem các Micrô có sẵn.", type=str)
    args = parser.parse_args()

    # Lần cuối cùng một bản ghi được lấy ra từ hàng đợi.
    phrase_time = None
    # Hàng đợi an toàn theo luồng để truyền dữ liệu từ cuộc gọi lại ghi theo luồng.
    data_queue = Queue()
    # Chúng tôi sử dụng SpeechRecognizer để ghi lại âm thanh của mình 
    # vì nó có một tính năng hay là có thể phát hiện khi lời nói kết thúc.
    recorder = sr.Recognizer()
    # khởi tạo mức tần số âm thanh nhận dạng đó là giọng nói
    recorder.energy_threshold = args.energy_threshold
    # Chắc chắn phải làm điều này, việc bù năng lượng động sẽ giảm đáng kể ngưỡng năng lượng đến mức 
    # mà SpeechRecognizer không bao giờ ngừng ghi.
    recorder.dynamic_energy_threshold = False

    # Quan trọng đối với người dùng linux.
    # Ngăn chặn tình trạng treo và treo ứng dụng vĩnh viễn do sử dụng sai Micrô!
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Các thiết bị micrô có sẵn là: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Micrô có tên \"{name}\" đã được tìm thấy")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Mô hình tải / tải xuống
    model = args.model
    # tại đầy chúng ta sẽ tải model từ model chỉ định do người 
    # dùng yêu cầu
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    # load mô hình
    audio_model = whisper.load_model(model)
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    # tạo một mảng để lưu trữ các kết quả vào 
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Chức năng gọi lại theo chuỗi để nhận dữ liệu âm thanh khi quá trình ghi kết thúc.
        âm thanh: Một AudioData chứa các byte đã ghi.
        """
        # Lấy các byte thô và đẩy nó vào hàng đợi an toàn của luồng.
        data = audio.get_raw_data()
        data_queue.put(data)
        # print("Dữ liệu byte thô: ", data)

    # Tạo một chuỗi nền sẽ truyền cho chúng tôi các byte âm thanh thô.
    # Chúng tôi có thể thực hiện việc này một cách thủ công nhưng SpeechRecognizer cung cấp một công cụ trợ giúp tuyệt vời.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    # Gợi ý cho người dùng rằng chúng tôi đã sẵn sàng hoạt động.
    print("Đã tải xong mô hình.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Kéo âm thanh đã ghi thô từ hàng đợi.
            if not data_queue.empty():
                phrase_complete = False
                # Nếu đã đủ thời gian giữa các bản ghi, hãy coi như cụm từ đã hoàn thành.
                # Xóa bộ đệm âm thanh đang hoạt động hiện tại để bắt đầu lại với dữ liệu mới.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # Đây là lần cuối cùng chúng tôi nhận được dữ liệu âm thanh mới từ hàng đợi.
                phrase_time = now
                
                # Kết hợp dữ liệu âm thanh từ hàng đợi
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Chuyển đổi bộ đệm trong ram thành thứ mà mô hình có thể sử dụng trực tiếp mà không cần tệp tạm thời.
                # Chuyển đổi dữ liệu từ số nguyên rộng 16 bit sang dấu phẩy động có chiều rộng 32 bit.
                # Kẹp tần số luồng âm thanh ở mức mặc định tương thích với bước sóng PCM ở mức tối đa 32768hz.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Đọc phiên âm
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Nếu chúng tôi phát hiện thấy khoảng dừng giữa các bản ghi, hãy thêm mục mới vào bản ghi của chúng tôi.
                 # Nếu không thì hãy chỉnh sửa cái hiện có.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Xóa bảng điều khiển để in lại bản ghi đã cập nhật.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Xóa thiết bị xuất chuẩn.
                print('', end='', flush=True)
            else:
                # Vòng lặp vô hạn có hại cho bộ xử lý, phải ngủ.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nPhiên mã:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
