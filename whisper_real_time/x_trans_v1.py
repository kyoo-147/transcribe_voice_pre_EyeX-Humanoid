# import os
import numpy as np
import speech_recognition as sr
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep

import whisperx

def main():
    # Deffine varriable
    # Khởi tạo biến
    RECORD_TIMEOUT = 2
    PHRASE_TIMEOUT = 3
    # requiments gpu tesla
    DEVICE_X = "cpu"
    # compute type
    COMPUTE_TYPE = "int8"
    BATCH_SIZE = 16
    MODEL_DIR = "."
    MIC_NAME = "pulse"
    
    phrase_time = None
    
    model_x = whisperx.load_model("small", DEVICE_X, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
    # Hàng đợi an toàn theo luồng để truyền dữ liệu từ cuộc gọi lại ghi theo luồng.
    data_queue = Queue()
    # Chúng tôi sử dụng SpeechRecognizer để ghi lại âm thanh của mình vì nó có một 
    # tính năng hay là có thể phát hiện khi lời nói kết thúc.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Chắc chắn phải làm điều này, việc bù năng lượng động sẽ giảm đáng kể ngưỡng năng lượng đến mức mà 
    # SpeechRecognizer không bao giờ ngừng ghi.
    recorder.dynamic_energy_threshold = False
    
    for index, name in enumerate(sr.Microphone.list_microphone_names()):        
        if MIC_NAME in name:
            print("Mã chỉ mục của thiết bị Microphone: ", index)
            print("Tên thiết bị tại vị trí index: ", MIC_NAME)
            source = sr.Microphone(sample_rate=16000, device_index=index)
            break
    
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Chức năng gọi lại theo chuỗi để nhận dữ liệu âm thanh khi quá trình ghi kết thúc.
        audio: Một AudioData chứa các byte đã ghi.
        """
        # Lấy các byte thô và đẩy nó vào hàng đợi an toàn của luồng.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Tạo một chuỗi nền sẽ truyền cho chúng tôi các byte âm thanh thô.
    # Chúng tôi có thể thực hiện việc này một cách thủ công nhưng 
    # SpeechRecognizer cung cấp một công cụ trợ giúp tuyệt vời.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=RECORD_TIMEOUT )
    print("Mô hình đã được tải hoàn tất.\n")
    
    while True:
        try:
            now = datetime.utcnow()
            # Kéo âm thanh đã ghi thô từ hàng đợi.
            if not data_queue.empty():
                phrase_complete = False
                # Nếu đã đủ thời gian giữa các bản ghi, hãy coi như cụm từ đã hoàn thành.
                # Xóa bộ đệm âm thanh đang hoạt động hiện tại để bắt đầu lại với dữ liệu mới.
                if phrase_time and now - phrase_time > timedelta(seconds=PHRASE_TIMEOUT):
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
                result_x = model_x.transcribe(audio_np, batch_size=BATCH_SIZE)
                # print(result_x["segments"]) 
                if result_x["segments"]:
                    result_text = result_x['segments'][0]['text']
                    # print(result_text.strip())
                    if phrase_complete:
                        transcription.append(result_text)
                    else:
                        # transcription[-1] = transcription[-1] + " " + result_text
                        transcription[-1] = result_text
                else:
                    # result_text = result_x['segments'][0]['text']
                    pass
                
                # Nếu chúng tôi phát hiện thấy khoảng dừng giữa các bản ghi, hãy thêm mục mới vào bản ghi của chúng tôi.
                # Nếu không thì hãy chỉnh sửa cái hiện có.
                
                    
                for line in transcription:
                    print(line)
                    
                print('', end='', flush=True)
            else:
                sleep(0.20)
        
        except KeyboardInterrupt:
            break
        
    print("\n\nBản phiên dịch dịch: ")
    for line in transcription:
        print(line)

# i think we always need it
if __name__ == "__main__":
    main()