# Mô hình phiên dịch thời gian thực Whisper

!Thử nghiệm](demo.gif)

Đây là bản demo chuyển lời nói thành văn bản theo thời gian thực với mô hình Whisper của OpenAI. Nó hoạt động bằng cách liên tục ghi lại âm thanh trong một luồng và ghép các byte thô qua nhiều bản ghi.

Cài đặt những phụ thuộc sau để khởi động chương trình
```
pip install -r requirements.txt
```
trong môi trường của bạn khởi tạo

Whisper cũng yêu cầu cài đặt công cụ dòng lệnh [`ffmpeg`](https://ffmpeg.org/) trên hệ thống của bạn, công cụ này có sẵn ở hầu hết các trình quản lý gói:

```
# trên Ubuntu hoặc Debian
sudo apt update && sudo apt install ffmpeg

# trên Arch Linux
sudo pacman -S ffmpeg

# trên MacOS sử dụng Homebrew (https://brew.sh/)
brew install ffmpeg

# trên Windows sử dụng Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# trên Windows sử dụng Scoop (https://scoop.sh/)
scoop install ffmpeg
```

Để biết thêm thông tin về Whisper, vui lòng xem https://github.com/openai/whisper

Mã trong kho lưu trữ này là miền công cộng.