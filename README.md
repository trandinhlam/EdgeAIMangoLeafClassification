# Project code inference bài toán phân loại bệnh trên lá xoài bằng thiết bị biên (edge-device)

+ Dataset sử dụng: Mango Pest Classification

+ Thiết bị biên sử dụng (SBC): Asus Tinker Edge T

+ Code gốc tại đây: https://colab.research.google.com/drive/1FtpOmoJypyHSRQ9Ss0wGJtn-XFTqnyOT#scrollTo=Z3nKpIIiGEFJ

+ Model sử dụng là model dựa trên backbone là EfficientNetB2 hoặc Resnet50. Link model đã train tại đây:

https://drive.google.com/file/d/1-omJp4YaAL4cczyOIiPzPVVtV2Qxs-UF/view?usp=share_link

# Giải thích các file: 

## convert_to_lite.py:

Code này để tải nạp một model cụ thể kèm theo trọng số đã được train của nó. Sau đó convert bằng Tensorflow Lite có qua
bước quantization để cho ra model .tflite (uint8)
để model được gọn nhẹ hơn.

Tiếp đến, để chạy được model lite trên thiết bị edge TPU thì ta cần dùng edgetpu_compiler để convert sang _
edge_tpu.tflite.

## lite_inference.py:

Code file này dùng để chạy inference cho model tflite trên toàn bộ tập test

## edge_inference.py:

Code file này dùng để chạy inference cho model edge tflite trên toàn bộ tập test

## demo.py:



## logs:

+ 24/12/2022: Chạy bằng CPU của edge device với Model train từ EfficientNetB2: 8s 2s/step. Kích thước model khoảng  9.2M tham số.

+ 03/02/2023: Chạy model Resnet50 sau khi convert về EdgeTPU với tốc độ khoảng 20FPS cho một model 25M tham số. 






