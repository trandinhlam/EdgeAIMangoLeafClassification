import time
import cv2
import lite_inference
# import edge_inference

# const
# video_file = './demo/How To Control Caterpillars On Plants Easy (IN HINDI) Caterpillars Eating Leaves.mp4'
import utils

video_file = './demo/Mango Leaf Cutting Weevil Control (IN HINDI) Mango Leaf Cutting Insects.mp4'
# model_file = './model_lite/model_full_integer_ResNet50_20230205-10:52:35.tflite'
# model_file = './model_edge_tpu/model_full_integer_ResNet50_20230205-10:52:35_edgetpu.tflite'
model_file = './model_lite/model_full_integer_ResNet8_20230218-11:26:04.tflite'
# model_file = './model_edge_tpu/model_full_integer_ResNet8_20230218-11:26:04_edgetpu.tflite'
# rate_ratio = 10
rate_ratio = 2
# run
if __name__ == '__main__':
    # load video file
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS) / rate_ratio
    print(fps)
    interpreter = lite_inference.load_interpreter(model_file)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('demo/demo.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    counter = 0
    label = None
    while True:
        # cut frames & show frame
        ret, frame = video.read()
        counter += 1
        inference_time = 0
        if counter % rate_ratio == 0:
            start = time.perf_counter()
            predict, label = lite_inference.do_inference(interpreter, frame)
            inference_time = time.perf_counter() - start
            print(predict, inference_time)
        frame = cv2.putText(img=frame,
                            text=f'(fake) coordinate:  (10.781648583014876, 106.65536393209882) ',
                            org=(0, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=0.75,
                            color=(0, 255, 0), thickness=2)
        frame = cv2.putText(img=frame,
                            text=f'(fake)temperature sensor: 32\u00B0C',
                            org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=0.75,
                            color=(0, 0, 255), thickness=2)
        if label is not None:
            frame = cv2.putText(img=frame,
                                text=f'label: {label}',
                                org=(0, 75), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=0.75,
                                color=(255, 0, 0), thickness=2)
        cv2.imshow("Video", frame)
        result.write(frame)
        gap = max(fps / 1000 - inference_time, 0)
        time.sleep(gap)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    result.release()
    cv2.destroyAllWindows()
