import time
import cv2
import lite_inference
import edge_inference

# const
video_file = './demo/How To Control Caterpillars On Plants Easy (IN HINDI) Caterpillars Eating Leaves.mp4'
# model_file = './model_lite/model_full_integer_ResNet50_20230205-10:52:35.tflite'
model_file = './model_edge_tpu/model_full_integer_ResNet50_20230205-10:52:35_edgetpu.tflite'
rate_ratio = 10
# run
if __name__ == '__main__':
    # load video file
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)/rate_ratio
    print(fps)
    interpreter = edge_inference.load_interpreter(model_file)

    counter = 0
    while True:
        # cut frames & show frame
        ret, frame = video.read()
        counter += 1
        inference_time = 0
        cv2.imshow("Video", frame)
        if counter % rate_ratio == 0:
            start = time.perf_counter()
            # todo: pest classification
            predict = lite_inference.do_inference(interpreter, frame)
            inference_time = time.perf_counter() - start
            print(predict, inference_time)
        # todo: show titles & position & temperature & light info on screen
        # cv2.imshow("Heatmap", image_heat)
        gap = min(abs(fps / 1000 - inference_time), 0)
        time.sleep(gap)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
