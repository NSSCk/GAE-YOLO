import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

# CPU（OpenCV）
def preprocess(image):
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image / 255.0).astype(np.float32)
    return np.transpose(image, (2, 0, 1))  # HWC→CHW

# GPU/DLA
def inference(engine, input_data):
    # GPU
    cuda.memcpy_htod(inputs[0]["device"], input_data)
    # DLA
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    output_data = np.empty(outputs[0]["shape"], dtype=outputs[0]["dtype"])
    cuda.memcpy_dtoh(output_data, outputs[0]["device"])
    return output_data

# CPU（NMS）
def postprocess(output_data):
    boxes = output_data[..., :4]
    scores = output_data[..., 4]
    keep = nms(boxes, scores, iou_threshold=0.5)
    return boxes[keep]


image = cv2.imread("tomato.jpg")
input_data = preprocess(image)          # CPU
detections = inference(engine, input_data)  # GPU/DLA推
results = postprocess(detections)       # CPU