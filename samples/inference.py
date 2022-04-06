from SoyNet import *
import time, cv2
import numpy as np
import argparse
import glob
import os
from numpy import random

class_names = [  # 80
    #"N/A",  # BG
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

color_table = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]

def detect(opt):
    source,source_type, model_name, batch_size, engine_serialize, class_count, nms_count, region_count, device_id, model_height, model_width, = \
        opt.source, opt.source_type, opt.model_name, opt.batch_size, opt.engine_serialize, opt.class_count, opt.nms_count, opt.region_count, \
        opt.device_id, opt.model_h, opt.model_w

    # load files
    p = os.path.abspath(source)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files

    vcaps = []
    if source_type == 'cam':  # cam
        vcaps.append(cv2.VideoCapture(0))
    elif source_type == 'video' or source_type=="img" :  # video
        for idx in range(batch_size):
            vcaps.append(cv2.VideoCapture(files[idx]))
    else:
        print('Error, Not supported source_type [%s], should be [video or cam]' % source_type)
        exit()


    # Input Size
    input_height, input_width = (
    int(vcaps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vcaps[0].get(cv2.CAP_PROP_FRAME_WIDTH)))

    cfg_file = "../mgmt/configs/%s.cfg" % model_name
    engine_file = "../mgmt/engines/%s.bin" % model_name
    weight_file = "../mgmt/weights/%s.weights" % model_name
    log_file = "../mgmt/logs/soynet_%s.log" % model_name
    license_file = "../mgmt/configs/license_trial.key"

    extend_param = \
        "BATCH_SIZE=%d ENGINE_SERIALIZE=%d MODEL_NAME=%s CLASS_COUNT=%d NMS_COUNT=%d REGION_COUNT=%d CONF_THRES=%f IOU_THRES=%f LICENSE_FILE=%s DEVICE_ID=%d ENGINE_FILE=%s WEIGHT_FILE=%s LOG_FILE=%s INPUT_SIZE=%d,%d MODEL_SIZE=%d,%d" % (
            batch_size, engine_serialize, model_name, class_count, nms_count, region_count, opt.conf_thres, opt.iou_thres, license_file, device_id,
            engine_file, weight_file, log_file, input_height, input_width, model_height, model_width)

    # Initialization
    soynet_handle = initSoyNet(cfg_file, extend_param)
    
    # Warm-up
    inference(soynet_handle)
    
    process_count = 0
    total_time = 0

    fps = 0
    avg_fps = 0
    k = [0 for i in range(batch_size)]

    is_bbox = 1  # put rect on
    is_text = 1


    inputs = np.zeros((batch_size, input_height,input_width, 3), dtype=np.uint8)
    tt = nms_count * batch_size
    dt = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float), ("id", c_int),
                ("prob", c_float)])
    output = np.zeros(tt, dtype=dt)

    is_bbox_flag = 0
    is_text_flag = 0
    is_window_flag = 0
    is_fps_flag = 0
    is_objInfo_flag = 0

    is_fps = opt.is_fps
    is_objInfo = opt.is_objInfo

    while 1:
        for vidx in range(batch_size):
            success, frame = vcaps[vidx].read()
            if not success:
                is_window_flag += 1
            else:
                inputs[vidx] = cv2.resize(frame, (input_width, input_height))

        if is_window_flag % 2:
            break

        begin_time = time.time()
        feedData(soynet_handle, inputs)
        inference(soynet_handle)
        getOutput(soynet_handle, output)
        end_time = time.time()
        dur_time = end_time - begin_time


        # display
        for vidx in range(batch_size):
            if is_bbox == 1:
                for nidx in range(nms_count):
                    r = output[nms_count * vidx + nidx]
                    id, prob = r[4], r[5]
                    if prob <= opt.conf_thres:
                        break
                    y1, x1, y2, x2 = (int(r[1]), int(r[0]), int(r[3]),
                                      int(r[2]))  # typecast (float - > int)
                    if is_objInfo == 1:
                        print("%3d %3d x1: %d, y1: %d, x2: %d, y2: %d, obj_id : %2d, prob : %.5f, class_name : %s"
                              % (vidx, nidx, x1, y1, x2, y2, int(id), prob, class_names[int(id)]))

                    c_n_idx = int(id) % len(color_table)
                    cv2.rectangle(inputs[vidx], (x1, y1), (x2, y2), color_table[c_n_idx], 2)

                    if is_text == 1:
                        text = "%s %.5f" % (class_names[int(id)], prob)
                        cv2.putText(inputs[vidx], text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_table[c_n_idx])

            if is_fps == 1:
                fps = 1. / (dur_time + 1e-10)
                text = "fps=%.2f" % (fps)
                print(text)
                cv2.putText(inputs[vidx], text, (5, model_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            if is_window_flag % 2:
                continue
            else:
                cv2.imshow("yolor model_%s" % vidx, inputs[vidx])

            key = 1
            if source_type == "img":
                key = 0

            k[vidx] = cv2.waitKey(key)

            if k[vidx] in [27, ord('q'), ord('Q')]:
                is_window_flag += 1

            elif k[vidx] == ord(' '):
                cv2.waitKey(0)
            elif k[vidx] in [ord('b'), ord('B')]:
                is_bbox_flag += 1
                if is_bbox_flag % 2:
                    is_bbox = 1 - is_bbox
                else:
                    is_bbox = 1
            elif k[vidx] in [ord('t'), ord('T')]:
                is_text_flag += 1
                if is_text_flag % 2:
                    is_text = 1 - is_text
                else:
                    is_text = 1
            elif k[vidx] in [ord('f'), ord('F')]:
                is_fps_flag += 1
                if is_fps_flag % 2:
                    is_fps = 1 - is_fps
                else:
                    is_fps = 1
            elif k[vidx] in [ord('i'), ord('I')]:
                is_objInfo_flag += 1
                if is_objInfo_flag % 2:
                    is_objInfo = 1 - is_objInfo
                else:
                    is_objInfo = 1

        if is_window_flag % 2:
            break

    # finalize
    freeSoyNet(soynet_handle)
    for vidx in range(batch_size):
        vcaps[vidx].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolov5x', help='*.cfg path')
    parser.add_argument('--source', type=str, default="../data/zidane.jpg", help='source')  # file/folder
    parser.add_argument('--source_type', type=str, default='img', help='[img / video / cam ]')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--region_count', type=int, default=1000, help='Output value count to be used for NMS.')
    parser.add_argument('--nms_count', type=int, default=100, help='Maximum number of objects to be displayed on the screen <= The number of final objects defined by the model')
    parser.add_argument('--batch_size', type=int, default=1, help='SoyNet batch size')
    parser.add_argument('--engine_serialize', type=int, default=0, help='Whether the engine file is regenerated (use the previously generated engine file if 0 and regenerate the engine if 1 \
                                                                        (*Note* changes the parameter related to the result value)]')
    parser.add_argument('--device_id', type=int, default=0, help='using device id')
    parser.add_argument('--model_h', type=int, default=640, help='SoyNet model height ( model input height ) ')
    parser.add_argument('--model_w', type=int, default=640, help='SoyNet model width ( model input width )')
    parser.add_argument('--class_count', type=int, default=80, help='class count')

    parser.add_argument('--is_fps', type=int, default=0, help='Display fps')
    parser.add_argument('--is_objInfo', type=int, default=1, help='Display detected object info')
    opt = parser.parse_args()
    print(opt)

    detect(opt)

