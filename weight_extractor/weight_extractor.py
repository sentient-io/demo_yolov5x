import torch
import numpy as np

model_path = "yolov5x.pt"  # Weight path to convert

model = torch.load(model_path)["model"]

# yolov5, (0 -> off, 1 -> on)
if 1:
    weight_path = "../mgmt/weights/yolov5x.weights" # Soynet weight file to create
    with open(weight_path, 'wb') as f:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        anchors = 3
        anchor_layer_idx = 738

        index = [0, 18, 30, 78, 18, 30,
                 78, 90, 102, 198, 90, 102,
                 198, 210, 222, 366, 210, 222,
                 366, 378, 390, 438, 378, 390,
                 438, 462, 474, 522, 462, 474, #9
                 522, 534, 546, 594, 534, 546,
                 594, 606, 618, 666, 606, 618,
                 666, 678, 690, 738, 678, 690,
                 739, 745,
                 738, 739   # anchor
                 ]

        for i_idx in range(int(len(index) / 2)):
            for idx in range(index[i_idx * 2], index[i_idx * 2 + 1]):  #
                key, w = weight_list[idx]
                if (idx == anchor_layer_idx): # anchor_grid
                    for i in range(anchors):
                        w_ = w[i] * (8 * (2 ** i))  # stride : 8, 16, 32, 64
                        w_ = w_.cpu().data.numpy().astype(np.float32)
                        w_.tofile(f)
                    print(0, idx, key, w.shape)
                    continue

                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy().astype(np.float32)
                else:
                    w = w.cpu().data.numpy().astype(np.float32)
                w.tofile(f)
                print(0, idx, key, w.shape)
    f.close()