#matplotlib inline

import numpy as np
import math
import matplotlib.pyplot as plt
import onnxruntime as rt
import cv2
import json

#labels = json.load(open("car.txt", "r"))

def loadData(filename):
    # dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():  # 逐行读取
        line = line.strip('\n')
        lineArr = line.strip()  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
        labelMat.append(lineArr)
    return labelMat

def img_stats(a, name={}):
    return {
        "name": name,
        "size": a.shape,
        "mean": "{:.2f}".format(a.mean()),
        "std": "{:.2f}".format(a.std()),
        "max": a.max(),
        "min": a.min(),
        "median": "{:.2f}".format(np.median(a)),
    }


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img


# read the image
fname = "./data/car1.jpg"
#fname = "grizzly.jpg"
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = np.transpose(img, (2, 0, 1))
# pre-process the image like mobilenet and resize it to 300x300
img = pre_process_edgetpu(img, (224, 224, 3))
plt.axis('off')
plt.imshow(img)
plt.show()

# create a batch of 1 (that batch size is buned into the saved_model)
img = np.transpose(img, (2, 0, 1))
img_batch = np.expand_dims(img, axis=0)

# load the model
sess = rt.InferenceSession("model_all.onnx")
##输入输出层名字(固定写法)
onnx_input_name = sess.get_inputs()[0].name
print(onnx_input_name)
##输出层名字,可能有多个
onnx_outputs_names = sess.get_outputs()
output_names = []
for o in onnx_outputs_names:
    output_names.append(o.name)
labels=loadData("./car.txt")

# run inference and print results
results = sess.run(output_names, input_feed={onnx_input_name: img_batch})[0]
result = reversed(results[0].argsort()[-5:])
for r in result:
    print(r, labels[r], results[0][r])


