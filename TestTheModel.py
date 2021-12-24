import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import tensorflow.keras.models as models
from tensorflow.python.platform import gfile


####tensorlfow2.0与1.0版本不兼容，在程序开始部分添加以下代码
tf.compat.v1.disable_eager_execution()

img="./data/car1.jpg"
image = Image.open(img)
model_pb_ath='./model_all'

def read_data():
    img = cv2.imread("/Users/yiche/Documents/onnx/beiqi2.jpeg")
    img = cv2.resize(img, (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    #img = img.view(1, 1280, 224, 224)
    input_blob = np.expand_dims(img, axis=0).astype(np.float32)  # NCHW
    return input_blob

def loadData(filename):
    # dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():  # 逐行读取
        line = line.strip('\n')
        lineArr = line.strip()  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
        labelMat.append(lineArr)
    return labelMat

def test_pt_model(img):
    model=torch.load('efficient.pt')
    model.eval()
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(img).unsqueeze(0)
    labelMat=loadData("./car.txt")
    with torch.no_grad():
        outputs = model(img)
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        if idx < len(labelMat):
            print('{labe:} ({p:.2f}%)'.format(labe=labelMat[idx], p=prob * 100))

def test_onnx_model(img):
    onnx_model = onnx.load('model_all.onnx')
    ort_session = ort.InferenceSession('model_all.onnx')
    ##输入输出层名字(固定写法)
    onnx_input_name = ort_session.get_inputs()[0].name
    ##输出层名字,可能有多个
    onnx_outputs_names = ort_session.get_outputs()

    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(img).unsqueeze(0)
    output_names = []
    for o in onnx_outputs_names:
        output_names.append(o.name)
    onnx_result = ort_session.run(output_names, input_feed={onnx_input_name: img})[0]
    labelMat = loadData("./data/car.txt")
    for idx in torch.topk(onnx_result, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(onnx_result, dim=1)[0, idx].item()
        if idx < len(labelMat):
            print('{labe:} ({p:.2f}%)'.format(labe=labelMat[idx], p=prob * 100))
    # # ort_session.get_outputs()[0].name()
    # onnx_result = torch.Tensor(onnx_result)  # 这里只有一个输出
    # arg = onnx_result.argmax()
    # labelMat = loadData("./data/car.txt")
    # print('{labe:} ({p:.2f}%)'.format(labe=labelMat[arg], p=prob * 100))

def test_pb_model(img):
    model=tf.saved_model.load(model_pb_ath)
    print(model)
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(img).unsqueeze(0)
    labelMat = loadData("./data/car.txt")
    # with torch.no_grad():
    #     outputs = model(img)
    # for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    #     prob = torch.softmax(outputs, dim=1)[0, idx].item()
    #     if idx < len(labelMat):
    #         print('{labe:} ({p:.2f}%)'.format(labe=labelMat[idx], p=prob * 100))

if __name__ =='__main__':
    test_pt_model(image)
    #test_pb_model(image)
    #test_onnx_model(image)