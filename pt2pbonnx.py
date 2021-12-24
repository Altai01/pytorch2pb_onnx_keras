import torch
import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
import keras
import tensorflow as tf


def pth_to_onnx(input_path, output_path):
    '''
    1)声明：使用本函数之前，必须保证你手上已经有了.pth模型文件.
    2)功能：本函数功能四将pytorch训练得到的.pth文件转化为onnx文件。
    '''
    torch_model = torch.load(input_path)  # pytorch模型加载,此处加载的模型包含图和参数
    # torch_model = selfmodel()  # 若只保存参数，selfmodel参考上述loadmodel进行改写
    torch_model.eval()
    x = torch.randn(1, 3, 224, 224)  # 输入一张28*28的灰度图像并生成张量
    export_onnx_file = output_path  # 输出.onnx文件的文件路径及文件名
    torch_model.set_swish(memory_efficient=False)
    torch.onnx.export(torch_model,
                      x,
                      export_onnx_file,
                      opset_version=9,  # 操作的版本，稳定操作集为9
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}}
                      )
    # onnx_model = onnx.load('model_all.onnx')    #加载.onnx文件
    # onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))       #打印.onnx文件信息


def onnx_to_pb(output_path):
    '''
    将.onnx模型保存为.pb文件模型
    '''
    model = onnx.load(output_path)  # 加载.onnx模型文件
    tf_rep = prepare(model)
    tf_rep.export_graph('model_all')  # 保存最终的.pb文件//graphmodel


def onnx_to_h5(output_path):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model('kerasModel.h5')
    model.summary()
    print(model)
def onnx_pre(output_path):
    onnx_model = onnx.load(output_path)    #加载.onnx文件
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))       #打印.onnx文件信息

if __name__ == '__main__':
    input_path = "efficient.pt"  # 输入需要转换的.pth模型路径及文件名
    output_path = "model_all.onnx"  # 转换为.onnx后文件的保存位置及文件名
    #pth_to_onnx(input_path, output_path)  # 执行pth转onnx函数，具体转换参数去该函数里面修改
    onnx_pre(output_path)   #【可选项】若有需要，可以使用onnxruntime进行部署测试，看所转换模型是否可用，其中，output_path指加载进去的onnx格式模型所在路径及文件名
    #onnx_to_pb(output_path)   #将onnx模型转换为pb模型
    # onnx_to_h5(output_path )   #将onnx模型转换为h5模型
