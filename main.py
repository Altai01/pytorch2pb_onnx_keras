# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
from pytorch2keras.converter import pytorch_to_keras
import torch
import torch.nn as nn
from keras import backend as K
import tensorflow as tf
import numpy as np
from torch.autograd import Variable
import onnx


# import torch
# from pytorch2keras.converter import pytorch_to_keras
#
# input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
# input_var = Variable(torch.FloatTensor(input_np))
# net =torch.load('./efficient.pt') # your model
# #x = torch.randn(1, 3, 224, 224, requires_grad=False) # dummy input
# k_model = pytorch_to_keras(net, input_var, [(3, None, None,)], verbose=True, name_policy='short')
# print(k_model.summry())
# k_model.save('keras.h5')
#
# input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
# input_var = Variable(torch.FloatTensor(input_np))
#
# def import_model():
#     # Model
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
#
#     # Images
#     data = 'street.jpg'  # or file, Path, PIL, OpenCV, numpy, list
#
#     # Inference
#     results = model(data)
#
#     # Results
#     results.print()  # or .sho
#     print("123")
#     return model
# def turn2keras(model):
#     return pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True)
#
# # Function below copied from here:
# # https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     """
#     Freezes the state of a session into a pruned computation graph.
#
#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.
#     """
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = \
#             list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
#
# def frozen_graph(k_model):
#     frozen_graph = freeze_session(K.get_session(),
#                               output_names=[k_model.out.name for out in k_model.outputs])
#     tf.train.write_graph(frozen_graph, ".", "my_model.pb", as_text=False)
#     print([i for i in k_model.outputs])
# #
# # class TestConv2d(nn.Module):
# #     """
# #     Module for Conv2d testing
# #     """
# #
# #     def __init__(self, inp=3, out=16, kernel_size=3):
# #         super(TestConv2d, self).__init__()
# #         self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=True)
# #
# #     def forward(self, x):
# #         x = self.conv2d(x)
# #         return x
# #
# # net=torch.load("efficient.pt")
# # #print(net)
# # net.eval()
# # model = net
# #
# # from pytorch2keras import pytorch_to_keras
# # # we should specify shape of the input tensor
# # k_model = pytorch_to_keras(model, input_var, [(3, None, None,)], verbose=True)
# # print('123')
# # print(k_model)
# # print('234')
# # k_model.save('k_model1.h5')
# # ##load weights here
# #model.load_state_dict(torch.load(path_to_weights.pth))
#
# #Press the green button in the gutter to run the script.
# #
# if __name__ == '__main__':
#     #mode=import_model()
#     net = torch.load("efficient.pt")
#     # #print(net)
#     net.eval()
#     model = net
#     k_model = turn2keras(model)
#     print(k_model)
#     #frozen_graph(k_model)
#     #k_model.save('keras2S.h5')
#     #print(torch.__version__)
#     print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/





