import torch
import torchvision
import numpy as np

# 通过跟踪转换为Torch脚本
# # 你模型的一个实例.
# model = torchvision.models.resnet18()

# # 您通常会提供给模型的forward()方法的示例输入。
# example = torch.rand(1, 3, 224, 224)

# # 使用`torch.jit.trace `来通过跟踪生成`torch.jit.ScriptModule`
# traced_script_module = torch.jit.trace(model, example)

# output = traced_script_module(torch.ones(1, 3, 224, 224))
# print(output[0, :5])

# 通过注释转换为Torch脚本
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input, input2):
        if input.sum() > 0:
          print(self.weight.shape)
          print(input.shape)
          output = self.weight.mv(input)
        else:
          output = self.weight + input + input2
        return output

input = -torch.ones(10, 20)
input2 = -torch.ones(10, 20)
my_module = MyModule(10,20)
# sm_script = torch.jit.script(my_module)
sm_trace = torch.jit.trace(my_module, (input, input2))
# print('sm_script: \n', sm_script)
# print('sm_trace: \n', sm_trace)
# print(sm_trace.forward(input))
print(sm_trace.forward(input, input2))

# output = sm.forward(torch.ones(10,20))
# print(output)

# 将脚本序列化为文件
sm_trace.save("my_module_model.pt")

sm_load = torch.jit.load("my_module_model.pt")
print(sm_load.forward(input, input2))
