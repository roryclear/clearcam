from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.helpers import fetch
class MaxPool2d:
    def __init__(self, kernel_size=2, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x): return x.pad((self.padding, self.padding, self.padding, self.padding)).max_pool2d(kernel_size=self.kernel_size, stride=self.stride)

class Seq():
    def __init__(self, size=0):
        super().__init__()
        self.list = [None] * size
    def __len__(self): return len(self.list)
    def __setitem__(self, key, value): self.list[key] = value
    def __getitem__(self, idx): return self.list[idx]
    def __delitem__(self, idx): del self.list[idx]
    def __call__(self, x):
        for y in self.list: x = y(x)
        return x

class BasicBlockIR_tiny():
    def __init__(self, in_channel, depth, stride):
        self.in_channel = in_channel
        self.depth = depth
        self.stride = stride

        self.res_layer0 = nn.BatchNorm2d(self.in_channel)
        self.conv_layer0 = nn.Conv2d(self.in_channel, self.depth, (3, 3), (1, 1), 1, bias=False)
        self.res_layer1 = nn.BatchNorm2d(self.depth)
        self.prelu_weight = Tensor.empty(self.depth)
        self.conv_layer1 = nn.Conv2d(self.depth, self.depth, (3, 3), self.stride, 1, bias=False)
        self.res_layer2 = nn.BatchNorm2d(self.depth)

        if self.depth == self.in_channel:
            self.shortcut_layer = MaxPool2d(1, self.stride)
        else:
            self.shortcut_layer0 = nn.Conv2d(self.in_channel, self.depth, (1, 1), self.stride, bias=False)
            self.shortcut_layer1 = nn.BatchNorm2d(self.depth)



    def __call__(self, x):
        if self.depth == self.in_channel:
            shortcut = self.shortcut_layer(x)
        else:
            shortcut = self.shortcut_layer0(x)
            shortcut = self.shortcut_layer1(shortcut)
        x = self.res_layer0(x)
        x = self.conv_layer0(x)
        x = self.res_layer1(x)
        x = Tensor.where(x > 0, x, self.prelu_weight.view(1, -1, 1, 1) * x)
        x = self.conv_layer1(x)
        x = self.res_layer2(x)
        return x + shortcut



sizes = [[64, 64, 2], [64, 64, 1], [64, 64, 1], [64, 128, 2], [128, 128, 1], [128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 256, 1], [256, 512, 2], [512, 512, 1], [512, 512, 1]]

class ADAFACE():
    def __init__(self):

        self.linear = nn.Linear(512 * 7 * 7, 512)
        self.bn = nn.BatchNorm2d(512)

        self.bn2 = nn.BatchNorm(512, affine=False)
        self.prelu_weight = Tensor.empty(64)
        self.conv0 = nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)

        self.body = Seq(size=24)
        for i in range(len(self.body)): self.body[i] = BasicBlockIR_tiny(sizes[i][0], sizes[i][1], sizes[i][2])
        state_dict = safe_load(fetch("https://huggingface.co/roryclear/AdaFace/resolve/main/adaface_ir50_ms1mv2.safetensors"))
        load_state_dict(self, state_dict)


    def __call__(self, x):
        x = ((x[:,:,::-1] / 255.) - 0.5) / 0.5
        x = x.permute(2,0,1).unsqueeze(0)

        x = self.conv0(x)
        x = self.bn0(x)
        x = Tensor.where(x > 0, x, self.prelu_weight.view(1, -1, 1, 1) * x)

        for module in self.body: x = module(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn2(x)
        norm = Tensor.sqrt(Tensor.sum(x * x, keepdim=True))
        output = x / norm
        return output, norm