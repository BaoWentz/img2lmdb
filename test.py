import torch
import torchvision.models as models
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片

from torchvision import transforms as transforms

from dataset import RawDataset, label_num

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 11
class VGGNet(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(VGGNet, self).__init__()
            net = models.vgg16(pretrained=False)
            net.classifier = nn.Sequential()
            self.features = net
            self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

model = VGGNet()
if torch.cuda.is_available():
    model.cuda()
pre=torch.load('.\saved_models\experiment01\iter_1000.pth', map_location=device)
model.load_state_dict(pre)

#text model-------------------------
class args(object):
    #必要的一些参数设置
    def __init__(self):
        self.rgb = True
        self.imgW = 128
        self.imgH = 128
        self.path = os.path.join(os.getcwd(), 'test_imgs')
        self.batch_size = 4

opt = args()
test_loader = RawDataset(opt.path, opt)
#length_of_data = len(test_loader)#图片数量
test_set=torch.utils.data.DataLoader(dataset=test_loader, batch_size=opt.batch_size, shuffle=False, pin_memory =True)

model.eval()
fig_i = 0
for batch_x, path_x in test_set:
    if len(batch_x) == 0:
        break
    fig_i += 1
    x_tensors = batch_x.to(device)
    out = model(x_tensors)
    pred = torch.max(out, 1)[1]
    pred = pred.tolist()#将tensor转为list
    
    label2num, num2label = label_num('all_labels.txt')
    label_list = [num2label[str(i)] for i in pred][::-1]#打印标签
    #print(label_list)
    
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.figure(fig_i, figsize=(10.24, 7.68))
    #plt.title('预测结果可视化')
    for j in range(len(batch_x)):
        image = mpimg.imread(path_x[j])
        plt.subplot(2, 2, j+1)#每行最多四张图片
        plt.imshow(image)
        plt.axis('off')
        plt.title(label_list.pop())
        
    plt.savefig("text_out{}.jpg".format(fig_i))
    plt.show()


sys.exit()
