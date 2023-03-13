import argparse
import os
import cv2
import numpy as np
from dataset import Dataset
import torch
from model.MGTNet import Interactive
from torchvision import transforms
import transform
from torch.utils import data


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())  # 是否使用cuda

parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=0)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--test_dataset', type=list, default=['DAVIS', 'FBMS', 'SegTrack-V2', 'ViSal', 'DAVSOD', 'VOS'])
parser.add_argument('--testsavefold', type=str, default='')

config = parser.parse_args()

composed_transforms_ts = transforms.Compose([
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])

dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_ts, mode='test')

test_loader = data.DataLoader(dataset, batch_size=1, num_workers=config.num_thread, drop_last=True, shuffle=False)

net_bone = Interactive()
name = "MGT-Net"
if config.cuda:
    net_bone = net_bone.cuda()

assert (config.model_path != ''), ('Test mode, please import pretrained model path!')
assert (os.path.exists(config.model_path)), ('please import correct pretrained model path!')
print('load model……')
ckpt = torch.load(config.model_path)
model_dict = net_bone.state_dict()
pretrained_dict = {k[7:]: v for k, v in ckpt.items() if k[7:] in model_dict}
model_dict.update(pretrained_dict)
net_bone.load_state_dict(model_dict)
net_bone.eval()

if not os.path.exists(config.testsavefold):
    os.makedirs(config.testsavefold)

for i, data_batch in enumerate(test_loader):
    print("progress {}/{}\n".format(i + 1, len(test_loader)))
    image, flow, name, split, size = data_batch['image'], data_batch['flow'], data_batch['name'], data_batch[
        'split'], data_batch['size']
    dataset = data_batch['dataset']

    if config.cuda:
        image, flow = image.cuda(), flow.cuda()
    with torch.no_grad():

        pre = net_bone(image, flow)

        for i in range(config.test_batch_size):
            presavefold = os.path.join(config.testsavefold, dataset[i], split[i])
            if not os.path.exists(presavefold):
                os.makedirs(presavefold)
            pre1 = torch.nn.Sigmoid()(pre[0][i])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1) + 1e-8)
            pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
            pre1 = cv2.resize(pre1, (size[0][1], size[0][0]))
            cv2.imwrite(presavefold + '/' + name[i], pre1)

