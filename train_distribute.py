import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import Dataset
from model.MGTNet import Interactive
import torchvision.utils as vutils
import os
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import IOU
import torch.distributed as dist
import random
from torchvision import transforms
import transform
import datetime
from options import config

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_distribute.py


EPSILON = 1e-8
p = OrderedDict()

p['lr_bone'] = 1e-3  # Learning rate
p['lr_branch'] = 5e-3
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [10, 20]
showEvery = 50
tmp_path = 'tmp_out'
save_fold = 'trans_qv'


def adjust_learning_rate(optimizer, decay_count, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(1e-5, param_group['lr'] * pow(decay_rate, decay_count))
        print(param_group['lr'])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    set_seed(1024)
    # ------- 1. configure environment --------
    args = config
    print("local_rank", args.local_rank)
    world_size = int(os.environ['WORLD_SIZE'])
    print("world size", world_size)
    dist.init_process_group(backend='nccl')
    # ------- 2. set the directory of training dataset --------
    model_dir = "./trans_qv/"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    epoch_num = 15
    batch_size_train = 2

    composed_transforms_ts = transforms.Compose([
        transform.FixedResize(size=(448, 448)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])

    dataset_train = Dataset(datasets=['DUTS-TR', 'FBMS', 'DAVIS', 'FBMS', 'FBMS', 'DAVSOD', 'FBMS', ],
                            transform=composed_transforms_ts, mode='train')
    datasampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=dist.get_world_size(),
                                                                  rank=args.local_rank, shuffle=True)

    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, sampler=datasampler,
                                             num_workers=8)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))

    # ------- 3. define model --------
    spatial_ckpt = None
    temporal_ckpt = None
    torch.cuda.set_device(args.local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Interactive(spatial_ckpt, temporal_ckpt, pretrained=True))
    net = torch.nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank],
                                                    find_unused_parameters=True)
    pretrained_net = []
    transformer = []
    for name, param in net.named_parameters():
        if "spatial_net" in name or "temporal_net" in name:
            param.requires_grad = True
            pretrained_net.append(param)
        else:
            transformer.append(param)
    param_group = [{"params": pretrained_net, "lr": p['lr_bone']}, {"params": transformer, 'lr': p['lr_branch']}]
    optimizer = optim.SGD(param_group, lr=p['lr_bone'], momentum=0.9, weight_decay=0.0005)
    re_load = False
    model_name = "epoch_20_bone.pth"
    if re_load:
        model_CKPT = torch.load(model_dir + model_name, map_location='cpu')
        net.load_state_dict(model_CKPT)
        print("Successfully load: {}".format(model_name))

    optimizer.zero_grad()
    iter_num = len(dataloader)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    if not os.path.exists(save_fold):
        os.mkdir(save_fold)

    for epoch in range(1, epoch_num + 1):
        optimizer.param_groups[0]['lr'] = p['lr_bone']
        optimizer.param_groups[1]['lr'] = p['lr_branch']
        running_loss = 0.0
        running_final_loss = 0.0
        running_spatial_loss = 0.0
        datasampler.set_epoch(epoch)
        net.train()

        if epoch > 10:
            adjust_learning_rate(optimizer, (epoch - 10))

        for i, data_batch in enumerate(dataloader):

            image, label, flow = data_batch['image'], data_batch['label'], data_batch['flow']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue

            image, label, flow = image.cuda(args.local_rank), label.cuda(args.local_rank), flow.cuda(args.local_rank)

            out1r, out2r, out3r, out4r, out5r, course_img, course_flo = net(image, flow)

            loss1r = structure_loss(out1r, label)
            loss2r = structure_loss(out2r, label)
            loss3r = structure_loss(out3r, label)
            loss4r = structure_loss(out4r, label)
            loss5r = structure_loss(out5r, label)

            img_loss = structure_loss(course_img, label)
            flo_loss = structure_loss(course_flo, label)

            loss = loss1r + loss2r + loss3r + loss4r + loss5r + img_loss + flo_loss

            running_loss += loss.item()
            running_final_loss += loss1r.item()
            running_spatial_loss += img_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % showEvery == 0:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss_final : %10.4f  ||  pre_loss : %10.4f || '
                    'sum : %10.4f' % (
                        datetime.datetime.now(), epoch, epoch_num, i, iter_num,
                        running_final_loss / (i + 1), running_spatial_loss / (i + 1), running_loss / (i + 1)))

            if i % 50 == 0:
                vutils.save_image(torch.sigmoid(out1r.data), tmp_path + '/iter%d-sal-0.jpg' % i,
                                  normalize=True, padding=0)
                vutils.save_image(image.data, tmp_path + '/iter%d-sal-data.jpg' % i, padding=0)
                vutils.save_image(label.data, tmp_path + '/iter%d-sal-target.jpg' % i, padding=0)

        if epoch % 5 == 0 and args.local_rank == 0:
            torch.save(net.state_dict(),
                       '%s/epoch_%d_bone.pth' % (save_fold, epoch))
