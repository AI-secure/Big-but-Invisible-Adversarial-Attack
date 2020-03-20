import os
from models import color_net
from dataloader import im_dataset
import torch
import torchvision.models as models
from torchvision.utils import save_image

import util
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./test_images', help='path to images to attack')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. Only support 1 for now')
parser.add_argument('--ab_max', type=float, default=110., help='maximimum ab value')
parser.add_argument('--ab_quant', type=float, default=10., help='quantization factor')
parser.add_argument('--l_norm', type=float, default=100., help='colorization normalization factor')
parser.add_argument('--l_cent', type=float, default=50., help='colorization centering factor')
parser.add_argument('--mask_cent', type=float, default=.5, help='mask centering factor')
parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
parser.add_argument('--target', type=int, default=444, help='target class')
parser.add_argument('--hint', type=int, default=50, help='number of hint to run with')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--targeted', type=int, default=1, help='targeted or untargeted attack')
parser.add_argument('--n_clusters', type=int, default=8, help='number of clusters for KMeans')
parser.add_argument('--k', type=int, default=4, help='number of segments we are changing')
parser.add_argument('--num_iter', type=int, default=500, help='number of updates')
parser.add_argument('--gpu', type=int, default=500, help='gpu to use')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

os.makedirs(opt.results_dir, exist_ok=True)

dataset = im_dataset(opt.dataroot)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)

model = color_net().cuda().eval()
model.load_state_dict(torch.load('./latest_net_G.pth'))

classifier = models.resnet50(pretrained=True).cuda().eval()
criterion = torch.nn.CrossEntropyLoss()
class_idx = json.load(open("imagenet_class_index.json"))
opt.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

threshold = 0.05 # stopping criteria

for i, (im, file_name) in enumerate(dataset_loader):
    im = im.cuda()

    # Prepare hints, mask, and get current classification
    data, target = util.get_colorization_data(im, opt, model, classifier)
    opt.target = opt.target if opt.targeted else target
    optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()], lr=opt.lr, betas=(0.9, 0.999))

    prev_diff = 0
    for itr in range(opt.num_iter):
        out_rgb, y = util.forward(model, classifier, opt, data)
        val, idx, labels = util.compute_class(opt, y)
        loss = util.compute_loss(opt, y, criterion)
        print(f'[{itr+1}/{opt.num_iter}] Loss: {loss:.3f} Labels: {labels}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("%.5f"%(loss.item()))

        diff = val[0] - val[1]

        if opt.targeted:
            if idx[0] == opt.target and diff > threshold and (diff-prev_diff).abs() < 1e-3:
                break
        else:
            if idx[0] != opt.target and diff > threshold and (diff-prev_diff).abs() < 1e-3:
                break
        prev_diff = diff

    file_name = file_name[0] + '.png'
    save_image(out_rgb, os.path.join(opt.results_dir, file_name))
