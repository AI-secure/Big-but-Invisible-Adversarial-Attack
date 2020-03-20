##This is a bare minimum code.. please ask question if you have any trouble. Thanks!

from __future__ import print_function, division                                                             
import time
import os
import sys
#image_dir = os.getcwd() + '/Images/'
model_dir = 'Models/'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict


from scipy.misc import imresize
from scipy.misc import imread
from scipy.misc import imsave

import pudb
import numpy as np
from jpeg import jpeg_compress_decompress

weight = float(sys.argv[1])
adv_weight = float(sys.argv[2])
max_iter = int(sys.argv[3])
result_folder = str(sys.argv[4])
model = str(sys.argv[5])
jpeg = str(sys.argv[6])

#pudb.set_trace()
try:
    os.makedirs(result_folder +'/weight_%d_adv_%4f/target'%(weight, adv_weight))
    os.makedirs(result_folder +'/weight_%d_adv_%4f/intermediate'%(weight, adv_weight))
    os.makedirs(result_folder +'/weight_%d_adv_%4f/original'%(weight, adv_weight))
    os.makedirs(result_folder +'/weight_%d_adv_%4f/l1_perturb'%(weight, adv_weight))
except OSError:
    pass

#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
    
img_neighbor = np.load('img_neighbors_pairs_train_test.npy')

celoss = nn.CrossEntropyLoss()
marginLoss = nn.MultiMarginLoss(margin=1)
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = torch.log(nn.MSELoss()(GramMatrix()(input), target))
        return(out)
    
class CrossGramMatrix(nn.Module):
    def forward(self, input1, input2):
        b1,c1,h1,w1 = input1.size()
        F1 = input1.view(b1, c1, h1*w1)
        ms = torch.nn.Upsample(size = (h1,w1), mode='bilinear')
        input2 = ms(input2)
        b2,c2,h2,w2 = input2.size()
        F2 = input2.view(b2, c2, h2*w2)

        G = torch.bmm(F1, F2.transpose(1,2)) 
        G.div_(h1*w1)
        return G    
        
class CrossGramMSELoss(nn.Module):
    def forward(self, input1, input2, target):
        out = (nn.MSELoss()(CrossGramMatrix()(input1,input2), target))
        return(out)    
    
# pre and post processing for images
img_size = 224 
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1, 1, 1]),
                           #transforms.Lambda(lambda x: x.mul_(255)),
                          ])

prep_res = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], #subtract imagenet mean
                                                std=[0.229, 0.224, 0.225]),
                           #transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([#transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1, 1, 1]),
                           #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
                    
postpb = transforms.Compose([transforms.ToPILImage()])
def post_normalize(tensor):
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    #t = F.upsample_bilinear(t.unsqueeze(0), size=(256, 256), scale_factor=None)
    #r =  torch.randint(256-224, (2, )).long()
    #t = t.squeeze(0)[:, r[0]:224, r[1]:224]
    t = prep_res(t)
    return t
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

#get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()
vgg.eval()
#define layers, loss functions, weights and compute optimization targets
style_layers = ['r11','r21','r31','r41', 'r51'] 
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [CrossGramMSELoss()] *( len(style_layers)-1) + [nn.MSELoss()] * len(content_layers)
#print(len(loss_fns),len([CrossGramMSELoss()]),len(style_layers),len([nn.MSELoss()])) 
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
#print(len(loss_fns),len([CrossGramMSELoss()]),len(style_layers))    
#these are good weights settings:
style_weights = [weight/n**2 for n in [64,128,256,512,512]]
content_weights = [0]

weights = style_weights + content_weights

if model == 'vgg':
	res50 = torchvision.models.vgg19(pretrained=True).cuda()#
	print('vgg model loaded')

elif model == 'densenet':
	res50 = torchvision.models.densenet121(pretrained=True).cuda()
	print('densenet model loaded')

elif model == 'resnet':
	res50 = torchvision.models.resnet50(pretrained=True).cuda() 
	print('resnet model loaded')
res50.eval()

class_dic={}
class_dic['n01774750'] = 76
class_dic['n01855032'] = 98
class_dic['n01968897'] = 117
class_dic['n02117135'] = 276
class_dic['n02814860'] = 437
class_dic['n03445924'] = 575
class_dic['n03924679'] = 713
class_dic['n04507155'] = 879
class_dic['n07695742'] = 932
class_dic['n09421951'] = 977

target_count = 0
intermediate_count = 0
original_count = 0
target_pred = torch.LongTensor(1, 1).cuda()

for i in range(0, 10):
    victim_img_all = img_neighbor[i][0]
    for j, images in enumerate(victim_img_all):
        victim_img = victim_img_all[j]
        content_id= os.path.join('/data/attack/processed_dataset_20/', victim_img[:9] , victim_img)
        for k in range(10):
            target_img = img_neighbor[i][k][j]

            style_id= os.path.join('/data/attack/processed_dataset/', target_img[:9] , target_img)
            
            #print(img_neighbor[i][0], img_neighbor[i][1])
            img_names = [style_id, content_id] #load images, ordered as [style_image, content_image]

            imgs = [Image.open(name) for i,name in enumerate(img_names)]
            imgs_torch = [prep(img) for img in imgs]
            if torch.cuda.is_available():
                imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
            else:
                imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
            style_image, content_image = imgs_torch

            opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True)
            opt_img.data.copy_(content_image) 
            #random init
            #opt_img = Variable(content_image.data.clone(), requires_grad=True)

            #compute optimization targets
            Crossstyle_targets = [CrossGramMatrix()(A,B).detach() for A, B in zip(vgg(style_image, style_layers[:4]), vgg(style_image, style_layers[1:]))]
            content_targets = [A.detach() for A in vgg(content_image, content_layers)]
            targets = Crossstyle_targets + content_targets

            #run style transfer
            #max_iter = 3
            show_iter = 1
            optimizer = optim.LBFGS([opt_img], max_iter=14);
            n_iter=0
            #print("original:%d, target:%d"%(class_dic[img_neighbor[idx][0][:9]], class_dic[img_neighbor[idx][1][:9]]))
            content_pred  = class_dic[victim_img[:9]]#.max(res50(post_normalize(content_image.squeeze(0)).unsqueeze(0)), 1)
            target_pred = class_dic[target_img[:9]]
            if content_pred == target_pred:
                continue
            target_pred =  torch.tensor(target_pred).long().unsqueeze(0)
            #print(torch.max(res50(post_normalize(style_image.squeeze(0)).unsqueeze(0)), 1))
            target_pred = target_pred.cuda()  
            for iter in range(max_iter):

                def closure():
                    optimizer.zero_grad()
                    Cross_out = zip(vgg(opt_img, style_layers[:4]),vgg(opt_img, style_layers[1:] ))
                    content_out = vgg(opt_img, content_layers)

                    #pudb.set_trace()

                    layer_losses = [weights[a] * loss_fns[a](A,B, Crossstyle_targets[a])/CrossGramMatrix()(A,B).std()  for a,(A,B) in enumerate(Cross_out)]
                    content_losses = [weights[4+a] * loss_fns[4+a](A, content_targets[a])  for a,A in enumerate(content_out)]
                    loss = sum(layer_losses)+sum(content_losses)

                    #adv_loss =0
            	    norm_opt_img = opt_img.clone()
                    #opt_img_res = opt_img[:, [2, 1, 0]]
                    global confidence
                    global pred

                    if jpeg == 'True':
                        if iter==0 and k==0:
                            print ("adding_jpeg")
                        k = 0.70
                        jpg_opt_img = jpeg_compress_decompress(norm_opt_img.permute(0,2,3,1) * 255, factor=float(k)).permute(0,3,1,2)/255
                        jpg_norms = post_normalize(jpg_opt_img.squeeze(0)).unsqueeze(0)
                        #jpg_norms = torch.stack(jpg_norms).squeeze()
                        norm_opt_img = post_normalize(norm_opt_img.squeeze(0)).unsqueeze(0)
                        logits = res50(torch.cat([jpg_norms, norm_opt_img], 0))
                        #all_pred =   F.softmax(logits)   
                        #confidence, pred = torch.max(all_pred, 1)
                    
                    #for k in np.arange(0.5,20,1):
                    
                    #jpg_opt_img = jpeg_compress_decompress(norm_opt_img.permute(0,2,3,1) * 255, factor=float(1)).permute(0,3,1,2)/255
                    #jpg_opt_img = post_normalize(jpg_opt_img.squeeze(0)).unsqueeze(0)
                    else:
                        norm_opt_img = post_normalize(norm_opt_img.squeeze(0)).unsqueeze(0)
                        logits = res50(norm_opt_img)

                    confidence, pred = torch.topk(F.softmax(logits), 3)             
                    adv_loss = celoss(logits, target_pred.repeat(logits.shape[0]))

                    

                    #adv_loss = celoss(logits, target_pred)

                    #adv_loss = marginLoss(logits, target_pred)

                    total_loss = loss+ adv_weight*adv_loss

                    total_loss.backward()

            #         if iter%show_iter == (show_iter-1):
            #             print('Iteration: %d, loss: %f, adv_loss: %f, pred: %d, target: %d'%(iter+1, loss.data[0], adv_loss.data[0], pred.data[0], target_pred.data[0]))
            # #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer

                    
                    return total_loss
                loss1 = optimizer.step(closure)


                if content_pred not in pred[0]:
                    break
                #loss1 = optimizer.step(closure)

            pred = pred[0]
            confidence = confidence[0]
            #pudb.set_trace()
            #print(content_id, style_id, content_pred, target_pred.data[0].cpu().numpy(), pred.data[0].cpu().numpy(), confidence.data[0].cpu().numpy())
            
            out_img_f = postp(opt_img.data[0].cpu().squeeze())

            #content_img_f = postp(content_image.data[0].cpu().squeeze())

            l1_perturb = (opt_img-content_image)
            l1_perturb_f = postpb(l1_perturb.data[0].cpu().squeeze())  
            #pudb.set_trace()
            if target_pred.data[0].cpu().numpy() == pred.data[0].cpu().numpy():
                imsave(result_folder +'/weight_%d_adv_%f/target/%s_%s_O%d_T%d_P%d_C%f.png'%(weight, adv_weight, victim_img[:-4], target_img[:-4], content_pred, 
                                                                    target_pred.data[0].cpu().numpy(), 
                                                                    pred.data[0].cpu().numpy(), confidence.data[0].cpu().numpy()), out_img_f)
                target_count+=1
            elif pred.data[0].cpu().numpy() == np.array(content_pred):
                imsave(result_folder +'/weight_%d_adv_%f/original/%s_%s_O%d_T%d_P%d_C%f.png'%(weight, adv_weight, victim_img[:-4], target_img[:-4], content_pred, 
                                                                                        target_pred.data[0].cpu().numpy(), pred.data[0].cpu().numpy(), 
                                                                                    confidence.data[0].cpu().numpy()), out_img_f)
                original_count+=1
            else:
                imsave(result_folder +'/weight_%d_adv_%f/intermediate/%s_%s_O%d_T%d_P%d_C%f.png'%(weight, adv_weight, victim_img[:-4], target_img[:-4], content_pred, 
                                                                                        target_pred.data[0].cpu().numpy(), pred.data[0].cpu().numpy(), 
                                                                                        confidence.data[0].cpu().numpy()), out_img_f)
                intermediate_count+=1

            imsave(result_folder +'/weight_%d_adv_%f/l1_perturb/%s_%s_O%d_T%d_P%d_C%f.png'%(weight, adv_weight, victim_img[:-4], target_img[:-4], content_pred, 
                                                                                        target_pred.data[0].cpu().numpy(), pred.data[0].cpu().numpy(), 
                                                                                        confidence.data[0].cpu().numpy()), l1_perturb_f)
                
            if k%4==0:
                print(iter)
