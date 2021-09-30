# load modules and functions
import os
import openslide
import numpy as np
import PIL.Image as Image
from skimage import morphology
import cv2 as cv
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
# # read slides
# slide_ob = openslide.OpenSlide(os.path.join(args.slide_root,args.slide_ID)+'.ndpi')
# # load model
# model = Attention_modern_multi(load_vgg16())
# model.load_state_dict(torch.load(args.model_path))
# # Inference
# heatmap_stroma, heatmap_necrosis, heatmap_viable = inference(slide_ob, model, unit, level,patch_shape)
# # saving
# np.save('results/Aaron/'+slide_ID+'/aggregate/heatmap_stroma_10x.npy',heatmap_stroma)
# np.save('results/Aaron/'+slide_ID+'/aggregate/heatmap_necrosis_10x.npy',heatmap_necrosis)
# np.save('results/Aaron/'+slide_ID+'/aggregate/heatmap_viable_10x.npy',heatmap_viable)

def binary_Aaron(img,adjust_otsu = 1.18, fill_size=100, remove_size=100):
    # 16714505 1.21
    # 16714503 1.25
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    otsu_threshold, _ = cv.threshold(gray, 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary = gray<= (otsu_threshold*adjust_otsu)
    binary = morphology.remove_small_objects(morphology.remove_small_holes(binary, fill_size),remove_size)
    return binary

def generate_tissue_mask(slide_ob,unit,adjust_otsu=1):
    """
    Generate tissue mask (downsampled) for WSI
    - Input
        slide_ob: slide object
        unit: downsample scale
        adjust_otsu: adjust the OTSU threshold
    - Return
        mask tissue
    """
    width,height = slide_ob.dimensions
    width_downsample, height_downsample = width//unit, height//unit
    thumbnail = slide_ob.get_thumbnail((width_downsample,width_downsample))
    thumbnail = cv.resize(np.array(thumbnail)[:,:,:3],(width_downsample,height_downsample))
    mask_tissue = np.array(binary_Aaron(thumbnail, adjust_otsu),dtype=np.uint8) 
    return mask_tissue
def inference(slide_ob, model, mask_tissue, level, patch_shape):
    """
    Inference
    - Input
        slide_ob
        model: trained model
        unit: downsampled scale of heatmaps (compared with x40 magnification)
        level: resolution of patches (level of WSI)
        patch_shape: size of patch at corresponding level
    -Return
        heatmap_stroma
        heatmap_necrosis
        heatmap_viable
    """
    # dataset
    unit = int(slide_ob.dimensions[0]/mask_tissue.shape[1])
    apply_set = TSbags_apply_neighbor(slide_ob = slide_ob,
                                  unit=unit,
                                  ROW=np.where(mask_tissue==1)[0],
                                  COL=np.where(mask_tissue==1)[1],
                                  level = level,
                                  patch_shape = patch_shape,
                                  transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    apply_loader = data_utils.DataLoader(apply_set, batch_size = 1, shuffle = False)
    # inference
    # initialize heatmaps
    heatmap_stroma = np.zeros_like(mask_tissue,dtype=float)
    heatmap_necrosis = np.zeros_like(mask_tissue,dtype=float)
    heatmap_viable = np.zeros_like(mask_tissue,dtype=float)
    heatmap_stroma[:] = np.nan
    heatmap_necrosis[:] = np.nan
    heatmap_viable[:] = np.nan
    device = torch.device("cuda")
    model.to(device=device)
    for batch_idx, (bag, row_center_bag, col_center_bag) in enumerate(apply_loader):
        bag = bag.to(device=device, dtype=torch.float)
        bag = Variable(bag,requires_grad=False)
        Y_prob, _ = model.forward(bag)
        Y_prob = Y_prob.cpu().data.numpy()[0]
        del bag
        row_center_bag = row_center_bag.cpu().data.numpy()[0]
        col_center_bag = col_center_bag.cpu().data.numpy()[0]
        if batch_idx%1000 == 0:
            print(batch_idx,'/',len(apply_set),Y_prob) 
        heatmap_stroma[row_center_bag,col_center_bag] = Y_prob[0]
        heatmap_viable[row_center_bag,col_center_bag] = Y_prob[1]
        heatmap_necrosis[row_center_bag,col_center_bag] = Y_prob[2]
    return heatmap_stroma, heatmap_necrosis, heatmap_viable

class Attention_modern_multi(nn.Module):
    def __init__(self,cnn,focal_loss=False):
        super(Attention_modern_multi,self).__init__()
        self.L = 1000
        self.D = 64
        self.K = 1 
        self.focal_loss = focal_loss     
        self.feature_extractor = cnn      
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 3)
        )
    def forward(self,x):
        x = x .squeeze(0)
        H = self.feature_extractor(x)
        A = self.attention(H)
        A = torch.transpose(A,1,0)
        A = F.softmax(A,dim=1)
        M = torch.mm(A,H)
        Y_prob = self.classifier(M)
        Y_prob = F.softmax(Y_prob,dim=1)
        return Y_prob, A
    def calculate_classification_error(self, X, Y):
        Y_prob,_ = self.forward(X)
        Y_hat = torch.argmax(Y_prob[0])
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat, Y_prob  
    def calculate_objective(self, X, Y):
        Y_prob, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if not self.focal_loss:
            loss = nn.CrossEntropyLoss()
            neg_log_likelihood = loss(Y_prob, Y)
        else:
            Y_prob_target = Y_prob[0,Y.cpu().data]  
            if Y_prob_target.cpu().data.numpy()[0]<0.2:
                gamma = 5
            else:
                gamma = 3
            neg_log_likelihood =-1. *(1-Y_prob_target)**gamma* torch.log(Y_prob_target)
        return neg_log_likelihood, A
def load_vgg16():
    vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    num_layer = 0
    for child in vgg.children():
        num_layer+=1
        if num_layer < 3:
            for param in child.parameters():
                param.requires_grad = False  
    return vgg
class TSbags_apply_neighbor(data_utils.Dataset):
    def __init__(self,slide_ob, unit, ROW, COL, level, patch_shape, transform = None):
        self.slide_ob = slide_ob
        self.unit = unit
        self.ROW, self.COL = ROW, COL
        self.level = level
        self.patch_shape = patch_shape
        self.transform = transform        
    def pack_one_bag(self, row, col):
        factor = self.slide_ob.level_downsamples[self.level]
        upperLeft_bag_x = int(col * self.unit+self.unit/2-(self.patch_shape)/2*factor)
        upperLeft_bag_y = int(row * self.unit+self.unit/2-(self.patch_shape)/2*factor)
        patch = Image.fromarray(np.array(self.slide_ob.read_region((upperLeft_bag_x,upperLeft_bag_y),self.level,(self.patch_shape,self.patch_shape)))[:,:,:3])
        if self.transform is not None:
            patch = self.transform(patch)        
        bag = np.stack([patch], axis=0)
        return bag
    def __len__(self):
        return len(self.ROW)
    def __getitem__(self,idx):
        row_center_bag, col_center_bag = self.ROW[idx], self.COL[idx]
        bag = self.pack_one_bag(row_center_bag, col_center_bag)
        return bag, row_center_bag, col_center_bag