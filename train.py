import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from scipy.stats import binom
import cv2 as cv
import numpy as np
from skimage import morphology
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import random

# pack bags from each slide, and then aggregate bags
def Train(model, Dataset_train, optimizer, scheduler, validation=False, Dataset_val = None):
    n_train = len(Dataset_train)
    split_train = 100
    indices_train = random.sample(list(range(n_train)),k=n_train)
    Train_loss = []
    Train_accuracy = []
    Val_loss = []
    Val_accuracy = []
    if validation:
        n_val = len(Dataset_val)
        indices_val = random.sample(list(range(n_val)),k=100)
    for i in range(n_train//split_train):
        Sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train[i*split_train:(i+1)*split_train])
        Dataloader_train = data_utils.DataLoader(Dataset_train, sampler = Sampler_train, batch_size = 1, shuffle = False)
        model.cuda()
        model, train_loss, train_accuracy = train(model, optimizer, Dataloader_train)
        Train_loss.append(train_loss)
        Train_accuracy.append(train_accuracy)
        scheduler.step()
        print("epoch={}/{}, train loss = {:.3f}, train_accuracy = {:.3f}".format(i, n_train//split_train, train_loss, train_accuracy))
        if validation:
            Sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
            Dataloader_val = data_utils.DataLoader(Dataset_val, sampler = Sampler_val, batch_size = 1, shuffle = False)
            val_loss, val_accuracy = val(model, Dataloader_val)
            Val_loss.append(val_loss)
            Val_accuracy.append(val_accuracy)
            print("epoch={}/{}, val loss = {:.3f}, val_accuracy = {:.3f}".format(i, n_train//split_train, val_loss, val_accuracy))
    return model, Train_loss, Train_accuracy, Val_loss, Val_accuracy
            

def create_dataset_mixbag(slides, tissue_masks, label_masks, num_bags, level, patch_shape,length_bag_mean = 10):
    """
    Generate data loaders
    - Input
        slides: dictionary {'slide_ID':slide_ob}
        tissue_masks: dictionary {'slide_ID':array}
        label_masks: dictionary {'slide_ID':array}
        num_bags:dict{'slide_ID':list}
        level:int
        patch:int
    - Return
        Dataset
    """
    # Training loaders
    Num_slide = 0
    for slide_ID in slides.keys():
        dataset = TSbags_random_oneslide(slide_ob = slides[slide_ID],
                                            mask_tissue = tissue_masks[slide_ID], 
                                            mask_label = label_masks[slide_ID], 
                                            level = level, 
                                            patch_shape = patch_shape, 
                                            length_bag_mean = length_bag_mean, 
                                            num_bags = num_bags[slide_ID], 
                                          transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))      
        if Num_slide == 0:
            Dataset = dataset
        else:
            Dataset = torch.utils.data.ConcatDataset([Dataset, dataset])
        Num_slide += 1
    return Dataset

def create_dataset_mixpatch(slides, tissue_masks, label_masks, num_bags, level, patch_shape,length_bag_mean = 10):
    """
    Generate data loaders
    - Input
        slides: dictionary {'slide_ID':slide_ob}
        tissue_masks: dictionary {'slide_ID':array}
        label_masks: dictionary {'slide_ID':array}
        num_bags:list
        level:int
        patch_shape:int
    - Return
        Dataset
    """
    # Training loaders
    Dataset = TSbags_random_mixpatch(slide_obs = slides,
                                            masks_tissue = tissue_masks, 
                                            masks_label = label_masks, 
                                            level = level, 
                                            patch_shape = patch_shape, 
                                            length_bag_mean = length_bag_mean, 
                                            num_bags = num_bags, 
                                          transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))      
    return Dataset



def train(model, optimizer, Dataloader_train):
    model.train()
    train_loss = 0.
    train_error = 0.      
    optimizer.zero_grad()
    for batch_idx, (data, label) in enumerate(Dataloader_train):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ = model.calculate_classification_error(data, bag_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.data.cpu().numpy()[0]
        train_error += error
        del data
        del bag_label
    train_loss /= len(Dataloader_train)
    train_error /= len(Dataloader_train)
    return model, train_loss, 1-train_error

def val(model, Dataloader_val):
    val_loss = 0.
    val_error = 0.      
    for batch_idx, (data, label) in enumerate(Dataloader_val):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data,requires_grad=False), Variable(bag_label,requires_grad=False)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ = model.calculate_classification_error(data, bag_label)
        val_loss += loss.data.cpu().numpy()[0]
        val_error += error
        del data
        del bag_label
    val_loss /= len(Dataloader_val)
    val_error /= len(Dataloader_val)
    return val_loss, 1-val_error
    
    
    
class TSbags_random_oneslide(data_utils.Dataset):
    def __init__(self, slide_ob, mask_tissue, mask_label, level, patch_shape, length_bag_mean, num_bags, transform):
        self.slide_ob = slide_ob
        self.mask_tissue = mask_tissue
        self.mask_label = mask_label
        self.level = level
        self.patch_shape =patch_shape
        self.length_bag_mean = length_bag_mean
        self.num_bags = num_bags
        self.transform = transform
        self.unit = int(self.slide_ob.dimensions[0]/mask_tissue.shape[1])
        self.bags_list, self.labels_list = self._create_bags()  
    def _create_bags(self):            
        bags_list = []
        labels_list = []
        for label in range(len(self.num_bags)):
            ROW, COL = np.where((self.mask_tissue==1)&(self.mask_label==label))
            for bag_idx in range(self.num_bags[label]):
                length_bag = binom.rvs (n=int(self.length_bag_mean*2), p=0.5)
                indices = np.random.randint(0,len(ROW),length_bag)
                bags_list.append((ROW[indices], COL[indices]))
                labels_list.append(label)
        return bags_list, labels_list
    def _pack_one_bag(self,row_list, col_list):
        Bag = []
        for i in range(len(row_list)):
            row_unit, col_unit = row_list[i], col_list[i]
            factor = self.slide_ob.level_downsamples[self.level]
            upperLeft_x = int(col_unit * self.unit + self.unit/2 - self.patch_shape/2*factor)
            upperLeft_y = int(row_unit * self.unit + self.unit/2 - self.patch_shape/2*factor)
            patch = self.slide_ob.read_region((upperLeft_x, upperLeft_y),self.level,(self.patch_shape,self.patch_shape))
            patch = Image.fromarray(np.array(patch)[:,:,:3])
            if self.transform is not None:
                patch = self.transform(patch)
            Bag.append(patch)
        Bag = np.stack(Bag,axis=0)
        return Bag  
    def __len__(self):
        return len(self.bags_list)  
    def __getitem__(self, index):
        row_list, col_list = self.bags_list[index]
        bag = self._pack_one_bag(row_list, col_list)
        label = self.labels_list[index]
        return bag, label
    
    
class TSbags_random_mixpatch(data_utils.Dataset):
    def __init__(self, slide_obs, masks_tissue, masks_label, level, patch_shape, length_bag_mean, num_bags, transform):
        self.slide_obs = slide_obs
        self.masks_tissue = masks_tissue
        self.masks_label = masks_label
        self.level = level
        self.patch_shape =patch_shape
        self.length_bag_mean = length_bag_mean
        self.num_bags = num_bags
        self.transform = transform
        self.Patch_list, self.Label_patch_list = self._mix_patches()
        self.bags_list, self.labels_list = self._create_bags()  
        
    def _mix_patches(self):
        # to return: [(slide_ID,row,col)]
        Patch_list = []
        Label_patch_list = []
        for slide_ID in self.slide_obs.keys():
            mask_tissue = self.masks_tissue[slide_ID]
            mask_label = self.masks_label[slide_ID]
            for label in range(len(self.num_bags)):
                ROW, COL = np.where((mask_tissue==1)&(mask_label==label))
                Patch_list.extend([(slide_ID,ROW[i],COL[i]) for i in range(len(ROW))])
                Label_patch_list.extend([label]*len(ROW))
        return Patch_list, np.array(Label_patch_list)
            
    def _create_bags(self):            
        bags_list = []
        labels_list = []
        for label in range(len(self.num_bags)):
            Indices = np.where(self.Label_patch_list==label)[0]
            for bag_idx in range(self.num_bags[label]):
                length_bag = binom.rvs (n=int(self.length_bag_mean*2), p=0.5)
                indices = random.sample(Indices.tolist(),length_bag)
                bags_list.append(indices)
                labels_list.append(label)
        return bags_list, labels_list
    def _pack_one_bag(self,indices):
        Bag = []
        for index in indices:
            slide_ID, row_unit, col_unit = self.Patch_list[index]
            factor = self.slide_obs[slide_ID].level_downsamples[self.level]
            unit = int(self.slide_obs[slide_ID].dimensions[0]/self.masks_tissue[slide_ID].shape[1])
            upperLeft_x = int(col_unit * unit + unit/2 - self.patch_shape/2*factor)
            upperLeft_y = int(row_unit * unit + unit/2 - self.patch_shape/2*factor)
            patch = self.slide_obs[slide_ID].read_region((upperLeft_x, upperLeft_y),self.level,(self.patch_shape,self.patch_shape))
            patch = Image.fromarray(np.array(patch)[:,:,:3])
            if self.transform is not None:
                patch = self.transform(patch)
            Bag.append(patch)
        Bag = np.stack(Bag,axis=0)
        return Bag  
    def __len__(self):
        return len(self.bags_list)  
    def __getitem__(self, index):
        indices = self.bags_list[index]
        bag = self._pack_one_bag(indices)
        label = self.labels_list[index]
        return bag, label
    
     
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

def generate_label_mask(slide_ob, annotations, annotation_label_mapping, unit):
    """
    Generate label mask (downsampled) for WSI
    - Input
        slide_ob: slide object
        annotations: {'annotation_key':{'outer': [(x,y),....],'inner':[(x,y),...]}}
        'annotation_label_mapping': {'annotation_key': int}
        unit: downsample scale
    - Return
        Mask: label mask
    """
    width, height = slide_ob.dimensions
    Mask = np.zeros((int(height/unit),int(width/unit)),dtype=float)
    Mask[:] = np.nan
    for annotation_key in annotations.keys():
        mask =  Image.new('1', (int(np.round(width/unit)),int(np.round(height/unit))))
        draw = ImageDraw.Draw(mask)
        for contour in annotations[annotation_key]['outer']:
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            draw.polygon(contour,fill=1,outline=0)
        for contour in annotations[annotation_key]['inner']:
            contour = [(i[0]/unit,i[1]/unit) for i in contour]
            draw.polygon(contour,fill=0,outline=0)
        mask = np.array(mask)
        Mask[mask==1] = annotation_label_mapping[annotation_key]
    return Mask