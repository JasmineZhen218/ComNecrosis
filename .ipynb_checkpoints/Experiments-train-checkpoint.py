### Notes
# - Date: `10/03/2021`
# - Contents: train a multi-class LC-MIL model on 5 pilot WSIs of osteosarcoma (leave 1 for validation). Then infer all 6 WSIs.
# - Setting: 
#     - MIX patch: firstly mix all patches from 5 slides, and then pack bags
#     - Magnification: 20x magnification
#     - Training parameters: 
#         - weight decay: 10e-4
#         - initial learning rate: 0.00005
#         - weight decay to 0.2 every 100 bags
key = 'mixpatch_x20_w4'

import torch.optim as optim
import xml.etree.ElementTree as ET
import openslide
import os
import matplotlib.pyplot as plt
import pickle
from auxillary import *
from train import *


unit = 256
level = 1 # 20x magnification
patch_shape = 256
annotation_label_mapping ={
    'stroma':0,
    'viable':1,
    'necrosis':2
}
Settings = {
    '16714495':{'adjust_otsu':1.1, 'num_bag_viable':0, 'num_bag_necrosis':625, 'num_bag_stroma':50},
    '16714498':{'adjust_otsu':1.18, 'num_bag_viable':10, 'num_bag_necrosis':625, 'num_bag_stroma':100},
    '16714499':{'adjust_otsu':1.1, 'num_bag_viable':0, 'num_bag_necrosis':10, 'num_bag_stroma':587},
    '16714503':{'adjust_otsu':1.25, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587},
    '16714505':{'adjust_otsu':1.21, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587},
    '16714507':{'adjust_otsu':1.18, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587}
}
slides_train_ID = ['16714495','16714498','16714499','16714503','16714505']
slides_val_ID = ['16714507']

slides_train = {}
num_bags_train = {}
annotations_train = {}
tissue_masks_train = {}
label_masks_train = {}
for slide_ID in slides_train_ID:
    slides_train[slide_ID] = openslide.OpenSlide(os.path.join('/cis/net/gaon1/data/zwang/Aaron',slide_ID)+'.ndpi')
    annotations_train[slide_ID] = read_Aaron_annotations(os.path.join('/cis/net/gaon1/data/zwang/Aaron',slide_ID)+'.xml')
    num_bags_train[slide_ID] = [Settings[slide_ID]['num_bag_stroma'],Settings[slide_ID]['num_bag_viable'],Settings[slide_ID]['num_bag_necrosis']]
    tissue_masks_train[slide_ID] = generate_tissue_mask(slides_train[slide_ID],unit=unit,adjust_otsu=Settings[slide_ID]['adjust_otsu'])
    label_masks_train[slide_ID] = generate_label_mask(slides_train[slide_ID], annotations_train[slide_ID], annotation_label_mapping, unit)

if len(slides_val_ID)>0:
    slides_val = {}
    num_bags_val = {}
    annotations_val = {}
    tissue_masks_val = {}
    label_masks_val = {}
    for slide_ID in slides_val_ID:
        slides_val[slide_ID] = openslide.OpenSlide(os.path.join('/cis/net/gaon1/data/zwang/Aaron' ,slide_ID)+'.ndpi')
        annotations_val[slide_ID] = read_Aaron_annotations(os.path.join('/cis/net/gaon1/data/zwang/Aaron',slide_ID)+'.xml')
        num_bags_val[slide_ID] = [Settings[slide_ID]['num_bag_stroma'],Settings[slide_ID]['num_bag_viable'],Settings[slide_ID]['num_bag_necrosis']]
        tissue_masks_val[slide_ID] = generate_tissue_mask(slides_val[slide_ID],unit=unit,adjust_otsu=Settings[slide_ID]['adjust_otsu'])
        label_masks_val[slide_ID] = generate_label_mask(slides_val[slide_ID], annotations_val[slide_ID], annotation_label_mapping, unit)

# n_train = len(slides_train_ID)
# f, ax = plt.subplots(n_train,3,figsize=(15,5*n_train))
# for i in range(n_train):
#     slide_ID = slides_train_ID[i]
#     slide_ob = slides_train[slide_ID]
#     ax[i][0].imshow(slide_ob.get_thumbnail((slide_ob.dimensions[0]/unit,slide_ob.dimensions[1]/unit)))
#     ax[i][1].imshow(tissue_masks_train[slide_ID])
#     ax[i][2].imshow(label_masks_train[slide_ID])

# if len(slides_val_ID)>0:    
#     n_val = len(slides_val_ID)
#     f, ax = plt.subplots(n_val+1,3,figsize=(15,5*n_val))
#     for i in range(n_val):
#         slide_ID = slides_val_ID[i]
#         slide_ob = slides_val[slide_ID]
#         ax[i][0].imshow(slide_ob.get_thumbnail((slide_ob.dimensions[0]/unit,slide_ob.dimensions[1]/unit)))
#         ax[i][1].imshow(tissue_masks_val[slide_ID])
#         ax[i][2].imshow(label_masks_val[slide_ID])

model = Attention_modern_multi(load_vgg16(),True)
optimizer = optim.Adam(model.parameters(),lr=0.00005, betas=(0.9, 0.999), weight_decay =10e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.2)

Dataset_train = create_dataset_mixbag(slides_train,
                                  tissue_masks_train, label_masks_train, 
                                  num_bags_train, level, patch_shape,length_bag_mean = 10)
if len(slides_val)>0:
    Dataset_val = create_dataset_mixbag(slides_val,
                                  tissue_masks_val, label_masks_val, 
                                  num_bags_val, level, patch_shape,length_bag_mean = 10)
    model, Train_loss, Train_accuracy, Val_loss, Val_accuracy = Train(model, Dataset_train, optimizer, scheduler, validation=True, Dataset_val = Dataset_val)
else:
    model, Train_loss, Train_accuracy, Val_loss, Val_accuracy = Train(model, Dataset_train, optimizer, scheduler, validation=False, Dataset_val = None)
    

# save model
torch.save(model.state_dict(), '/cis/home/zwang/Data/ComNecrosis/Aaron/model_'+key+'.pth')
# Saving training process
TP = {
    'Train_loss': Train_loss,
    'Train_accuracy': Train_accuracy,
    'Val_accuracy': Val_accuracy,
    'Val_loss':Val_loss
}
with open('/cis/home/zwang/Data/ComNecrosis/Aaron/TP_'+key+'.sav', 'wb') as handle:
    pickle.dump(TP, handle, protocol = pickle.HIGHEST_PROTOCOL)
