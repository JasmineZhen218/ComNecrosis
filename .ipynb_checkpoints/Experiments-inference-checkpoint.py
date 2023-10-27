from inference import *
Settings = {
    '16714495':{'adjust_otsu':1.1, 'num_bag_viable':0, 'num_bag_necrosis':625, 'num_bag_stroma':50},
    '16714498':{'adjust_otsu':1.18, 'num_bag_viable':10, 'num_bag_necrosis':625, 'num_bag_stroma':100},
    '16714499':{'adjust_otsu':1.1, 'num_bag_viable':0, 'num_bag_necrosis':10, 'num_bag_stroma':587},
    '16714503':{'adjust_otsu':1.25, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587},
    '16714505':{'adjust_otsu':1.21, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587},
    '16714507':{'adjust_otsu':1.18, 'num_bag_viable':833, 'num_bag_necrosis':625, 'num_bag_stroma':587}
}

# Inference, using model_mixpatch_x20_w5, for all 6 slides
Slide_IDs = ['16714495','16714498','16714499','16714503','16714505','16714507']
unit = 256
level = 1 # x20 magnification
patch_shape = 256
# load model
model = Attention_modern_multi(load_vgg16())
model.load_state_dict(torch.load('/cis/home/zwang/Data/ComNecrosis/Aaron/model_mixpatch_x20_w5.pth'))

for slide_ID in Slide_IDs:
    # read slides
    adjust_otsu= Settings[slide_ID]['adjust_otsu']
    slide_ob = openslide.OpenSlide(os.path.join('/cis/net/gaon1/data/zwang/Aaron',slide_ID)+'.ndpi')
    width,height = slide_ob.dimensions
    width_downsample, height_downsample = width//unit, height//unit
    thumbnail = slide_ob.get_thumbnail((width_downsample,width_downsample))
    mask_tissue = generate_tissue_mask(slide_ob,unit,adjust_otsu)
#     f, ax = plt.subplots(1,2)
#     ax[0].imshow(thumbnail)
#     ax[1].imshow(mask_tissue)
#     plt.show()
    # Inference
    heatmap_stroma, heatmap_necrosis, heatmap_viable = inference(slide_ob, model, mask_tissue, level, patch_shape)
    np.save('/cis/home/zwang/Data/ComNecrosis/Aaron/'+slide_ID+'/aggregate/heatmap_stroma_mixpatch_20x_w5.npy',heatmap_stroma)
    np.save('/cis/home/zwang/Data/ComNecrosis/Aaron/'+slide_ID+'/aggregate/heatmap_necrosis_mixpatch_20x_w5.npy',heatmap_necrosis)
    np.save('/cis/home/zwang/Data/ComNecrosis/Aaron/'+slide_ID+'/aggregate/heatmap_viable_mixpatch_20x_w5.npy',heatmap_viable)
