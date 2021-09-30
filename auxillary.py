import xml.etree.ElementTree as ET
import PIL.Image as Image
import numpy as np
import PIL.ImageDraw as ImageDraw
import cv2 as cv
from skimage import morphology
def read_Aaron_annotations(xml_path):
    root = ET.parse(xml_path)
    Annotations={'viable':{'outer':[], 'inner':[]},
                'necrosis':{'outer':[], 'inner':[]},
                'stroma':{'outer':[], 'inner':[]}}
    for a in root.iter('Annotation'):
        for r in a.iter('Region'):
            Annotation = []
            for v in r.iter('Vertex'):
                Annotation.append((float(v.attrib['X']), float(v.attrib['Y'])))
            if a.attrib['LineColor'] == '16711680' :
                Annotations['viable']['outer'].append(Annotation)
            elif a.attrib['LineColor'] == '255':
                Annotations['necrosis']['outer'].append(Annotation)
            elif a.attrib['LineColor'] == '65280' or a.attrib['LineColor'] == '1376057':
                Annotations['stroma']['outer'].append(Annotation)
    return Annotations

def create_viable_mask(slide_ob,Annotations,downsample_scale):
    mask =  Image.new('1', (int(np.round(slide_ob.dimensions[0]/downsample_scale)),int(np.round(slide_ob.dimensions[1]/downsample_scale))))
    draw = ImageDraw.Draw(mask)
    for i in range(len(Annotations[0])):
        Annotation = Annotations[0][i]
        Annotation = [(i[0]/downsample_scale,i[1]/downsample_scale) for i in Annotation]
        draw.polygon(Annotation,fill=1,outline=0)
    for i in range(len(Annotations[1])):
        Annotation = Annotations[1][i]
        Annotation = [(i[0]/downsample_scale,i[1]/downsample_scale) for i in Annotation]
        draw.polygon(Annotation,fill=0,outline=0)
    mask = np.array(mask)
    mask = (mask == 1).astype(np.uint8)
    return mask

def convert_cv_findContours(contours):
    contours_1 = []
    for contour in contours:
        contour_1 = [(i[0][0],i[0][1]) for i in contour]
        contours_1.append(contour_1)
    return contours_1
def find_contours(heatmap,v0=0.6, remove_size = 100, fill_size=100):
    mask = morphology.remove_small_holes(morphology.remove_small_objects(heatmap > v0, remove_size), fill_size)
    contours,_ = cv.findContours(mask.astype(np.uint8), 
                                   mode = cv.RETR_EXTERNAL, 
                                   method=  cv.CHAIN_APPROX_NONE)
    contours = convert_cv_findContours(contours)
    return contours


def write_annotation_0(contours,LineColor,root):
    new_Annotation = ET.Element("Annotation")
    new_Annotation.attrib = {"Id":"1" ,"Name":"" ,"ReadOnly":"0", "NameReadOnly":"0", "LineColorReadOnly":"0" ,"Incremental":"0", "Type":"4", "LineColor":LineColor, "Visible":"1", "Selected":"1", "MarkupImagePath":"", "MacroName":""}
    Attributes = ET.SubElement(new_Annotation, "Attributes")
    Regions = ET.SubElement(new_Annotation, "Regions")
    
    ET.SubElement(Regions,"RegionAttributeHeaders")
    for i in range(len(contours)):
        contour = contours[i]
        Region = ET.SubElement(Regions,"Region")
        Region.attrib = {
            "Id":str(i), "Selected":"0", "ImageLocation":"", "ImageFocus":"-1" ,"Text":"" ,"NegativeROA":"0" ,"InputRegionId":"0", "Analyze":"0", "DisplayId":str(i), "Type":"0"
        }
        ET.SubElement(Region,"Attributes")
        Vertices = ET.SubElement(Region,"Vertices")
        for x,y in contour:
            Vertex = ET.Element("Vertex")
            Vertex.attrib = {'Z':'0','X':str(x),'Y':str(y)}
            Vertices.append(Vertex)
    root.append(new_Annotation)    
