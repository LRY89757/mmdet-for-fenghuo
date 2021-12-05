import json
import glob
import os

Classes = []
images = glob.glob(pathname='/home/lry/data/780b/*json')
# print(images)

for img in images:
    with open(img, 'r') as f:
        shapes_img = json.load(f)['shapes']
    for sing in shapes_img:
        Classes.append(sing['label'])
    Classes = list(set(Classes))

print(Classes)

# Classes = ['GFI1-1', 'O622-8', 'GFI1-2', 'XGI2', 'E1-63', 'fake', 'O2500-4', 'O9953', 'unrecognizable', 'O622-4', 'GFI2', 'others', '10GFEC', 'empty', '780B', 'GFI2-R', 'O155-8', 'GFI1-3', 'O2500']
# label = [cls for cls in Classes if cls in 'GFI1-2XSLJDLJFLJSKFLJ'][0]
# print(label)

