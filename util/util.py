from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io
from PIL import Image
import inspect, re
import numpy as np
import os
import os.path
from pathlib import Path
import math
from PIL import Image
import torchvision.transforms as transforms
import collections
from util import order as ORDER
from shutil import copyfile



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def read_image(path, witdh):
    I = Image.open(path).convert('RGB')
    transform = get_transform(witdh)
    return transform(I).unsqueeze(0)

def get_transform(witdh):
    transform_list = []
    osize = [witdh, witdh]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]

    return transforms.Compose(transform_list)

def save_final_image(image, name, save_dir):
    im_numpy = tensor2im(image)
    save_image(im_numpy, os.path.join(save_dir, name + '.png'))

def save_map_image(map_values, name, save_dir, level=0, binary_color=False):
    if level == 0:
        map_values = map_values
    else:
        scale_factor = int(math.pow(2,level-1))
        map_values = upsample_map(map_values, scale_factor)
    if binary_color==True:
        map_image = binary2color_image(map_values)
    else:
        map_image = map2image(map_values)
    save_image(map_image, os.path.join(save_dir, name + '.png'))

def upsample_map(map_values, scale_factor, mode='nearest'):
    if scale_factor == 1:
        return map_values
    else:
        upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsampler(Variable(map_values)).data

def downsample_map(map_values, scale_factor):
    if scale_factor == 1:
        return map_values
    else:
        d = scale_factor
        downsampler = torch.nn.AvgPool2d((d, d), stride=(d, d))
        return downsampler(Variable(map_values)).data

def tensor2im(image_tensor, imtype=np.uint8, index=0):
    image_numpy = image_tensor[index].cpu().float().numpy()
    mean = np.zeros((1,1,3))
    mean[0,0,:] = [0.485, 0.456, 0.406]
    stdv = np.zeros((1,1,3))
    stdv[0,0,:] = [0.229, 0.224, 0.225]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * stdv + mean) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def feature2images(feature, size=[1,1], imtype=np.uint8):
    feature_np = feature.cpu().float().numpy()
    mosaic = np.zeros((size[0]*feature_np.shape[2], size[1]*feature_np.shape[3]))
    for i in range(size[0]):
       for j in range(size[1]):
           single_feature = feature_np[0,i*size[1]+j,:,:]
           stretched_feature = stretch_image(single_feature)
           mosaic[(i*feature_np.shape[2]):(i+1)*(feature_np.shape[2]),
               j*feature_np.shape[3]:(j+1)*(feature_np.shape[3])] = stretched_feature
    mosaic = np.transpose(np.tile(mosaic, [3,1,1]), (1,2,0))
    return mosaic.astype(np.uint8)

def grad2image(grad, imtype=np.uint8):
    grad_np = grad.cpu().float().numpy()
    image = np.zeros((grad.shape[2], grad.shape[3]))
    for i in range(grad_np.shape[1]):
           image = np.maximum(image, grad_np[0,i,:,:])
    return stretch_image(image).astype(imtype)

def batch2im(images_tensor, imtype=np.uint8):
    image_numpy = images_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def map2image(values_map, imtype=np.uint8):
    image_numpy = values_map[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = stretch_image(image_numpy)
    image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def binary2color_image(binary_map, color1=[0,185,252], color2=[245,117,255], imtype=np.uint8):
    assert(binary_map.size(1)==1)
    binary_ref = binary_map[0].cpu().float().numpy()
    binary_ref = np.transpose(binary_ref, (1, 2, 0))
    binary_ref = np.tile(binary_ref, [1,1,3])
    color1_ref = np.tile(np.array(color1), [binary_map.size(2),binary_map.size(3),1])
    color2_ref = np.tile(np.array(color2), [binary_map.size(2),binary_map.size(3),1])
    color_map = binary_ref*color1_ref + (1-binary_ref)*color2_ref

    return color_map.astype(imtype)

def stretch_image(image):
    min_image = np.amin(image)
    max_image = np.amax(image)
    if max_image != min_image:
        return (image - min_image)/(max_image - min_image)*255.0
    else:
        return image

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_tensor_as_mat(tensor, path):
    tensor_numpy = tensor.cpu().numpy()
    print(path)
    scipy.io.savemat(path, mdict={'dna': tensor_numpy})

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_mask(path):
    image = Image.open(path)
    np_image = np.array(image)
    np_image = np_image[:,:,0]
    print(np_image.shape)
    return np.where(np_image>128, 1, 0)

def arrangeSourceDir(path,source,target):
    #default baseDir = "/home/talh/neural_best_buddies/input"
    print("images inside the input_dir(",path,")")
    source_name = os.path.basename(str(source))  
    target_name = os.path.basename(str(target))  
    source_index = -1 
    target_index = -1
    
    image_count = 0
    
    target_dir_path = "/home/talh/neural_best_buddies/input/input_images"

    for filename in sorted(os.listdir(path)):
        if filename.endswith(".png") or filename.endswith(".jpg"): 
            print(os.path.join(path, filename),"index = ",image_count)
            if source_name == os.path.basename(str(filename)):
                source_index = image_count
            elif target_name == os.path.basename(str(filename)):
                target_index = image_count
            if image_count < 10:
                new_name = str("0") + str(image_count) + str(".png")
            else:
                new_name =  str(image_count)+str(".png")
            src_path = os.path.join(path, filename)
            dst_path1 = os.path.join(path, new_name)
            dst_path = os.path.join(target_dir_path, new_name)
            os.rename(src_path,dst_path1)
            # copyfile(src_path, dst_path)
            image_count +=1 
    print("image_count == ", image_count , " source_index == ",source_index, " target_index == ",target_index)
    # print("SOURCE DIR FILES: (",path,")")
    # arr = sorted(os.listdir(path))
    # for i in range(len (arr)):
    #     print (arr[i])
    return image_count,source_index,target_index

def print_separator_line():
    print("---------------------------------------------------------------------")


def get_file_path_from_index(opt,index):
    dir_path = os.path.dirname(str(opt.sourceImg))  
    if index < 10:
        base_filename = str('0') + str(index)
    else:
        base_filename = str(index)
    suffix = '.png'
    img_path = os.path.join(dir_path, base_filename + suffix)
    print("get_file_path_from_index----> ",img_path)
    return img_path

def create_points_matrix (pic_num):
    zeors_array = np.zeros( (pic_num, pic_num))
    return zeors_array