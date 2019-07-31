import math   # This will import math module
from util import util
from . import feature_metric as FM
import os.path
import numpy as np
from torch.nn import functional as F

def run_NBB_and_get_points(opt, nbbs):
    A = util.read_image(opt.sourceImg, opt.imageSize)
    B = util.read_image(opt.targetImg, opt.imageSize)
    points = nbbs.run(A, B)
    new_file_name = os.path.basename(opt.sourceImg)+ "_" + os.path.basename(opt.targetImg)+".txt"

    with open(os.path.join(opt.results_dir, new_file_name), 'w') as f:
      for item in points:
        f.write("%s\n" % item)
    f.close()
    dist = get_distance(points,A,B)  
    
    return dist,  points
    #return get_images_points_and_distance(points)

def get_euclidean_distance(A, B):
    print("Start get_euclidean_distance")
    distance = 0
    num_of_points = 0
    for Apoint, Bpoint in zip(A, B):
        num_of_points += 1
        A_y = Apoint[0]
        A_x = Apoint[1]
        B_y = Bpoint[0]
        B_x = Bpoint[1]
        temp =  math.sqrt(math.pow((A_y-B_y), 2)  + math.pow((A_x-B_x), 2))
        distance += temp
    return distance/num_of_points

def get_images_points_and_distance(points):
    A_points_positions = []
    B_points_positions = []
    for p1,p2, ac in zip(*points):
        A_points_positions.append(p1)
        B_points_positions.append(p2)
        print(p1, p2, '\t', ac)
    distance = get_euclidean_distance(A_points_positions,B_points_positions)
    print("The Euclidean distance between all images points is: ", distance)
    new_points = [A_points_positions, B_points_positions]
    return distance, new_points

# print("points -- patch_distance: ")
# print(FM.patch_distance(A,B))

def reg_distance(correspondense,A,B):
	return np.square((np.array(correspondense[0])-np.array(correspondense[1]))).sum()
def ptc_distance(correspondense,A,B,patch_size):
	patch_size = 10
	dists = np.array([get_inv_patch_distance(
					get_patch(A,xa,ya,patch_size),
					get_patch(B,xb,yb,patch_size))
					for (xa,ya),(xb,yb),ac in zip(*correspondense)])+1
	return (np.square(np.array(correspondense[0])-np.array(correspondense[1])).sum(axis=1)*dists).sum()

def get_patch(A,x,y,size):
#	return A[:,:,x-size:x+size+1,y-size:y+size+1]
	return F.pad(A,[size,size,size,size,0,0,0,0]).narrow(2,x,2*size+1).narrow(3,y,2*size+1)

def normalize_patch(A):
	sz = A.size(2)*A.size(3)
	A = A.clone()
	A -= A.sum((2,3),keepdim=True)/sz # avg-0
	A /= (A.pow(2).sum((2,3),keepdim=True)/sz).pow(0.5)  # var-1
	return A

def get_inv_patch_distance(A,B): # 1 - closest, -1 - farthest
	A_norm = normalize_patch(A)
	B_norm = normalize_patch(B)
	return (A_norm*B_norm).sum() / (A_norm.pow(2).sum()*B_norm.pow(2).sum()).pow(0.5)


get_distance = lambda c,A,B: ptc_distance(c,A,B,2)