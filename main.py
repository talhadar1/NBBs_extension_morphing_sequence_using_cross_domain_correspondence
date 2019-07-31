import os
import numpy as np
from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
from util import util
from util import MLS
from util import order as ORDER
import warnings
from numpy import median
from options.options import Options

opt = Options().parse()
vgg19 = vgg19_model.define_Vgg19(opt)
save_dir = os.path.join(opt.results_dir, opt.name)
input_dir = os.path.join(opt.input_dir, opt.name)
nbbs = NBBs.sparse_semantic_correspondence(vgg19, opt.gpu_ids, opt.tau, opt.border_size, save_dir, opt.k_per_level, opt.k_final, opt.fast)
image_count,source_index,target_index = util.arrangeSourceDir(input_dir,opt.sourceImg,opt.targetImg)
# -----create new matrix for the ordering of the pics ----------------
matrix = ORDER.create_matrix(image_count)
points_matrix = util.create_points_matrix(image_count)
#---------------------------------------------------------------------
data_file_path = save_dir +'/distance_data.dat'
point_file_path = save_dir +'/points_matrix_data.dat'
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if ((opt.contiue_calc) | (opt.data_stored)):
        print ("LOADED DATA:")
        # default dtype for  np.loadtxt is also floating point, change it, to be able to load mixed data.
        matrix = np.loadtxt(data_file_path, dtype=np.object)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] = float(matrix[i][j])
        print(matrix)

        if(opt.data_stored):
            source_index= opt.src_index
            target_index= opt.trg_index

    elif (opt.data_stored == False):
        exit

    if  (opt.data_stored == False):
        for img_index in range(image_count):
            # calculate from img_index image to all others
            img_path = util.get_file_path_from_index(opt,img_index)
            print("IMAGE_PATH: ",img_path)
            opt.sourceImg = img_path
            matrix,points_matrix = ORDER.image_distance_from_all_others(img_index,matrix,opt, nbbs,points_matrix,data_file_path,point_file_path)
            print(matrix)
            util.print_separator_line()
            matrix = np.maximum( matrix, matrix.transpose() )  # validate the matrix is symetric semetric \
            np.savetxt(data_file_path, matrix, fmt='%s')
            np.savetxt(point_file_path, points_matrix, fmt='%s')
            #END OF FOR LOOP
    util.print_separator_line()
    util.print_separator_line()

    print("Source_index = ",source_index, "| Target_index = ", target_index)
    print("Direct distance from Source to Target is: ", min(matrix[source_index][target_index] , matrix[target_index][source_index] ))
    our_path ,our_path_sum,edge_weight_list = ORDER.calculate_route_from_source_to_target(matrix,source_index,target_index)
    dijsktra_path = ORDER.run_dijsktra(matrix,source_index,target_index)
    util.print_separator_line()
    util.print_separator_line()


    print("CREATING PATH FILES")    
    with open('morph_path.txt', 'w') as f:
        string = str(our_path).strip('[] ').replace(" ", "")
        print(string)
        f.write(string)
    with open('morph_dijkstra.txt', 'w') as f:
        string = str(dijsktra_path).strip('[] ').replace(" ", "")
        print(string)
        f.write(string)
        
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i//10 == j//10: #images from the same domain
                value = matrix[i][j]/2
                matrix[i][j] = value

    print("After manipulate images from the same domain:")            
    print("Direct distance from Source to Target is: ", min(matrix[source_index][target_index] , matrix[target_index][source_index] ))
    our_path2,our_path_sum2,edge_weight_list2 = ORDER.calculate_route_from_source_to_target(matrix,source_index,target_index)
    dijsktra_path2 = ORDER.run_dijsktra(matrix,source_index,target_index)

    print("CREATING PATH FILE--DOMAIN-MANIPULATION")    
    with open('morph_path_domain_manipulation.txt', 'w') as f:
        string = str(our_path2).strip('[] ').replace(" ", "")
        print(string)
        f.write(string)
    with open('morph_dijkstra_domain_manipulation.txt', 'w') as f:
        string = str(dijsktra_path2).strip('[] ').replace(" ", "")
        print(string)
        f.write(string)

    with open('information.txt', 'w') as f:
        f.write("our_order:%s\n" % our_path)
        avg = our_path_sum/(len(our_path)-1)    
        f.write("Avg. transiton distance: %s\n" % avg)
        f.write("Median. transiton distance: %s\n" % median(edge_weight_list))
        f.write("dijsktra_order:%s\n" % dijsktra_path)
        f.write("____________________AFTER MANPULATION______________________________\n")
        f.write("our_order2:%s\n" % our_path2)
        avg = our_path_sum2/(len(our_path2)-1)    
        f.write("Avg. transiton distance: %s\n" % avg)
        f.write("Median. transiton distance: %s\n" % median(edge_weight_list2))
        f.write("dijsktra2_order: %s\n" % dijsktra_path2)