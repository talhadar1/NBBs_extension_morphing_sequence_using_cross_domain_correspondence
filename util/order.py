#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import math
import torch
import numpy as np   
from numpy import median
# generate random integer values
from random import seed
from random import randint
from algorithms import image_distance  as IMG_DIST
# seed random number generator
# seed(0)
from . import util as UT

from collections import defaultdict

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


def create_matrix (pic_num):
   zeors_array = np.zeros( (pic_num, pic_num) )
   for j in range(pic_num): 
            for i in range(pic_num):
               zeors_array[j][i] = math.inf
               # zeors_array[j][i] = randint(0, 10);
   return zeors_array

def add_image_distances_from_set(source_index , distace_array, pic_num, distance_matrix):
   if pic_num != len(distace_array):
          print("pic_num(",pic_num,")", " != length of array(",len(distace_array),")")
   for j in range(pic_num):
         if j != source_index:
               if (math.isinf(distance_matrix[source_index][j])):
                  distance_matrix[source_index][j] = distace_array[j]

               if (math.isinf(distance_matrix[j][source_index])):
                  distance_matrix[j][source_index] = distace_array[j]
         else:
               distance_matrix[j][j] = 0
   return distance_matrix

def get_minimal_distance_from_source_to_target(distance_matrix,source_index,target_index,nodes):
   min_distance =  distance_matrix[source_index][target_index]
   src_tgt_dist = min_distance
   gateway = -1
   if(source_index > len(distance_matrix) | source_index  < 0 | target_index > len(distance_matrix) | target_index < 0 ):
         print("ERROR IN: get_minimal_distance_from_source_to_target()- index's incorrect!")
         return -1,-1,-1
   for j in range(len(distance_matrix)):
      if(j != source_index and j != target_index):
         if (j not in nodes):
            new_dist = (distance_matrix[source_index][j] + distance_matrix[j][target_index])/2
            if min_distance > new_dist:
               gateway = j;
               min_distance = new_dist
   if gateway != -1:
      #Add new gateway point, decide wich edge to keep.
      src_gtw_dist = distance_matrix[source_index][gateway]
      gtw_tgt_dist = distance_matrix[gateway][target_index]
      if src_gtw_dist <=  distance_matrix[gateway][target_index]:
            print("KEEP Source----> Gateway (",source_index,gateway,") = ",src_gtw_dist)
            print("min distacne is: ",min_distance)
            return 1,gateway,src_gtw_dist
      else:
            print("KEEP Gateway----> Traget (",gateway, target_index,") = ",gtw_tgt_dist)
            print("min distacne is: ",min_distance)
            return 2,gateway,gtw_tgt_dist
   else:
            print("KEEP Source----> Traget (",source_index, target_index,") = ", src_tgt_dist)
            print("min distacne is: ",min_distance)
   return 0,gateway,min_distance

def calculate_route_from_source_to_target(matrix,source,target):
   edge_weight_list = []
   nodes = [source,target]
   image_count = len(matrix)
   source_route = [source]
   target_route = [target]
   route_sum = 0
   res = -1
   gateway = -1
   edge_weight = -1
   for j in range(image_count): 
      res,gateway,edge_weight = get_minimal_distance_from_source_to_target(matrix,source,target,nodes)
      if res == 0: # the edge is between the source and the target directly
         route_sum += matrix[source][target]
         break
      elif res == 1: # the edge is Source----> Gateway
         source_route.extend([gateway])
         route_sum += matrix[source][gateway]
         source = gateway
         nodes.append(gateway)
      elif res == 2: # the edge is Gateway----> Traget
         target_route.extend([gateway])
         route_sum += matrix[gateway][target]
         target = gateway
         nodes.append(gateway)
      else:
         print("PROBLAM!:")
         print(matrix)
         print("source_route",source_route)
         print("target_route",target_route)
      print("get_minimal_distance_from_source_to_target:",source,"---->",target)
   print("NODES = ",nodes)

   edge_weight_list.append(edge_weight)
   print("---------------------------------------------------------------------")
   print("calculate_route_from_source_to_target finished with:")
   print("route_sum = ",route_sum)
   print("source_route: ",source_route)
   print("target_route: ",target_route) 
   total_route = get_total_route (source_route,target_route)
   print("route_from_source_to_target: ",total_route)
   print("Avg. transiton distance : ", route_sum/(len(total_route)-1))
   print("Median. transiton distance : ",median(edge_weight_list))
   return total_route , route_sum , edge_weight_list



def get_total_route (source_route,target_route):
   target_len = len(target_route)
   for j in range(target_len):
          source_route.append(target_route[target_len-j-1])
   return source_route


def build_graph (matrix,graph,source_index,target_index):
   image_count = len(matrix)
   for i in range(image_count):
         for j in  range(image_count):
                edge = (i,j,matrix[i][j])
                graph.add_edge(*edge)


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def run_dijsktra(matrix, source_index, target_index):
      graph = Graph()
      build_graph (matrix,graph,source_index,target_index)
      path = dijsktra(graph,source_index,target_index)
      print("Dijsktra Shortest Path:",path)
      return path


# def images_distance(source_index,target_index):
#    print("calac: dits <",source_index,",",target_index)

def image_distance_from_all_others(source_index,matrix,opt, nbbs,points_matrix,data_file_path,point_file_path):
   print("image_distance_from_all_others: SOURCE = ",source_index)
   source = source_index;
   for i in range(len(matrix)):
      if (math.isinf(matrix[source][i]) and i != source):
         opt.targetImg = UT.get_file_path_from_index(opt,i)
         # dist,points = images_distance(source_index,i)
         print("Calc_distance_from",source," to ", i)
         # print(matrix)
         try:
            dist,points = IMG_DIST.run_NBB_and_get_points(opt, nbbs)
         except IndexError as e:
            print("IndexError --- : ")
            dist = math.inf         ####NEED THIS## print(points)
         matrix[source][i] = dist
         matrix[i][source] = dist
         print("MATRIX[",source,",",i,"] = ",dist)
         # points_matrix[source_index-1][i] = points
         # points_matrix[i][source_index-1] = points
      elif (i == source):
         matrix[i][i] = 0
         print("ALLREADY CALCULATED --> MATRIX[",i,",",i,"] = ",matrix[i][i])
      else:
         print("MATRIX[",source,",",i,"] = ",matrix[source][i])
      np.savetxt(data_file_path, matrix, fmt='%s')
      np.savetxt(point_file_path, points_matrix, fmt='%s')
   return matrix,points_matrix

   def clean_loops():
      route = [1,2,3,4,2,5,6,7]
      i = j = 0
      for i in range(route):
         j=i+1
         while j<len(route):    
            if route[i]==route[j]:
               print ("route with no loops" ,route[i:j])
      return 0