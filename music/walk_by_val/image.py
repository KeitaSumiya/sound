import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import chain, zip_longest
import random
import time
import math
import wave
import struct
import random

# # image
def partition1d(lst, partition_num, is_length):
    if(is_length == 1):
        partition_len = partition_num
        partitioned_lst = list(zip(*[iter(lst)]*partition_len))

        rest_num = len(lst) % partition_len
        if( rest_num > 0 ):
            rest_lst = lst[(len(lst) - rest_num):len(lst)]
            partitioned_lst.append(rest_lst)

        for i in range( len(partitioned_lst) ):
            partitioned_lst[i] = list( partitioned_lst[i] )

    if(is_length == 0):
        group_num = partition_num
        partition_len = int( len(lst) / group_num )
        longer_lst_num = len(lst) % group_num
        
        if longer_lst_num == 0 :
            partitioned_lst = partition1d(lst, partition_len, 1)
        elif longer_lst_num > 0 :
            partitioned_former_lst = partition1d(lst, partition_len+1, 1)[0:longer_lst_num]
            rest_lst = lst[ (partition_len+1)*longer_lst_num : len(lst) ]
            partitioned_latter_lst = partition1d(rest_lst, partition_len, 1)
            partitioned_lst = partitioned_former_lst + partitioned_latter_lst[0:len(partitioned_latter_lst)]

    return(partitioned_lst)

def partition2d(array, row_grid_num, column_grid_num):
    all_column_num, all_row_num = array.shape[0:2]
    column_id_lst =  partition1d(range(all_column_num), column_grid_num, 0)
    row_id_lst =  partition1d(range(all_row_num), row_grid_num, 0)
    
    partitioned_array = []
    for column in range(column_grid_num):
        a_column = []
        for row in range(row_grid_num):
            a_column.append( array[column_id_lst[column][0]:(column_id_lst[column][-1]+1), row_id_lst[row][0]:(row_id_lst[row][-1]+1) ] )
        partitioned_array.append(a_column)

    return(partitioned_array)


def to_mean(img):
    flattened_img = np.array(list(chain.from_iterable(np.float_(img))))
    mean_color =  sum(flattened_img)/len(flattened_img)
    return(mean_color)

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
    
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def make_all_candidate(present_id, shift):
    id_width, id_height = present_id
    
    candidates = [
        [id_width + shift, id_height],
        [id_width,             id_height + shift],
        [id_width - shift,  id_height],
        [id_width,             id_height - shift]
        ]
    return(candidates)

def dont_back(previous_id, candidates):
    available_candidates = []
    
    for i in range(len (candidates)):
        if(candidates[i] != previous_id):
            available_candidates.append(candidates[i])
            
    return(available_candidates)

def select_available(present_id, candidates, id_minmax):
    id_width, id_height = present_id
    id_width_min, id_width_max, id_height_min, id_height_max = id_minmax
    available_candidates = []
    
    for i in range(len (candidates)):
        candidate = candidates[i]

        if( (id_width_min <= candidate[0] <= id_width_max) and (id_height_min <= candidate[1] <= id_height_max)):
            available_candidates.append(candidate)

        elif( (candidate[0] > id_width_max) and (id_width < id_width_max) and (id_height_min <= candidate[1] <= id_height_max)):
            available_candidates.append([id_width_max, candidate[1]])

        elif( (candidate[0] < id_width_min) and (id_width > id_width_min) and (id_height_min <= candidate[1] <= id_height_max)):
            available_candidates.append([id_width_min, candidate[1]])

        elif( (candidate[1] > id_height_max) and (id_height < id_height_max) and (id_width_min <= candidate[0] <= id_width_max)):
            available_candidates.append([candidate[0], id_height_max])

        elif( (candidate[1] < id_height_min) and (id_height > id_height_min) and (id_width_min <= candidate[0] <= id_width_max)):
            available_candidates.append([candidate[0], id_height_min])
    
    return(available_candidates)


def decide_next(present_id, candidates, reduced_color, selection_mode):
    id_width, id_height = present_id
    
    id_and_gap = []
    dtype = [('candidate_id_width', 'int'), ('candidate_id_height', 'int'), ('gap', 'float')]
    for i in range(len(candidates)):
        candidate_id_width, candidate_id_height = candidates[i]
        id_and_gap.append(tuple([candidate_id_width, candidate_id_height, reduced_color[id_height, id_width] - reduced_color[candidate_id_height, candidate_id_width]]))
        
    id_and_gap = list( np.sort(np.array(id_and_gap, dtype=dtype), order='gap') )
    
    # tuple2list for elements
    for i in range(len(id_and_gap)):
        id_and_gap[i] = list(id_and_gap[i])

    if(selection_mode == "closest"):
        next_id_width  = id_and_gap[0][0]
        next_id_height = id_and_gap[0][1]
    elif(selection_mode == "most_different"):
        next_id_width  = id_and_gap[len(id_and_gap)-1][0]
        next_id_height = id_and_gap[len(id_and_gap)-1][1]

    next_id = [next_id_width, next_id_height]

    return(next_id)


def decide_shift(present_id, unique_past_id, shift_lst):
    id_width, id_height = present_id

    if( set([str(id_width)+","+str(id_height)]).issubset(unique_past_id) ):
        shift_lst[id_height][id_width] += 1

    shift = shift_lst[id_height][id_width]
        
    return([shift, shift_lst])

def jump_unknown(present_id, unique_past_id, unique_all_id, reduced_color, selection_mode):
    unique_rest_id = unique_all_id.difference(unique_past_id)
    rest_id = list(unique_rest_id)

    # str2int
    for i in range(len(rest_id)):
        rest_id[i] = rest_id[i].split(',')
        rest_id[i] = [int(rest_id[i][0]), int(rest_id[i][1])]

    next_id = decide_next(present_id, rest_id, reduced_color, selection_mode)
    return(next_id)

def mk_reduced_color_img(img_size, partition_num_width, partition_num_height, log_id, reduced_rgb, label_name):
    element_img_size = int(img_size/max([partition_num_height, partition_num_width])) # pixel
    print(element_img_size)
    reduced_color_img = np.zeros((element_img_size*partition_num_height, element_img_size*partition_num_width, 3), np.uint8)
    pointer_edge_px = int(element_img_size/3)

    for path_order in range(len(log_id)):
        id_width, id_height = log_id[path_order]

        img_pt1 = [element_img_size*id_width,                 element_img_size*id_height]
        img_pt2 = [element_img_size*(id_width + 1) - 1, element_img_size*(id_height + 1) - 1]
        pt_pt1   = [img_pt1[0] + pointer_edge_px,       img_pt1[1] + pointer_edge_px]
        pt_pt2   = [img_pt2[0] -  pointer_edge_px - 1, img_pt2[1] -  pointer_edge_px - 1]
        cv2.rectangle(reduced_color_img, (img_pt1[0], img_pt1[1]), (img_pt2[0], img_pt2[1]), reduced_rgb[id_height, id_width], -1, 0)
        cv2.rectangle(reduced_color_img, (pt_pt1[0],    pt_pt1[1]),   (pt_pt2[0],    pt_pt2[1]),   np.array([255.,255.,255.]),            -1, 0)
        cv2.imwrite("path/"+label_name+"_reduced_color_img_"+str(path_order)+".jpg", cv2.cvtColor(reduced_color_img,cv2.COLOR_RGB2BGR) )
        cv2.rectangle(reduced_color_img, (img_pt1[0], img_pt1[1]), (img_pt2[0], img_pt2[1]), reduced_rgb[id_height, id_width], -1, 0)

    return(reduced_color_img)

def normalize1d(array):
    array_min  = min(array)
    array_max = max(array)

    normalized_array = []
    for i in range(len(array)):
        normalized_array.append( (array[i] - array_min)/(array_max - array_min) )
    normalized_array = np.array(normalized_array)
    
    return(normalized_array)

def path_array(log_id, input_array):
    array = []
    for path_order in range(len(log_id)):
        id_width, id_height = log_id[path_order]
        array.append(input_array[id_height, id_width])
    array = np.array(array)
    return(array)

def mk_reduced_rgb(partitioned_img, partition_num_height, partition_num_width):
    reduced_rgb = []
    for id_height in range(partition_num_height):
        a_width_rgb = []
        for id_width in range(partition_num_width):
            a_width_rgb.append( to_mean(partitioned_img[id_height][id_width]) )
        reduced_rgb.append(a_width_rgb)
    reduced_rgb = np.array(reduced_rgb)

    return(reduced_rgb)


def mk_reduced_hsv(partitioned_img, partition_num_height, partition_num_width):
    reduced_hsv = []
    for id_height in range(partition_num_height):
        a_width_hsv = []
        for id_width in range(partition_num_width):
            mean_rgb = to_mean(partitioned_img[id_height][id_width])
            a_width_hsv.append( rgb2hsv(mean_rgb[0], mean_rgb[1], mean_rgb[2]) )
        reduced_hsv.append(a_width_hsv)
    reduced_hsv = np.array(reduced_hsv)

    return(reduced_hsv)


def img2path(img, partition_num_width, partition_num_height, last_path_order, first_shift, repeatable_num):

    first_id_width = int(partition_num_width/2)
    first_id_height = int(partition_num_height/2)

    unique_all_id = []
    for id_height in range(partition_num_height):
        for id_width in range(partition_num_width):
            unique_all_id.append(str(id_width)+","+str(id_height))
    unique_all_id = set(unique_all_id)

    partitioned_img = partition2d(img, partition_num_width, partition_num_height)

    id_minmax = [0, partition_num_width  - 1, 0, partition_num_height-  1]

    reduced_rgb = mk_reduced_rgb(partitioned_img, partition_num_height, partition_num_width)
    print(reduced_rgb.shape)

    reduced_hsv = mk_reduced_hsv(partitioned_img, partition_num_height, partition_num_width)
    print(reduced_hsv.shape)

    reduced_h = reduced_hsv[:,:,0]

    path_order = 0
    reduced_color = reduced_h
    unique_past_id = set([])

    while (len(unique_past_id) != len(unique_all_id)) and (path_order <= last_path_order) :
        if(path_order == 0):
            present_id = [first_id_width, first_id_height]
            log_id = [present_id]
            shift = first_shift
            shift_lst = []
            for i in range(partition_num_height):
                a_width_shift = []
                for j in range(partition_num_width):
                    a_width_shift.append(first_shift)
                shift_lst.append(a_width_shift)

        if(path_order >= 1):
            previous_id = present_id
            present_id = next_id
            log_id.append(present_id)
            shift, shift_lst = decide_shift(present_id, unique_past_id, shift_lst)

        candidates = make_all_candidate(present_id, shift)

        if(path_order >= 1):
            candidates = dont_back(previous_id, candidates)

        candidates = select_available(present_id, candidates, id_minmax)

        next_id = decide_next(present_id, candidates, reduced_color, "closest")

        next_id_width, next_id_height = next_id
        repeat_num = shift_lst[next_id_height][next_id_width]
        if( repeat_num > repeatable_num ):
            next_id = jump_unknown(present_id, unique_past_id, unique_all_id, reduced_h, "most_different")

        unique_past_id.add( str(present_id[0])+","+str(present_id[1]) )

        path_order += 1
        
        
    return(log_id, reduced_rgb, reduced_hsv)
