# coding: utf-8

import numpy as np
import cv2
import random
import time
import image
import music

start_whole_time = time.time()
sr = 44100 # sampling rate

img = cv2.imread("/Users/koichi-sakaguchi/prv/private/chaofan.jpg")
img_name = "picasso"
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

partition_num_width = 10
partition_num_height = 10
last_path_order = 100
first_shift = 1
repeatable_num = 3
is_loop = 0

if is_loop == 0 :
    mode_name = "mldy"
elif is_loop == 1 :
    mode_name = "loop"

log_id, reduced_rgb, reduced_hsv = image.img2path(img, partition_num_width, partition_num_height, last_path_order, first_shift, repeatable_num)


print(log_id)

img_size = 400 # pixel
reduced_color_img = image.mk_reduced_color_img(img_size, partition_num_width, partition_num_height, log_id, reduced_rgb, mode_name)

path_h = image.path_array(log_id, reduced_hsv[:,:,0])
path_s = image.path_array(log_id, reduced_hsv[:,:,1])
path_v = image.path_array(log_id, reduced_hsv[:,:,2])

path_normalized_h = image.normalize1d(path_h)
path_normalized_s = image.normalize1d(path_s)
path_normalized_v = image.normalize1d(path_v)
path_normalized_hsv = [path_normalized_h, path_normalized_s, path_normalized_v]

is_cut_loop = 0
cut_loop_num = 1
keyname = "F_mj"
first_octave = 3 # ex: 0=[C0, C#0, ..., C1] for "all"
plus_octave = 1
key_id_shift = 0 # ex: 0=C=ド for "all"

first_time_ratio = 8
time_length_ratio = 12
isnt_degeneratable = 1
bpm = 150
loop_count = 8
len_id_usage = len(log_id)

bps = bpm/60
time_1count    = 1/bps
time_element   = time_1count/2
time_1loop       = loop_count*time_1count

unified_sound_wave, path_freq, path_first_time, path_time_length = music.path2wave(path_normalized_hsv, keyname, first_octave, plus_octave, key_id_shift, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, loop_count, len_id_usage, sr)
if is_cut_loop == 1:
    len_1loop = int(cut_loop_num*time_1loop*sr)
    unified_sound_wave = unified_sound_wave[0:len_1loop]
    path_freq                   = path_freq[0:len_1loop]
    path_first_time         = path_first_time[0:len_1loop]
    path_time_length     = path_time_length[0:len_1loop]

raw_unified_sound_wave = unified_sound_wave

whole_time_length = len(unified_sound_wave)/sr

if is_loop == 1:
    unified_sound_wave = np.array(int(len(unified_sound_wave_melody)/len(raw_unified_sound_wave) + 1)*list(raw_unified_sound_wave))
    unified_sound_wave_loop = unified_sound_wave    

max_amp = max(abs(unified_sound_wave))
unified_sound_wave = unified_sound_wave/max_amp
music.wavwrite(unified_sound_wave,"music_w"+str(partition_num_width)+"h"+str(partition_num_height)+".wav")

img_prm = [partition_num_width, partition_num_height, last_path_order, first_shift, repeatable_num]
msc_prm = [is_cut_loop, cut_loop_num, is_loop, keyname, first_octave, plus_octave, key_id_shift, loop_count, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, len_id_usage]
prm_mldy = mode_name+'_' + '-'.join(list(map(str, img_prm))) + '_' + '-'.join(list(map(str, msc_prm)))
prm_mldy

log_id_melody = log_id
reduced_rgb_melody = reduced_rgb
reduced_hsv_melody = reduced_hsv

unified_sound_wave_melody = raw_unified_sound_wave
partition_num_width_melody = partition_num_width
partition_num_height_melody = partition_num_height

partition_num_width = 3
partition_num_height = 3
last_path_order = 100
first_shift = 1
repeatable_num = 3
is_loop = 1

if is_loop == 0 :
    mode_name = "mldy"
elif is_loop == 1 :
    mode_name = "loop"

log_id, reduced_rgb, reduced_hsv = image.img2path(img, partition_num_width, partition_num_height, last_path_order, first_shift, repeatable_num)

img_size = 400 # pixel
reduced_color_img = image.mk_reduced_color_img(img_size, partition_num_width, partition_num_height, log_id, reduced_rgb, mode_name)

path_h = image.path_array(log_id, reduced_hsv[:,:,0])
path_s = image.path_array(log_id, reduced_hsv[:,:,1])
path_v = image.path_array(log_id, reduced_hsv[:,:,2])

path_normalized_h = image.normalize1d(path_h)
path_normalized_s = image.normalize1d(path_s)
path_normalized_v = image.normalize1d(path_v)
path_normalized_hsv = [path_normalized_h, path_normalized_s, path_normalized_v]

is_cut_loop = 1
cut_loop_num = 1
keyname = "F_mj"
first_octave = 2 # ex: 0=[C0, C#0, ..., C1] for "all"
plus_octave = 1
key_id_shift = 0 # ex: 0=C=ド for "all"
loop_count = 4

first_time_ratio = 3
time_length_ratio = 12
isnt_degeneratable = 0
bpm = 150
len_id_usage = len(log_id)

bps = bpm/60
time_1count    = 1/bps
time_element   = time_1count/2
time_1loop       = loop_count*time_1count

unified_sound_wave, path_freq, path_first_time, path_time_length = music.path2wave(path_normalized_hsv, keyname, first_octave, plus_octave, key_id_shift, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, loop_count, len_id_usage, sr)
whole_time_length = len(unified_sound_wave)/sr

if is_cut_loop == 1:
    len_1loop = int(cut_loop_num*time_1loop*sr)
    unified_sound_wave = unified_sound_wave[0:len_1loop]
    path_freq                   = path_freq[0:len_1loop]
    path_first_time         = path_first_time[0:len_1loop]
    path_time_length     = path_time_length[0:len_1loop]

raw_unified_sound_wave = unified_sound_wave

whole_time_length = len(unified_sound_wave)/sr

if is_loop == 1:
    looped_unified_sound_wave = np.array(int(len(unified_sound_wave_melody)/len(raw_unified_sound_wave) + 1)*list(raw_unified_sound_wave))
    unified_sound_wave = looped_unified_sound_wave

max_amp = max(abs(unified_sound_wave))
print(max_amp)
unified_sound_wave = unified_sound_wave/max_amp
music.wavwrite(unified_sound_wave,"music_w"+str(partition_num_width)+"h"+str(partition_num_height)+".wav")

img_prm = [partition_num_width, partition_num_height, last_path_order, first_shift, repeatable_num]
msc_prm = [is_cut_loop, cut_loop_num, is_loop, keyname, first_octave, plus_octave, key_id_shift, loop_count, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, len_id_usage]
prm_loop = mode_name+'_' + '-'.join(list(map(str, img_prm))) + '_' + '-'.join(list(map(str, msc_prm)))
prm_loop

log_id_loop = log_id
reduced_rgb_loop = reduced_rgb
reduced_hsv_loop = reduced_hsv

raw_unified_sound_wave_loop = raw_unified_sound_wave
unified_sound_wave_loop = looped_unified_sound_wave
partition_num_width_loop = partition_num_width
partition_num_height_loop = partition_num_height

unified_sound_wave = music.unify1d(1*unified_sound_wave_melody, 1*unified_sound_wave_loop)

max_amp = max(abs(unified_sound_wave))
print(max_amp)
unified_sound_wave = unified_sound_wave/max_amp
music.wavwrite(unified_sound_wave,"music_"+prm_mldy+"_"+prm_loop+"_"+img_name+".wav")

end_whole_time = time.time()
elapsed_whole_time = end_whole_time - start_whole_time
print(elapsed_whole_time, "sec   ", elapsed_whole_time/60, "min   ", elapsed_whole_time/3600, "hour")

