import numpy as np
from itertools import chain, zip_longest
import random
import time
import math
import wave
import struct
import random

def makewave0(amp,freq,sec,sr):
    outwave = []
    for i in range( int(sec*sr) ) :
        w = amp*np.sin(2*np.pi*freq*i/sr)
        outwave.append(w)
    return(outwave)


def wavwrite(inputlist,filename):
    maxamp = 32767.0
    int16wave = [int(x * maxamp) for x in inputlist]
    binwave = struct.pack("h"*len(int16wave),*int16wave)

    nchannnles = 1 # 1=monoral , 2=stereo
    sampwitdth = 2 # 1=8bit,2=16bit,3=,...
    framerate = 44100 # sampling rate ex.44100Hz
    nframes = len(inputlist) # framerate * sec
    
    of = wave.open(filename,"w")
    of.setparams((nchannnles,sampwitdth,framerate,nframes,"NONE","not compressed"))
    of.writeframes(binwave)
    of.close
    
def ampFn(amp_type,t,t0,T,amp,aparam):
    # t0: initial time
    # T: length of time for sound
    # t1: time of peak
    # a1: trancation exponent(a1=1: NoEffect, 1<<a1: too tracate)
    # a2: trancation exponent(0<a2, 0<a2<<1: too tracate, 1<<a2: NoEffect
    # a1=-1&a2=1: convex trancation shape
    t1 = aparam[0]
    a1 = aparam[1]
    a2 = aparam[2]
    if amp_type == 1:
        if t <= t0:
            ampVal = 0
        else:
            if t <= t0 + t1:
                ampVal = amp*(t-t0)/t1
            else:
                ampVal = amp*(-(t-T)/T)*((t1+a2)/(t-t0+a2))**a1
    elif amp_type == 2:
        if t <= t0:
            ampVal = 0
        else:
            ampVal = amp*(1-(t-t0)/T)**2
    elif amp_type == 3:
        if t <= t0:
            ampVal = 0
        else:
            ampVal = amp*(1-((t-t0)/T)**(1/10))

    if ampVal > 5:
        ampVal = 5

    return(ampVal)


def waveFn(wave_type,t,t1,param,freq):
    #Sin Wave
    if   wave_type == 1:
        wave = np.sin(2*np.pi * freq*t)
    #Sawtooth Wave
    elif wave_type == 2:
        wave = 2*((t-t1)*freq - np.floor((t-t1)*freq + 0.5))
    #Triangle Wave
    elif wave_type == 3:
        wave = 2*abs(2*((t-t1+1/(4*freq))*freq - np.floor((t-t1+1/(4*freq))*freq + 0.5)))-1
    #Square Wave
    elif wave_type == 4:
        wave = np.sign(np.sin(2*np.pi * freq*t))
    #Pulse Wave 1
    elif wave_type == 5:
        wave = np.sign(- (t-t1)*freq + np.floor((t-t1)*freq) + param[0])
    return(wave)

def makewave(amp, freq, T, sr, aparam):
    outwave = []
    dt = 1/sr
    t0 = 0
    for t in np.arange( 0., T+dt, dt ) :
        w = ampFn(2, t, t0, T, amp, aparam) * ( waveFn(1, t, 0, 1, freq) + waveFn(1, t, 0, 1, 2*freq)/4 + waveFn(1, t, 0, 1, 3*freq)/6 + waveFn(1, t, 0, 1, 4*freq)/8 )
        outwave.append(w)
    return(outwave)

def makewave2(amp, freq, T, sr, aparam):
    outwave = []
    dt = 1/sr
    t0 = 0
    period_freq = 1/freq
    wave_period_freq = []
    for t in np.arange( 0., period_freq+dt, dt ) :
        w = random.uniform(-1,1)
        wave_period_freq.append(w)
        
    base_wave = int(T*freq+1)*wave_period_freq

    i = 0
    for t in np.arange( 0., T+dt, dt ) :
        w = ampFn(2, t, t0, T, amp, aparam) * base_wave[i]
        outwave.append(w)
        i += 1
    
    return(outwave)

def set_key(keyname):
    if(keyname == "all"):
        base_key_id = list(range(12))
    elif(keyname == "white"):
        base_key_id = [0,2,4,5,7,9,11]
    elif(keyname == "black"):
        base_key_id = [  1,3,   6, 8, 10]
    elif(keyname[len(keyname)-2:len(keyname)] == "mj"):
        base_major_key_id = [0,2,4,5,7,9,11]
        if(keyname ==    "Cf_mj"):
            base_key_id = list(np.array(base_major_key_id) - 1)
        elif(keyname == "C_mj"):
            base_key_id = list(np.array(base_major_key_id) + 0)
        elif(keyname == "Cs_mj"):
            base_key_id = list(np.array(base_major_key_id) + 1)
        elif(keyname == "Df_mj"):
            base_key_id = list(np.array(base_major_key_id) + 1)
        elif(keyname == "D_mj"):
            base_key_id = list(np.array(base_major_key_id) + 2)
        elif(keyname == "Eb_mj"):
            base_key_id = list(np.array(base_major_key_id) + 3)
        elif(keyname == "E_mj"):
            base_key_id = list(np.array(base_major_key_id) + 4)
        elif(keyname == "F_mj"):
            base_key_id = list(np.array(base_major_key_id) + 5)
        elif(keyname == "Fs_mj"):
            base_key_id = list(np.array(base_major_key_id) + 6)
        elif(keyname == "Gf_mj"):
            base_key_id = list(np.array(base_major_key_id) + 6)
        elif(keyname == "G_mj"):
            base_key_id = list(np.array(base_major_key_id) + 7)
        elif(keyname == "Af_mj"):
            base_key_id = list(np.array(base_major_key_id) + 8)
        elif(keyname == "A_mj"):
            base_key_id = list(np.array(base_major_key_id) + 9)
        elif(keyname == "Bf_mj"):
            base_key_id = list(np.array(base_major_key_id) +10)
        elif(keyname == "B_mj"):
            base_key_id = list(np.array(base_major_key_id) + 11)
    elif(keyname[len(keyname)-2:len(keyname)] == "mn"):
        base_major_key_id = [0,2,3,5,7,8,10,12]
        if(keyname ==    "C_mn"):
            base_key_id = list(np.array(base_major_key_id) + 0)
        elif(keyname == "Cs_mn"):
            base_key_id = list(np.array(base_major_key_id) + 1)
        elif(keyname == "D_mn"):
            base_key_id = list(np.array(base_major_key_id) + 2)
        elif(keyname == "Ds_mn"):
            base_key_id = list(np.array(base_major_key_id) + 3)
        elif(keyname == "Ef_mn"):
            base_key_id = list(np.array(base_major_key_id) + 3)
        elif(keyname == "E_mn"):
            base_key_id = list(np.array(base_major_key_id) + 4)
        elif(keyname == "F_mn"):
            base_key_id = list(np.array(base_major_key_id) + 5)
        elif(keyname == "Fs_mn"):
            base_key_id = list(np.array(base_major_key_id) + 6)
        elif(keyname == "G_mn"):
            base_key_id = list(np.array(base_major_key_id) + 7)
        elif(keyname == "Gs_mn"):
            base_key_id = list(np.array(base_major_key_id) + 8)
        elif(keyname == "Af_mn"):
            base_key_id = list(np.array(base_major_key_id) + 8)
        elif(keyname == "A_mn"):
            base_key_id = list(np.array(base_major_key_id) + 9)
        elif(keyname == "As_mn"):
            base_key_id = list(np.array(base_major_key_id) + 10)
        elif(keyname == "Bf_mn"):
            base_key_id = list(np.array(base_major_key_id) + 10)
        elif(keyname == "B_mn"):
            base_key_id = list(np.array(base_major_key_id) + 11)
        
    octave_num = len(base_key_id)
    
    all_key_id = []
    for i in range(0, 10):
        all_key_id.append( np.array(base_key_id)+12*i )
    all_key_id = list( np.concatenate(all_key_id) )
    
    return(all_key_id, octave_num)


# ex) "white", freq = value2freq(3, 1, 0, all_key_id, octave_num, 0.01) >> 130.8(C3)
# ex) "white", freq = value2freq(3, 1, 0, all_key_id, octave_num, 0.2)   >> 146.8(D3)
def value2freq(first_octave, plus_octave, key_id_shift, all_key_id, octave_num, normalized_value):
    first_key_id = octave_num*first_octave + key_id_shift
    last_key_id = octave_num*(first_octave + plus_octave) + key_id_shift
    key_id_gap = last_key_id - first_key_id

    key_id = all_key_id[ int( first_key_id + key_id_gap * normalized_value ) ]
    freq = 16.35159783 * ( (math.pow(2.0, key_id))**(1.0/12.0) )

    return(freq)

def unify1d(lst0, lst1):
    len0 = len(lst0)
    len1 = len(lst1)
    aft0 = max([len0, len1])-len0
    aft1 = max([len0, len1])-len1
    unified_lst = np.lib.pad(lst0,(0, aft0),"constant",constant_values = 0) + np.lib.pad(lst1,(0, aft1),"constant",constant_values = 0)
    return(unified_lst)


def value2first_time(orbit_first_time, isnt_degeneratable, time_element, first_time_ratio, normalized_value):
    if(len(orbit_first_time) == 0):
        first_time = 0
    elif(len(orbit_first_time) > 0):
        previous_first_time = orbit_first_time[-1]
        first_time = (previous_first_time + isnt_degeneratable*time_element) + int(first_time_ratio*(1 - normalized_value))*time_element

    return(first_time)

def value2time_length(orbit_order, len_id_usage, time_element, time_1loop, time_length_ratio, first_time, normalized_value, sr):
    time_length = time_element + int(time_length_ratio*(1 - normalized_value))*time_element
    last_time = first_time + time_length
    is_over = int(last_time / time_1loop) - int(first_time / time_1loop)
    over_time = last_time % time_1loop
    if( is_over > 0 ):
        time_length = int(last_time / time_1loop)*time_1loop - first_time
        last_time = int(last_time / time_1loop)*time_1loop
        
    aft = 0
    if orbit_order == len_id_usage - 1:
        aft = int( ( (int(last_time / time_1loop) + 1)*time_1loop - last_time )*sr)
        
    return(time_length, last_time, aft)

def orbit2wave(orbit_normalized_hsv, keyname, first_octave, plus_octave, key_id_shift, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, loop_count, len_id_usage, sr):
    orbit_normalized_h, orbit_normalized_s, orbit_normalized_v = orbit_normalized_hsv
    
    all_key_id, octave_num = set_key(keyname)
    bps = bpm/60
    time_1count    = 1/bps
    time_element   = time_1count/2
    time_1loop       = loop_count*time_1count
    time_half_loop = time_1loop/2

    orbit_freq = []
    orbit_first_time = []
    orbit_time_length = []
    for orbit_order in range(len_id_usage):
        nml_h = orbit_normalized_h[orbit_order]
        nml_s = orbit_normalized_s[orbit_order]
        nml_v = orbit_normalized_v[orbit_order]
#        print(nml_h, nml_s, nml_v)

        freq = value2freq(first_octave, plus_octave, key_id_shift, all_key_id, octave_num, nml_h)
        orbit_freq.append(freq)

        first_time = value2first_time(orbit_first_time, isnt_degeneratable, time_element, first_time_ratio, nml_s)
        orbit_first_time.append(first_time)

        time_length, last_time, aft = value2time_length(orbit_order, len_id_usage, time_element, time_1loop, time_length_ratio, first_time, nml_v, sr)
        orbit_time_length.append(time_length)

        aparam = [0.02*time_length*nml_v, 1, 0.1*nml_v] #t1,a1,a2
        bfr = int(first_time*sr)
        sound_wave = makewave(1, freq, time_length, sr, aparam)
        sound_wave = np.lib.pad(sound_wave, (bfr, aft), "constant", constant_values = 0)
        if(orbit_order == 0):
            unified_sound_wave = sound_wave
        elif(orbit_order > 0):
            unified_sound_wave = unify1d(sound_wave, unified_sound_wave)

    return(unified_sound_wave, orbit_freq, orbit_first_time, orbit_time_length)




def orbit_hsv2wave(orbit_normalized_hsv, msc_prm):
    keyname, first_octave, plus_octave, key_id_shift, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, loop_count, sr, mode_name, is_cut_loop, cut_loop_num, whole_wave_length = msc_prm

    len_id_usage = np.array(orbit_normalized_hsv).shape[1]
    bps = bpm/60
    time_1count    = 1/bps
    time_element   = time_1count/2
    time_1loop       = loop_count*time_1count

    unified_sound_wave, orbit_freq, orbit_first_time, orbit_time_length = orbit2wave(orbit_normalized_hsv, keyname, first_octave, plus_octave, key_id_shift, first_time_ratio, time_length_ratio, isnt_degeneratable, bpm, loop_count, len_id_usage, sr)
    if is_cut_loop == 1:
        len_1loop = int(cut_loop_num*time_1loop*sr)
        unified_sound_wave = unified_sound_wave[0:len_1loop]
        orbit_freq = orbit_freq[0:len_1loop]
        orbit_first_time = orbit_first_time[0:len_1loop]
        orbit_time_length = orbit_time_length[0:len_1loop]


#    print(orbit_freq)
#    print(orbit_first_time)
#    print(orbit_time_length)

    raw_unified_sound_wave = unified_sound_wave

    whole_time_length = len(unified_sound_wave)/sr

    if mode_name[0:4] == "loop":
        unified_sound_wave = np.array(int(whole_wave_length/len(raw_unified_sound_wave) + 1)*list(raw_unified_sound_wave))
        unified_sound_wave_loop = unified_sound_wave    

    max_amp = max(abs(unified_sound_wave))
    unified_sound_wave = unified_sound_wave/max_amp
    wavwrite(unified_sound_wave,"music_"+mode_name+".wav")

    return(unified_sound_wave)



