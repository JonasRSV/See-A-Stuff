import os
import sys
import numpy as np
import cv2
import see_a_thing.utils.files as files

def preprocess_image(image):
    #####################
    # Preprocess images #
    #####################
    reduce_size = cv2.resize(image, (299, 299))
    
    ####################################################
    # Training is faster with numerical scales -1 -> 1 #
    # rather than 0 -> 256                             #
    ####################################################

    rescale_channels = (reduce_size - 128) / 128

    return rescale_channels


def unnumpyfy(x):
    if isinstance(x, list):
        return list(map(unnumpyfy, x))

    if    isinstance(x, np.float16) 
       or isinstance(x, np.float32)
       or isinstance(x, np.float64):
        return float(x)

    if    isinstance(x, np.int16)
       or isinstance(x, np.int32)
       or isinstance(x, np.int64):
        return int(x)

    return x


def decode_bytes(b):
    return list(map(lambda x: x.decode("utf-8"), b))


# TODO: Add more prequisite checks if need be
def check_prequisites(settings):
    files.check_file_prequisites(settings)


# TODO: Add more fixes if need be
def fix_prequisites(settings):
    files.fix_file_prequisites(settings)

    print("\nPrequisites Fixed! :)")



def print_logo():
   print('                                                                                                                                ')
   print('\033[92m' + '         _______. _______  _______           ___           .___________. __    __   __  .__   __.   _______     __       ___      ' + '\033[0m')
   print('\033[92m' + '        /       ||   ____||   ____|         /   \          |           ||  |  |  | |  | |  \ |  |  /  _____|   /_ |     / _ \   ' + '\033[0m')
   print('\033[92m' + '       |   (----`|  |__   |  |__    ______ /  ^  \   ______`---|  |----`|  |__|  | |  | |   \|  | |  |  __      | |    | | | |  ' + '\033[0m')
   print('\033[92m' + '        \   \    |   __|  |   __|  |______/  /_\  \ |______|   |  |     |   __   | |  | |  . `  | |  | |_ |     | |    | | | |  ' + '\033[0m')
   print('\033[92m' + '    .----)   |   |  |____ |  |____       /  _____  \           |  |     |  |  |  | |  | |  |\   | |  |__| |     | |  __| |_| |  ' + '\033[0m')
   print('\033[92m' + '    |_______/    |_______||_______|     /__/     \__\          |__|     |__|  |__| |__| |__| \__|  \______|     |_| (__)\___/   ' + '\033[0m')
   print( '                                                                                                                                ')




