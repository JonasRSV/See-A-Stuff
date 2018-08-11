import os
import sys
import numpy as np
import cv2
import see_a_thing.files as files

def preprocess_image(image):
    #####################
    # Preprocess images #
    #####################
    reduce_size = cv2.resize(image, (299, 299)) / 256

    return reduce_size

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




