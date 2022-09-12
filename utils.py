"""
@purpose : Contains argument parser and sets the values for hyperparams to default
@when : 09/01/22
NOTE : Please change the root folder path in 'data_path' before running train.py
"""

import argparse

raw_data = 'Propofol Dataset 1-4.csv' # 'Propofol Dataset Final.csv'  # 'Propofol_NONMEM_dataset_1_to_4_V01.csv' 
data_path = '/home/cepheus/DFKI/C3DI/code_17.5.22/PHARMOPRO-KI/PHARMOPRO-KI/code/new data/' # Provide the path to <code> folder here

# Argument parser
parser = argparse.ArgumentParser(description='train')

parser.add_argument('--data_path', type=str, default=data_path,
                    help='Location of data')
parser.add_argument('--model', type=str, default='rf', choices=['rf', 'nn', 'svm', 'gb'],
                    help='Choose the model for training (rf-random forest/nn-neural net/rfc-custom random forest)') 