import os
import sys
from scapy.all import rdpcap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from hilbertcurve.hilbertcurve import HilbertCurve

SIZE = 4

def hilbert_mapping(byte_array, image_size=SIZE):
    pad_len = max(0, image_size * image_size - len(byte_array))
    padded = np.pad(byte_array, (0, pad_len), 'constant')

    data = padded[:image_size * image_size]
    mat = np.zeros((image_size, image_size), dtype=np.uint8)

    p = int(np.log2(image_size))  # image_size = 2^p
    hilbert_curve = HilbertCurve(p, 2)
    for i in range(image_size * image_size):
        x, y = hilbert_curve.point_from_distance(i)
        mat[y][x] = data[i]  # y,x because PIL uses row,col
    return mat


array = np.array([i for i in range(SIZE*SIZE)])

mat = hilbert_mapping(array, image_size=SIZE)

for i in range(SIZE):
    for j in range(SIZE):
        print(mat[i][j], end=' ')
    print()