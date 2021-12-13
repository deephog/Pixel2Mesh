import pywt
import numpy as np

sample = np.zeros((8, 1, 256, 256))
print(sample.shape)
coeff1 = pywt.dwt2(sample, 'db1')
coeff2 = pywt.dwt2(coeff1[0], 'db1')

stack2 = np.concatenate([np.concatenate(coeff2[1], axis=1), coeff2[0]], axis=1)
print(stack2.shape)
print(len(coeff2))