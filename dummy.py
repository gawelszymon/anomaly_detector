import numpy as np
import matplotlib.pyplot as plt

def abs_sin(n):
    x = np.linspace(0, 20 * np.pi, n)
    d = np.abs(np.sin(x))
    
    d[12] -= 0.8
    d[444] += 0.7
    d[111] += 1.0
    
    return print(d)

abs_sin(500)