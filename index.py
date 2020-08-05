import numpy as np

a = list(np.array([1,2,3,4]))
b = np.array([5,6,7])
c = []

c.extend(a)
# c.extend(b)

print(c)