import numpy as np
# array1 = np.array([[1,1,1],
#                    [2,2,2],
#                    [3,3,3]])
# print(array1)

# array1 = np.delete(array1, 0, axis = 0)
# print(f'new array{array1}')
# ys = np.delete(ys, 0, axis = 0)

array = np.ones([1, 5, 10])
array = np.vstack((array, np.ones([1, 5, 10]))) 
print(array)