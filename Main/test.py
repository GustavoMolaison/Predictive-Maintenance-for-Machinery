import numpy as np
# array1 = np.array([[1,1,1],
#                    [2,2,2],
#                    [3,3,3]])
# print(array1)

# array1 = np.delete(array1, 0, axis = 0)
# print(f'new array{array1}')
# ys = np.delete(ys, 0, axis = 0)

array = np.ones([1, 3, 5, 10])

array2 = np.ones([15, 10])

array2 = array2.reshape([1, 3, 5, 10])

array = np.vstack((array, array2))
print(array)
print('XDDDDDDDDDDDDDDD')
array = array.reshape([3,5,10])

# array3 = np.ones([1, 1, 5, 10])

# array3 = np.vstack((array3, array))
# print('XDDDDDDDDDDDDDDDDDDDDDDDDDD')
# print(array3)



# array = np.ones([1, 10, 148])

# array2 = np.ones([10, 148])

# array2 = array2.reshape([1, 10, 148])

# array = np.vstack((array, array2))

# print(array.shape)