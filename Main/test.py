import numpy as np
# xd = []
# xdd = []
# papa =(np.array([[2,2,2],
#                      [1,1,1]]))
# xdd.append(papa)
# patrz = xdd
# xd.append(patrz)
# xdd.clear()
# print(xd[0][0])
list1 = []
list2 = []
for g in range(10):
  list1 = []
  for i in range(10):
    num = i
    list1.append(num)
  list2.append(list1)
print(list2)    