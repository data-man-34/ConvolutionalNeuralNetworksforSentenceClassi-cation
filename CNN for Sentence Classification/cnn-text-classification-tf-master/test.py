import numpy as np
import tensorflow as tf
# a = [1,2,3]
# b= [5 ,6,7]
#
# c = np.concatenate((a,b), 0)
#
# c = a + b
# print (c)
# print (type(a))
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# a = tf.concat([t1, t2], 0)
# b = tf.concat([t1, t2], 1)
# print (a)
# print (type(a))
# dict = {'Name': 'Zara', 'Age': 27}
#
# print ("Value : %s" %  dict.get('Age'))
# print ("Value : %s" %  dict.get('Sex'))

a = [[2,3,3],
     [4,2,2],
     [4,5,2]]

b = [[0,1],
     [1,0],
     [1,0]]
c = list(zip(a, b))
print (c)
sxl = np.array(c)
print (sxl)
print (sxl.shape)