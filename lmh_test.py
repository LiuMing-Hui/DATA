from __future__ import print_function, absolute_import
import collections
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


# dates = pd.date_range('20200315', periods = 5)
# df = pd.DataFrame(np.arange(20).reshape((5,4)), index = dates, columns = ['A','B','C','D'])
# print(df.C)
# df.B[df.C==2]='True'
#
# print('\n')
# print(df)

# a = torch.tensor([1,2,3])
# b = len(a)
# print(b)

# i = max(1,2,3,4,0)
# print(i)

# pred = torch.LongTensor([[1,2,3,4,5,6], [1,2,3,4,5,6]])
# pred = pred.reshape(1, -1).squeeze()
#
# i = torch.LongTensor([1,2,])
# a = pred[i]
# b = a


# pseudo_labels = torch.range(int(0), int(7), step=1)
# print()


# y_true = torch.randint(low=9, high=20, size=(3,1)).squeeze(1)
# y_pred = torch.randint(low=9, high=20, size=(3,1)).squeeze(1)
#
# d = max(y_pred.max(), y_true.max()) + 1
# w = np.zeros((d, d), dtype=np.int64)
#
# for i in range(y_pred.size(0)):
#     w[y_pred[i], y_true[i]] += 1
#
# row_ind, col_ind = linear_sum_assignment(w.max() - w)
# r = w[row_ind, col_ind].sum() / y_pred.size(0)
# print(r)

a = torch.LongTensor([1,2,3,4,5,6])
b = torch.LongTensor([2,2,3,4,5,6])
c = torch.LongTensor([3,2,3,4,5,6])

for y, z in enumerate(zip(a, b, c)):
    print(y)
    print(z[1])