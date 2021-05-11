import torch


z = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])


print(z)
print(z.shape)
z = z.unsqueeze(0)
print(z)
print(z.shape)


z = z.repeat(1, 1, 1)


print(z)
print(z.shape)
