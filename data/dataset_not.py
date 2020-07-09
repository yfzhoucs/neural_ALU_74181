from torch.utils.data import Dataset
import random
import torch


class DatasetNotGate(Dataset):
	def __init__(self, num_samples):
		self.num_samples = num_samples

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		x_int = torch.randint(0, 2, (1,))
		x = x_int.float() * 0.9 + torch.rand((1,)) * 0.1
		
		y = torch.zeros(1)
		y[0] = 0 if x_int.item() == 1 else 1

		return x, y


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	dataset = DatasetNotGate(20)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	for x in dataloader:
		print(x)
