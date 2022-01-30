import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from modules.LeNet5 import LeNet
from downloads import download_mnist_datasets_lenet
from evaluate_lenet import evaluate

BATCH_SIZE = 128
EPOCHS = 1200
LEARNING_RATE = 0.001
N_CLASSES = 10

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
	for inputs, targets in data_loader:
		inputs, targets = inputs.to(device), targets.to(device)

		predictions = model(inputs)
		loss = loss_fn(predictions, targets)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

	print(f"Loss: {loss.item()}")

def train(model, train_data_loader, test_data_loader, loss_fn, optimiser, device, epochs):
	for i in range(epochs):
		print(f"Epoch {i+1}")
		train_one_epoch(model, train_data_loader, loss_fn, optimiser, device)
		evaluate(model, test_data_loader, device)
		print("----------------")
	print("Training is done.")

if __name__ == "__main__":
	train_data, test_data = download_mnist_datasets_lenet()
	print("MNIST dataset downloaded")

	train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
	test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
	print(device)
	lenet = LeNet().to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(lenet.parameters(), lr=LEARNING_RATE)

	train(lenet, train_data_loader, test_data_loader, loss_fn, optimiser, device, EPOCHS)

	torch.save(lenet.state_dict(), os.path.join('saves/'+"lenet.pth"))
	print("Model trained and stored in saves at lenet.pth")
