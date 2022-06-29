from torchvision import datasets, transforms

def download_mnist_datasets_lenet():
	# download MNIST dataset for LeNet5 (28x28 -> 32x32)
	transforming = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
	train_data = datasets.MNIST(
		root="data",
		download=True,
		train=True,
		transform=transforming
	)
	test_data = datasets.MNIST(
		root="data",
		download=True,
		train=False,
		transform=transforming
	)
	return train_data, test_data

#def download_imagenet_datasets():
	#to-do: ~150 GB