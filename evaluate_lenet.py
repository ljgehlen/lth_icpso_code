import torch
from torch.utils.data import DataLoader
from modules.LeNet5 import LeNet
from downloads import download_mnist_datasets_lenet

BATCH_SIZE = 128

def evaluate(model, data_loader, device):
    correct = 0
    samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            predicted = prediction[0].argmax(0)
            samples += prediction.shape[0]
            correct += (predicted == targets).sum()

        print(f'Got {correct} / {samples} with accuracy: {float(correct) / float(samples) * 100:.2f}')

if __name__ == "__main__":
    _, test_data = download_mnist_datasets_lenet()
    print("MNIST dataset downloaded")

    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    lenet = LeNet().to(device)
    state_dict = torch.load("saves/lenet.pth")
    lenet.load_state_dict(state_dict)

    evaluate(lenet, test_data_loader, device)
