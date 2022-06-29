import torch
from torch.utils.data import DataLoader
from modules.LeNet import LeNet5
from downloads import download_mnist_datasets_lenet

BATCH_SIZE = 128

def evaluate_lenet(model, data_loader, device):
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

            for label, prediction in zip(targets, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # print accuracy for each class
        #for classname, correct_count in correct_pred.items():
            #accuracy = 100 * float(correct_count) / total_pred[classname]
            #print('Accuracy for class {} is : {:.1f}'.format(classname, accuracy))
        #print(f'Accuracy is {float(correct) / float(total) * 100:.2f}')
        return (float(correct) / float(total))

if __name__ == "__main__":
    _, test_data = download_mnist_datasets_lenet()
    print("MNIST dataset downloaded")

    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    lenet = LeNet5().to(device)
    state_dict = torch.load("saves/lenet.pth")
    lenet.load_state_dict(state_dict)

    evaluate(lenet, test_data_loader, device)
