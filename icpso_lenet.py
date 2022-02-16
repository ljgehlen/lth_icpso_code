import torch
import random
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from modules.LeNet import LeNet5
from downloads import download_mnist_datasets_lenet
from evaluate_lenet import evaluate

BATCH_SIZE = 128
NUMBER_PARTICLES = 10
ITERATIONS = 20
PERCENTAGE = 0.3

class Swarm(module, test_data):
	def __init__(self, module, test_data):
		self.module = module
		self.modus = modus
		self.test_data = test_data
		self.particles = []
		for i in range(NUMBER_PARTICLES):
			new_particle = Particle(module)
			self.particles.append(new_particle)

	def run(self):
		for i in range(ITERATIONS):

class Particle(module):
    def __init__(self, module, test_data):
        self.conv1 = (module.conv1.weight).clone()
        for i in range(self.conv1.shape[0]):
            for j in range(self.conv1[0].shape[0]):
                for k in range(self.conv1[0][0].shape[0]):
                    for l in range(self.conv1[0][0][0].shape[0]):
                        self.conv1[i][j][k][l] = 1
        self.conv2 = (module.conv2.weight).clone()
        for i in range(self.conv2.shape[0]):
            for j in range(self.conv2[0].shape[0]):
                for k in range(self.conv2[0][0].shape[0]):
                    for l in range(self.conv2[0][0][0].shape[0]):
                        self.conv2[i][j][k][l] = 1
        self.fc1 = (module.fc1.weight).clone()
        for i in range(self.fc1.shape[0]):
            for j in range(self.fc1[0].shape[0]):
                for k in range(self.fc1[0][0].shape[0]):
                    for l in range(self.fc1[0][0][0].shape[0]):
                        self.fc1[i][j][k][l] = 1
        self.fc2 = (module.fc2.weight).clone()
        for i in range(self.fc2.shape[0]):
            for j in range(self.fc2[0].shape[0]):
                for k in range(self.fc2[0][0].shape[0]):
                    for l in range(self.fc2[0][0][0].shape[0]):
                        self.fc2[i][j][k][l] = 1
        self.fc3 = (module.fc3.weight).clone()
        for i in range(self.fc3.shape[0]):
            for j in range(self.fc3[0].shape[0]):
                for k in range(self.fc3[0][0].shape[0]):
                    for l in range(self.fc3[0][0][0].shape[0]):
                        self.fc3[i][j][k][l] = 1
        self.layers = []
        self.total_weight_parameter = 0
        self.conv1_weight_parameter = 0
        self.conv2_weight_parameter = 0
        self.fc1_weight_parameter = 0
        self.fc2_weight_parameter = 0
        self.fc3_weight_parameter = 0
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad: continue
            param = name, parameter.numel()
            self.layers.append(param)
            if "weight" in name:
                self.total_weight_parameter += parameter.numel()
                if "conv1" in name: self.conv1_weight_parameter += parameter.numel()
                if "conv2" in name: self.conv2_weight_parameter += parameter.numel()
                if "fc1" in name: self.fc1_weight_parameter += parameter.numel()
                if "fc2" in name: self.fc2_weight_parameter += parameter.numel()
                if "fc3" in name: self.fc3_weight_parameter += parameter.numel()
        self.kernel_size = self.conv1[0][0].shape[0] * self.conv1[0][0][0].shape[0]
        self.conv1_distribution = []
        self.conv1_velocity = []
        self.conv2_distribution = []
        self.conv2_velocity = []
        self.fc1_distribution = []
        self.fc1_velocity = []
        self.fc2_distribution = []
        self.fc2_velocity = []
        self.fc3_distribution = []
        self.fc3_velocity = []
        for i in range(self.conv1_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.conv1_distribution.append(rand)
            self.conv1_velocity.append(rand2)
        for i in range(self.conv2_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.conv2_distribution.append(rand)
            self.conv2_velocity.append(rand2)
        for i in range(self.fc1_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc1_distribution.append(rand)
            self.fc1_velocity.append(rand2)
        for i in range(self.fc2_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc2_distribution.append(rand)
            self.fc2_velocity.append(rand2)
        for i in range(self.fc3_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc3_distribution.append(rand)
            self.fc3_velocity.append(rand2)
        self.randomize()
        # pbest logic missing
        self.update(test_data)

    def randomize(self):
        #conv1
        position = 0
        used = 0
        needed = self.conv1_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.conv1.shape[0]):
                if used >= needed: break
                for j in range(self.conv1[0].shape[0]):
                    if used >= needed: break
                    for k in range(self.conv1[0][0].shape[0]):
                        if used >= needed: break
                        for l in range(self.conv1[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv1[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv1_distribution[position]
                                    self.conv1[i][j][k][l] = 0
                                    used += 1
        #conv2
        position = 0
        used = 0
        needed = self.conv2_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.conv2.shape[0]):
                if used >= needed: break
                for j in range(self.conv2[0].shape[0]):
                    if used >= needed: break
                    for k in range(self.conv2[0][0].shape[0]):
                        if used >= needed: break
                        for l in range(self.conv2[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv2[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv2_distribution[position]
                                    self.conv2[i][j][k][l] = 0
                                    used += 1
        #fc1
        position = 0
        used = 0
        needed = self.fc1_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.fc1.shape[0]):
                if used >= needed: break
                for j in range(self.fc1[0].shape[0]):
                    if used >= needed: break
                    for k in range(self.fc1[0][0].shape[0]):
                        if used >= needed: break
                        for l in range(self.fc1[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.fc1[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.fc1_distribution = [position]
                                    self.fc1[i][j][k][l] = 0
                                    used += 1
        #fc2
        position = 0
        used = 0
        needed = self.fc2_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.fc2.shape[0]):
                if used >= needed: break
                for j in range(self.fc2[0].shape[0]):
                    if used >= needed: break
                    for k in range(self.fc2[0][0].shape[0]):
                        if used >= needed: break
                        for l in range(self.fc2[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.fc2[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.fc2_distribution = [position]
                                    self.fc2[i][j][k][l] = 0
                                    used += 1

        #fc3
        position = 0
        used = 0
        needed = self.fc3_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.fc3.shape[0]):
                if used >= needed: break
                for j in range(self.fc3[0].shape[0]):
                    if used >= needed: break
                    for k in range(self.fc3[0][0].shape[0]):
                        if used >= needed: break
                        for l in range(self.fc3[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.fc3[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.fc3_distribution = [position]
                                    self.fc3[i][j][k][l] = 0
                                    used += 1

    def update(self, module, test_data):

    def accuracy(self, module, test_data):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        test_module = module.copy()
        test_module.conv1 = prune.custom_from_mask(test_module.conv1, 'weight', self.conv1)
        test_module.conv2 = prune.custom_from_mask(test_module.conv2, 'weight', self.conv2)
        test_module.fc1 = prune.custom_from_mask(test_module.fc1, 'weight', self.fc1)
        test_module.fc2 = prune.custom_from_mask(test_module.fc2, 'weight', self.fc2)
        test_module.fc3 = prune.custom_from_mask(test_module.fc3, 'weight', self.fc3)
        acc = evaluate(test_module, test_data, device)
        return acc

if __name__ == "__main__":
    _, test_data = download_mnist_datasets_lenet()
	test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
	lenet = LeNet5().to(device)
	informations = module_information(lenet, "weight")
	swarm = Swarm(lenet, "weight", test_data_loader, informations)
