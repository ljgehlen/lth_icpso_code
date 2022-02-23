import torch
import random
import copy
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from modules.LeNet import LeNet5
from downloads import download_mnist_datasets_lenet
from evaluate_lenet import evaluate

BATCH_SIZE = 128
NUMBER_PARTICLES = 10
ITERATIONS = 20
PERCENTAGE = 0.2
INERTIA = 0.7
PBEST_CE = 1.5
GBEST_CE = 1.5

class Swarm:
    def __init__(self, module, test_data):
        self.module = module
        self.test_data = test_data
        self.particles = []
        self.gbest_acc = 0
        for i in range(NUMBER_PARTICLES):
            new_particle = Particle(self.module, self.test_data)
            self.particles.append(new_particle)
            if i == 0:
                self.gbest = copy.deepcopy(new_particle)
            else:
                if new_particle.accuracy() > self.gbest_acc:
                    self.gbest = copy.deepcopy(new_particle)

    def update_gbest(self):
        for i in range(NUMBER_PARTICLES):
            acc = self.particles[i].accuracy()
            if acc > self.gbest_acc:
                self.gbest = copy.deepcopy(self.particles[i])
                self.gbest_acc = acc

    def test(self):
        for i in range(NUMBER_PARTICLES):
            print(self.particles[i].accuracy())

    def run(self):
        for i in range(ITERATIONS):
            print("Iteration: ", i)
            for j in range(NUMBER_PARTICLES):
                self.particles[j].update_distribution(self.gbest)
            self.update_gbest()
            for j in range(NUMBER_PARTICLES):
                self.particles[j].update_pbest()
                print(self.particles[j].accuracy())
            print("--------------------------------")


class Particle:
    def __init__(self, module, test_data):
        self.module = module
        self.test_data = test_data
        self.conv1 = torch.ones(module.conv1.weight.size())
        self.conv2 = torch.ones(module.conv2.weight.size())
        self.fc1 = torch.ones(module.fc1.weight.size())
        self.fc2 = torch.ones(module.fc2.weight.size())
        self.fc3 = torch.ones(module.fc3.weight.size())
        self.layers = []
        self.total_weight_parameter = 0
        self.conv1_weight_parameter = 0
        self.conv2_weight_parameter = 0
        self.fc1_weight_parameter = 0
        self.fc2_weight_parameter = 0
        self.fc3_weight_parameter = 0
        for name, parameter in self.module.named_parameters():
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
        self.conv1_pbest = []
        self.conv2_distribution = []
        self.conv2_velocity = []
        self.conv2_pbest = []
        self.fc1_distribution = []
        self.fc1_velocity = []
        self.fc1_pbest = []
        self.fc2_distribution = []
        self.fc2_velocity = []
        self.fc2_pbest = []
        self.fc3_distribution = []
        self.fc3_velocity = []
        self.fc3_pbest = []
        for i in range(self.conv1_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.conv1_distribution.append(rand)
            self.conv1_pbest.append(rand)
            self.conv1_velocity.append(rand2)
        for i in range(self.conv2_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.conv2_distribution.append(rand)
            self.conv2_pbest.append(rand)
            self.conv2_velocity.append(rand2)
        for i in range(self.fc1_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc1_distribution.append(rand)
            self.fc1_pbest.append(rand)
            self.fc1_velocity.append(rand2)
        for i in range(self.fc2_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc2_distribution.append(rand)
            self.fc2_pbest.append(rand)
            self.fc2_velocity.append(rand2)
        for i in range(self.fc3_weight_parameter):
            rand = random.random()
            rand2 = random.uniform(-0.1, 0.1)
            self.fc3_distribution.append(rand)
            self.fc3_pbest.append(rand)
            self.fc3_velocity.append(rand2)
        self.update_mask()
        self.conv1_pbest_m = copy.deepcopy(self.conv1)
        self.conv2_pbest_m = copy.deepcopy(self.conv2)
        self.fc1_pbest_m = copy.deepcopy(self.fc1)
        self.fc2_pbest_m = copy.deepcopy(self.fc2)
        self.fc3_pbest_m = copy.deepcopy(self.fc3)

    def update_distribution(self, gbest):
        for i in range(self.conv1_weight_parameter):
            dist = self.conv1_distribution[i] + (INERTIA * self.conv1_velocity[i]) + (random.uniform(0, PBEST_CE) * (self.conv1_pbest[i] - self.conv1_distribution[i])) + (random.uniform(0, GBEST_CE) * (gbest.conv1_distribution[i] - self.conv1_distribution[i]))
            if dist < 0: dist = 0
            if dist > 1: dist = 1
            self.conv1_velocity[i] = dist - self.conv1_distribution[i]
            self.conv1_distribution[i] = dist
        for i in range(self.conv2_weight_parameter):
            dist = self.conv2_distribution[i] + (INERTIA * self.conv2_velocity[i]) + (random.uniform(0, PBEST_CE) * (self.conv2_pbest[i] - self.conv2_distribution[i])) + (random.uniform(0, GBEST_CE) * (gbest.conv2_distribution[i] - self.conv2_distribution[i]))
            if dist < 0: dist = 0
            if dist > 1: dist = 1
            self.conv2_velocity[i] = dist - self.conv2_distribution[i]
            self.conv2_distribution[i] = dist
        for i in range(self.fc1_weight_parameter):
            dist = self.fc1_distribution[i] + (INERTIA * self.fc1_velocity[i]) + (random.uniform(0, PBEST_CE) * (self.fc1_pbest[i] - self.fc1_distribution[i])) + (random.uniform(0, GBEST_CE) * (gbest.fc1_distribution[i] - self.fc1_distribution[i]))
            if dist < 0: dist = 0
            if dist > 1: dist = 1
            self.fc1_velocity[i] = dist - self.fc1_distribution[i]
            self.fc1_distribution[i] = dist
        for i in range(self.fc2_weight_parameter):
            dist = self.fc2_distribution[i] + (INERTIA * self.fc2_velocity[i]) + (random.uniform(0, PBEST_CE) * (self.fc2_pbest[i] - self.fc2_distribution[i])) + (random.uniform(0, GBEST_CE) * (gbest.fc2_distribution[i] - self.fc2_distribution[i]))
            if dist < 0: dist = 0
            if dist > 1: dist = 1
            self.fc2_velocity[i] = dist - self.fc2_distribution[i]
            self.fc2_distribution[i] = dist
        for i in range(self.fc3_weight_parameter):
            dist = self.fc3_distribution[i] + (INERTIA * self.fc3_velocity[i]) + (random.uniform(0, PBEST_CE) * (self.fc3_pbest[i] - self.fc3_distribution[i])) + (random.uniform(0, GBEST_CE) * (gbest.fc3_distribution[i] - self.fc3_distribution[i]))
            if dist < 0: dist = 0
            if dist > 1: dist = 1
            self.fc3_velocity[i] = dist - self.fc3_distribution[i]
            self.fc3_distribution[i] = dist
        self.update_mask()

    def update_pbest(self):
        self.update_pbest_mask()
        if self.accuracy() > self.accuracy_pbest():
            self.conv1_pbest = copy.deepcopy(self.conv1_distribution)
            self.conv2_pbest = copy.deepcopy(self.conv2_distribution)
            self.fc1_pbest = copy.deepcopy(self.fc1_distribution)
            self.fc2_pbest = copy.deepcopy(self.fc2_distribution)
            self.fc3_pbest = copy.deepcopy(self.fc3_distribution)

    def update_mask(self):
        self.conv1 = torch.ones(self.module.conv1.weight.size())
        self.conv2 = torch.ones(self.module.conv2.weight.size())
        self.fc1 = torch.ones(self.module.fc1.weight.size())
        self.fc2 = torch.ones(self.module.fc2.weight.size())
        self.fc3 = torch.ones(self.module.fc3.weight.size())
        #conv1
        position = 0
        used = 0
        needed = self.conv1_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.conv1.shape[0]):
                for j in range(self.conv1[0].shape[0]):
                    for k in range(self.conv1[0][0].shape[0]):
                        for l in range(self.conv1[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv1[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv1_distribution[position]:
                                    self.conv1[i][j][k][l] = 0
                                    used += 1
                            position += 1
            position = 0
        #conv2
        position = 0
        used = 0
        needed = self.conv2_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.conv2.shape[0]):
                for j in range(self.conv2[0].shape[0]):
                    for k in range(self.conv2[0][0].shape[0]):
                        for l in range(self.conv2[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv2[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv2_distribution[position]:
                                    self.conv2[i][j][k][l] = 0
                                    used += 1
                            position += 1
            position = 0
        #fc1
        position = 0
        used = 0
        needed = self.fc1_weight_parameter * PERCENTAGE
        while(used < needed):
            for i in range(self.fc1.shape[0]):
                for j in range(self.fc1[0].shape[0]):
                    if used >= needed: break
                    if self.fc1[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc1_distribution[position]:
                            self.fc1[i][j] = 0
                            used += 1
                        position += 1
            position = 0
        #fc2
        position = 0
        used = 0
        needed = self.fc2_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.fc2.shape[0]):
                for j in range(self.fc2[0].shape[0]):
                    if used >= needed: break
                    if self.fc2[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc2_distribution[position]:
                            self.fc2[i][j] = 0
                            used += 1
                        position += 1
            position = 0
        #fc3
        position = 0
        used = 0
        needed = self.fc3_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.fc3.shape[0]):
                for j in range(self.fc3[0].shape[0]):
                    if used >= needed: break
                    if self.fc3[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc3_distribution[position]:
                            self.fc3[i][j] = 0
                            used += 1
                        position += 1
            position = 0

    def update_pbest_mask(self):
        self.conv1_pbest_m = torch.ones(self.module.conv1.weight.size())
        self.conv2_pbest_m = torch.ones(self.module.conv2.weight.size())
        self.fc1_pbest_m = torch.ones(self.module.fc1.weight.size())
        self.fc2_pbest_m = torch.ones(self.module.fc2.weight.size())
        self.fc3_pbest_m = torch.ones(self.module.fc3.weight.size())
        # conv1
        position = 0
        used = 0
        needed = self.conv1_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.conv1_pbest_m.shape[0]):
                for j in range(self.conv1_pbest_m[0].shape[0]):
                    for k in range(self.conv1_pbest_m[0][0].shape[0]):
                        for l in range(self.conv1_pbest_m[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv1_pbest_m[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv1_pbest[position]:
                                    self.conv1_pbest_m[i][j][k][l] = 0
                                    used += 1
                            position += 1
            position = 0
        # conv2
        position = 0
        used = 0
        needed = self.conv2_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.conv2_pbest_m.shape[0]):
                for j in range(self.conv2_pbest_m[0].shape[0]):
                    for k in range(self.conv2_pbest_m[0][0].shape[0]):
                        for l in range(self.conv2_pbest_m[0][0][0].shape[0]):
                            if used >= needed: break
                            if self.conv2_pbest_m[i][j][k][l] > 0:
                                rand = random.random()
                                if rand <= self.conv2_pbest[position]:
                                    self.conv2_pbest_m[i][j][k][l] = 0
                                    used += 1
                            position += 1
            position = 0
        # fc1
        position = 0
        used = 0
        needed = self.fc1_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.fc1_pbest_m.shape[0]):
                for j in range(self.fc1_pbest_m[0].shape[0]):
                    if used >= needed: break
                    if self.fc1_pbest_m[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc1_pbest[position]:
                            self.fc1_pbest_m[i][j] = 0
                            used += 1
                        position += 1
            position = 0
        # fc2
        position = 0
        used = 0
        needed = self.fc2_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.fc2_pbest_m.shape[0]):
                for j in range(self.fc2_pbest_m[0].shape[0]):
                    if used >= needed: break
                    if self.fc2_pbest_m[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc2_pbest[position]:
                            self.fc2_pbest_m[i][j] = 0
                            used += 1
                        position += 1
            position = 0
        # fc3
        position = 0
        used = 0
        needed = self.fc3_weight_parameter * PERCENTAGE
        while (used < needed):
            for i in range(self.fc3_pbest_m.shape[0]):
                for j in range(self.fc3_pbest_m[0].shape[0]):
                    if used >= needed: break
                    if self.fc3_pbest_m[i][j] > 0:
                        rand = random.random()
                        if rand <= self.fc3_pbest[position]:
                            self.fc3_pbest_m[i][j] = 0
                            used += 1
                        position += 1
            position = 0

    def accuracy(self):
        #if torch.cuda.is_available():
            #device = "cuda"
        #else:
            #device = "cpu"
        device = "cpu"
        test_module = copy.deepcopy(self.module)
        test_module.conv1 = prune.custom_from_mask(test_module.conv1, 'weight', self.conv1)
        test_module.conv2 = prune.custom_from_mask(test_module.conv2, 'weight', self.conv2)
        test_module.fc1 = prune.custom_from_mask(test_module.fc1, 'weight', self.fc1)
        test_module.fc2 = prune.custom_from_mask(test_module.fc2, 'weight', self.fc2)
        test_module.fc3 = prune.custom_from_mask(test_module.fc3, 'weight', self.fc3)
        acc = evaluate(test_module, self.test_data, device)
        return acc

    def accuracy_pbest(self):
        #if torch.cuda.is_available():
            #device = "cuda"
        #else:
            #device = "cpu"
        device = "cpu"
        test_module = copy.deepcopy(self.module)
        test_module.conv1 = prune.custom_from_mask(test_module.conv1, 'weight', self.conv1_pbest_m)
        test_module.conv2 = prune.custom_from_mask(test_module.conv2, 'weight', self.conv2_pbest_m)
        test_module.fc1 = prune.custom_from_mask(test_module.fc1, 'weight', self.fc1_pbest_m)
        test_module.fc2 = prune.custom_from_mask(test_module.fc2, 'weight', self.fc2_pbest_m)
        test_module.fc3 = prune.custom_from_mask(test_module.fc3, 'weight', self.fc3_pbest_m)
        acc = evaluate(test_module, self.test_data, device)
        return acc

if __name__ == "__main__":
    _, test_data = download_mnist_datasets_lenet()
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    # if torch.cuda.is_available():
    # device = "cuda"
    # else:
    # device = "cpu"
    device = "cpu"
    lenet = LeNet5().to(device)
    swarm = Swarm(lenet, test_data_loader)
    swarm.run()
