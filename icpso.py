import os
import time
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from modules.LeNet import LeNet5
from downloads import download_mnist_datasets_lenet
from evaluate_lenet import evaluate

BATCH_SIZE = 128
NUMBER_PARTICLES = 10
ITERATIONS = 30
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
                self.gbest_acc = new_particle.accuracy
            else:
                if new_particle.accuracy > self.gbest_acc:
                    self.gbest = copy.deepcopy(new_particle)
                    self.gbest_acc = new_particle.accuracy
        self.part_acc = []
        self.part_pb_acc = []
        self.gb_acc = []
        self.swarm_av = []
        self.swarm_pb_av = []
        for i in range(NUMBER_PARTICLES):
            self.part_acc.append([])
            self.part_pb_acc.append([])

    def update_gbest(self):
        for i in range(NUMBER_PARTICLES):
            acc = self.particles[i].accuracy
            if acc > self.gbest_acc:
                self.gbest = copy.deepcopy(self.particles[i])
                self.gbest_acc = acc

    def update_status(self):
        self.gb_acc.append(self.gbest_acc)
        average = 0
        pb_average = 0
        for i in range(NUMBER_PARTICLES):
            self.part_acc[i].append(self.particles[i].accuracy)
            self.part_pb_acc[i].append(self.particles[i].pbest_accuracy)
            average += self.particles[i].accuracy
            pb_average += self.particles[i].pbest_accuracy
        self.swarm_av.append(average/NUMBER_PARTICLES)
        self.swarm_pb_av.append(pb_average/NUMBER_PARTICLES)

    def test(self):
        for i in range(NUMBER_PARTICLES):
            print(self.particles[i].accuracy)
            print(self.particles[i].mask["conv1.weight"])

    def run(self):
        self.update_status()
        for i in range(ITERATIONS):
            print("Iteration: ", i)
            for j in range(NUMBER_PARTICLES):
                self.particles[j].new_distribution(self.gbest)
                #self.particles[j].update()
            self.update_gbest()
            for j in range(NUMBER_PARTICLES):
                self.particles[j].update_pbest()
                print(self.particles[j].accuracy)
            self.update_status()
            print("--------------------------------")
        self.save_data()

    def save_data(self):
        path = "results/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(path))
        txt_path = path + "/data.txt"
        f = open(os.path.join(txt_path), "w")
        f.write("Number Particles: " + repr(NUMBER_PARTICLES) + "\nIterations: " + repr(ITERATIONS) + "\nInertia: " + repr(INERTIA) + "\nPBest_CE: " + repr(PBEST_CE) + "\nGBest_CE: " + repr(GBEST_CE))
        f.close()
        mod = path + "/module"
        mask = path + "/mask"
        torch.save(self.module.state_dict(), os.path.join(mod))
        torch.save(self.gbest.mask, os.path.join(mask))
        particle_acc = path + "/part_acc.npy"
        particle_pbest_acc = path + "/part_pbest_acc.npy"
        gbest_acc = path + "/gbest_acc.npy"
        swarm_avrg = path + "/swarm_avrg.npy"
        swarm_pbest_avrg = path + "/swarm_pbest_avrg.npy"
        np.save(os.path.join(particle_acc), np.asarray(self.part_acc))
        np.save(os.path.join(particle_pbest_acc), np.asarray(self.part_pb_acc))
        np.save(os.path.join(gbest_acc), np.asarray(self.gb_acc))
        np.save(os.path.join(swarm_avrg), np.asarray(self.swarm_av))
        np.save(os.path.join(swarm_pbest_avrg), np.asarray(self.swarm_pb_av))
        #gbest = path + "/gbest_acc_plot.jpg"
        pb_avg = path + "/avrg_pbest_plot.jpg"
        #plt.title("GBest")
        plt.title("Average PBest")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.plot(self.swarm_pb_av, color="red")
        plt.xticks(range(0, len(self.swarm_pb_av), 1))
        plt.savefig(pb_avg)

class Particle:
    def __init__(self, module, test_data):
        self.module = module
        self.test_data = test_data
        self.module_copy = copy.deepcopy(self.module.state_dict())
        self.distribution = {k: torch.rand(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}
        self.velocity = {k: (0.2 * torch.rand(v.size()) - 0.1) for k, v in self.module_copy.items() if k.endswith("weight")}
        self.pbest = copy.deepcopy(self.distribution)
        self.mask = self.update_mask()
        self.accuracy = self.update_accuracy()
        self.pbest_accuracy = self.accuracy

    def update(self):
        self.mask = self.update_mask()
        self.accuracy = self.update_accuracy()

    def update_mask(self):
        rand = {k: torch.rand(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}
        return {k: torch.where(v < rand[k], 0, 1) for k, v in self.distribution.items()}

    def update_accuracy(self):
        test_module = copy.deepcopy(self.module)
        for k, v in test_module.state_dict().items():
            if k.endswith("weight"):
                if k == "conv1.weight":
                    v = prune.custom_from_mask(test_module.conv1, 'weight', self.mask[k])
                elif k == "conv2.weight":
                    v = prune.custom_from_mask(test_module.conv2, 'weight', self.mask[k])
                elif k == "fc1.weight":
                    v = prune.custom_from_mask(test_module.fc1, 'weight', self.mask[k])
                elif k == "fc2.weight":
                    v = prune.custom_from_mask(test_module.fc2, 'weight', self.mask[k])
                elif k == "fc3.weight":
                    v = prune.custom_from_mask(test_module.fc3, 'weight', self.mask[k])
        return evaluate(test_module, self.test_data, "cpu")

    def update_pbest(self):
        if self.accuracy > self.pbest_accuracy:
            self.pbest = copy.deepcopy(self.distribution)
            self.pbest_accuracy = self.accuracy

    def new_distribution(self, gbest):
        for k, v in self.distribution.items():
            zwischen = torch.add(v, self.velocity[k], alpha=INERTIA)
            zwischen = torch.add(zwischen, torch.mul(torch.sub(self.pbest[k], v), PBEST_CE * torch.rand(zwischen.size())))
            zwischen = torch.add(zwischen, torch.mul(torch.sub(gbest.distribution[k], v), GBEST_CE * torch.rand(zwischen.size())))
            self.velocity[k] = torch.sub(zwischen, v)
            v = zwischen
        self.update()

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