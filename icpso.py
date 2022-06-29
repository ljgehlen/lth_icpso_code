import os
import time
import copy
import numpy as np
import torch
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from evaluate_lenet import evaluate_lenet

BATCH_SIZE = 128

class Swarm:
    def __init__(self, settings):
        # settings enthält alle anpassbaren informationen
        self.conf = settings

        # swarm pso
        self.gbest_acc = 0 # genauigkeit/fitness von global_best
        self.particles = []
        for i in range(self.conf.number_particles):
            new_particle = Particle(settings)
            self.particles.append(new_particle)
            if i == 0:
                self.gbest = copy.deepcopy(new_particle)
                self.gbest_acc = new_particle.accuracy
            else:
                if new_particle.accuracy > self.gbest_acc:
                    self.gbest = copy.deepcopy(new_particle)
                    self.gbest_acc = new_particle.accuracy

        # save
        self.part_acc = []
        self.part_pb_acc = []
        self.gb_acc = []
        self.swarm_av = []
        self.swarm_pb_av = []
        self.perc_change = []
        for i in range(self.conf.number_particles):
            self.part_acc.append([])
            self.part_pb_acc.append([])
            self.perc_change.append([])

    # funktion updated global_best/kontrolliert alle partikel für eine mögliche besser lösung
    def update_gbest(self):
        for i in range(self.conf.number_particles):
            acc = self.particles[i].accuracy
            if acc > self.gbest_acc:
                self.gbest = copy.deepcopy(self.particles[i])
                self.gbest_acc = acc

    # speichert zwischeninformationen für das spätere auswerten
    def update_status(self):
        self.gb_acc.append(self.gbest_acc)
        average = 0
        pb_average = 0
        for i in range(self.conf.number_particles):
            self.part_acc[i].append(self.particles[i].accuracy)
            self.part_pb_acc[i].append(self.particles[i].pbest_accuracy)
            average += self.particles[i].accuracy
            pb_average += self.particles[i].pbest_accuracy
            self.perc_change[i].append(self.particles[i].change)
        self.swarm_av.append(average/self.conf.number_particles)
        self.swarm_pb_av.append(pb_average/self.conf.number_particles)

    def test(self):
        for i in range(self.conf.number_particles):
            print(self.particles[i].accuracy)
            print(self.particles[i].mask["conv1.weight"])

    def run(self):
        self.update_status()
        print(evaluate_lenet(self.conf.module, self.conf.test_data, "cpu"))
        percentage = np.linspace(0.01, self.conf.percentage, self.conf.iterations).tolist()
        for i in range(self.conf.iterations):
            print("Iteration: ", i)
            for j in range(self.conf.number_particles):
                if self.conf.learning == "true":
                    if self.conf.percentage_mode == "inkrementell":
                        self.particles[j].new_distribution(self.gbest, percentage[i])
                    elif self.conf.percentage_mode == "total":
                        self.particles[j].new_distribution(self.gbest, self.conf.percentage)
                elif self.conf.learning == "false": # naives probieren
                    self.particles[j].update()
            self.update_gbest()
            for j in range(self.conf.number_particles):
                self.particles[j].update_pbest()
                print(self.particles[j].accuracy)
            self.update_status()
            print("--------------------------------")
        self.save_data()

    # speichert auswertungsdaten
    def save_data(self):
        base_acc = evaluate_lenet(self.conf.module, self.conf.test_data, "cpu")
        path = "results/" + time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(path))
        txt_path = path + "/data.txt"
        f = open(os.path.join(txt_path), "w")
        f.write("Number Particles: " + repr(self.conf.number_particles) + "\nIterations: " + repr(self.conf.iterations) + "\nPruning Percentage: " + repr(self.conf.percentage) + "\nInertia: " + repr(self.conf.inertia) + "\nPBest_CE: " + repr(self.conf.pbest_ce) + "\nGBest_CE: " + repr(self.conf.gbest_ce) + "\nLearning: " + self.conf.learning + "\nBase Accuracy: " + repr(base_acc) + "\nGBest Accuracy: " + repr(self.gbest_acc))
        f.close()
        mod = path + "/module.pth"
        mask = path + "/mask"
        torch.save(self.conf.module.state_dict(), os.path.join(mod))
        torch.save(self.gbest.mask, os.path.join(mask))
        particle_acc = path + "/part_acc.npy"
        particle_pbest_acc = path + "/part_pbest_acc.npy"
        gbest_acc = path + "/gbest_acc.npy"
        swarm_avrg = path + "/swarm_avrg.npy"
        swarm_pbest_avrg = path + "/swarm_pbest_avrg.npy"
        particle_change = path + "/particle_change.npy"
        np.save(os.path.join(particle_acc), np.asarray(self.part_acc))
        np.save(os.path.join(particle_pbest_acc), np.asarray(self.part_pb_acc))
        np.save(os.path.join(gbest_acc), np.asarray(self.gb_acc))
        np.save(os.path.join(swarm_avrg), np.asarray(self.swarm_av))
        np.save(os.path.join(swarm_pbest_avrg), np.asarray(self.swarm_pb_av))
        np.save(os.path.join(particle_change), np.asarray(self.perc_change))
        gbest = path + "/gbest_acc_plot.jpg"
        pb_avg = path + "/avrg_pbest_plot.jpg"
        swarm_avg = path + "/avrg_plot.jpg"
        plt.figure(0)
        plt.title("Average PBest")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.plot(self.swarm_pb_av, color="red")
        plt.xticks(range(0, len(self.swarm_pb_av), 1))
        plt.savefig(pb_avg)
        plt.clf()
        plt.figure(1)
        plt.title("GBest")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.plot(self.gb_acc, color="blue")
        plt.xticks(range(0, len(self.gb_acc), 1))
        plt.savefig(gbest)
        plt.clf()
        plt.figure(2)
        plt.title("Average")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.plot(self.swarm_av, color="green")
        plt.xticks(range(0, len(self.swarm_av), 1))
        plt.savefig(swarm_avg)
        plt.clf()

class Particle:
    #def __init__(self, module, test_data, percentage, inertia, pbest_ce, gbest_ce):
    def __init__(self, settings):

        self.conf = settings
        self.module_copy = copy.deepcopy(self.conf.module.state_dict())

        # particle
        # gewichte
        if self.conf.initialize == "weights": # normalisiert gewichte über 0.1 bis 0.9 bevor eine maske für das verrauschen von -0.1 bis 0.1 dazu addiert wird
            self.distribution = {k: torch.clone(v) for k, v in self.module_copy.items() if k.endswith("weight")}
            first = 0
            for k, v in self.distribution.items():
                if first == 0:
                    values = torch.abs(v.flatten())
                    first = 1
                else:
                    values = torch.cat((values.flatten(), torch.abs(v.flatten())))
                self.distribution[k] = torch.abs(self.distribution[k])
            min_value = torch.min(values)
            max_value = torch.max(values)
            for k, v in self.distribution.items():
                new_dist = torch.sub(self.distribution[k], min_value)
                new_dist = torch.mul(new_dist, torch.div(0.8, torch.sub(max_value, min_value)))
                new_dist = torch.add(new_dist, 0.1)
                new_dist = torch.add(new_dist, torch.ones(v.size()).uniform_(-0.1, 0.1))
                new_dist = torch.sub(torch.ones(v.size()), new_dist)
                self.distribution[k] = new_dist

        # gradienten (nach training)
        elif (self.conf.initialize == "gradients") & (self.conf.trained == "true"):
            pass
        # random
        else:
            self.distribution = {k: torch.rand(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}

        self.velocity = {k: (0.2 * torch.rand(v.size()) - 0.1) for k, v in self.module_copy.items() if k.endswith("weight")}
        self.pbest = copy.deepcopy(self.distribution)
        if self.conf.percentage_mode == "inkrementell":
            self.mask = self.update_mask(0.01)
        elif self.conf.percentage_mode == "total":
            self.mask = self.update_mask(self.conf.percentage)
        self.accuracy = self.update_accuracy(self.mask)
        self.pbest_accuracy = self.accuracy
        self.change = 0

    def update(self, percentage):
        if self.conf.formation == "random" and self.conf.number_rand_masks > 1:
            masks = []
            masks_acc = []
            best_mask = 0
            for i in range(self.conf.number_rand_masks):
                new_mask = self.update_mask(percentage)
                masks.append(new_mask)
            for i in range(self.conf.number_rand_masks):
                acc = self.update_accuracy(masks[i])
                masks_acc.append(acc)
                if acc >= masks_acc[best_mask]:
                    best_mask = i
            equal = 0
            count = 0
            for k, v in masks[best_mask].items():
                equal += torch.sum(v == self.mask[k])
                count += torch.numel(v)
            self.change = equal/count
            self.mask = masks[best_mask]
            self.accuracy = masks_acc[best_mask]
        else:
            new_mask = self.update_mask(percentage)
            equal = 0
            count = 0
            for k, v in new_mask.items():
                equal += torch.sum(v == self.mask[k])
                count += torch.numel(v)
            self.change = equal/count
            self.mask = new_mask
            self.accuracy = self.update_accuracy(new_mask)

    def update_mask(self, percentage):

        if self.conf.formation == "random":
            mask = {k: torch.ones(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}
            rand = {k: torch.rand(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}
            if self.conf.formation_reach == "global":
                first = 0
                for k, v in self.distribution.items():
                    if first == 0:
                        values = torch.sub(self.distribution[k], rand[k]).flatten()
                        first = 1
                    else:
                        values = torch.cat((values.flatten(), torch.sub(self.distribution[k], rand[k]).flatten()))
                values = torch.sort(values)
                cutoff = int(len(values.__getitem__(0)) * percentage)
                global_cutoff_value = values.__getitem__(0)[cutoff]
            for k, v in mask.items():
                pruning_weighting = torch.sub(self.distribution[k], rand[k])
                if self.conf.formation_reach == "local":
                    number_elements = round(torch.numel(v) * percentage)
                    mask[k] = torch.where(torch.isin(pruning_weighting, torch.topk(pruning_weighting.flatten(), number_elements)[0]), torch.tensor(0), torch.tensor(1))
                elif self.conf.formation_reach == "global":
                    mask[k] = torch.where(torch.le(pruning_weighting, global_cutoff_value), torch.tensor(0), torch.tensor(1))

        elif self.conf.formation == "topk":
            first = 0
            for k, v in self.distribution.items():
                if first == 0:
                    values = v.flatten()
                    first = 1
                else:
                    values = torch.cat((values.flatten(), v.flatten()))
            values = torch.sort(values)
            cutoff = int(len(values.__getitem__(0)) * self.conf.percentage)
            global_cutoff_value = values.__getitem__(0)[cutoff]
            mask = {k: torch.ones(v.size()) for k, v in self.module_copy.items() if k.endswith("weight")}
            for k, v in mask.items():
                if self.conf.formation_reach == "local":
                   number_elements = round(torch.numel(v)*self.conf.percentage)
                   mask[k] = torch.where(torch.isin(self.distribution[k], torch.topk(self.distribution[k].flatten(), number_elements)[0]), torch.tensor(0), torch.tensor(1))
                elif self.conf.formation_reach == "global":
                   mask[k] = torch.where(torch.le(self.distribution[k], global_cutoff_value), torch.tensor(0), torch.tensor(1))

        return mask

    def update_accuracy(self, mask):
        test_module = copy.deepcopy(self.conf.module)
        for k, v in test_module.state_dict().items():
            names = k.split(sep='.')
            if "weight" in names:
                act_mod = getattr(test_module, names[0])
                for i in range(len(names) - 2):
                    act_mod = getattr(act_mod, names[i + 1])
                prune.custom_from_mask(act_mod, "weight", mask[k])
        acc = 0
        if self.conf.architecture == 'lenet':
            acc = evaluate_lenet(test_module, self.conf.test_data, "cpu")
        return acc

    def update_pbest(self):
        if self.accuracy > self.pbest_accuracy:
            self.pbest = copy.deepcopy(self.distribution)
            self.pbest_accuracy = self.accuracy

    def new_distribution(self, gbest, percentage):
        for k, v in self.distribution.items():
            zwischen = torch.add(v, self.velocity[k], alpha=self.conf.inertia)
            zwischen = torch.add(zwischen, torch.mul(torch.sub(self.pbest[k], v), self.conf.pbest_ce * torch.rand(zwischen.size())))
            zwischen = torch.add(zwischen, torch.mul(torch.sub(gbest.distribution[k], v), self.conf.gbest_ce * torch.rand(zwischen.size())))
            self.velocity[k] = torch.sub(zwischen, v)
            zwischen = torch.where(zwischen < torch.zeros(zwischen.size()), torch.tensor(0).float(), zwischen)
            zwischen = torch.where(zwischen > torch.ones(zwischen.size()), torch.tensor(1).float(), zwischen)
            self.distribution[k] = zwischen
        self.update(percentage)