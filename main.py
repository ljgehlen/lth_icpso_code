import torch
from torch.utils.data import DataLoader
from modules.LeNet import LeNet5
from downloads import download_mnist_datasets_lenet
from icpso import Swarm
from settings import config

BATCH_SIZE = 128

#NUMBER_PARTICLES = 3
#ITERATIONS = 5
#PERCENTAGE = 0.25
#INERTIA = 0.7
#PBEST_CE = 1.5
#GBEST_CE = 1.5

_, test_data = download_mnist_datasets_lenet()
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
# if torch.cuda.is_available():
# device = "cuda"
# else:
#device = "cpu"
device = "cpu"

#lenet = LeNet5().to(device)
#state_dict = torch.load("saves/module.pth")
#lenet.load_state_dict(state_dict)
lenet = LeNet5().to(device)

#conf = config(architecture, module, module_untrained, test_data, number_particles, iterations, learning,
#                 inertia, pbest_ce, gbest_ce, percentage, percentage_mode,
#                 initialize, trained, formation, formation_reach, analyze, number_rand_masks)
conf = config("lenet", lenet, _, test_data_loader, 2, 5, "true",
              0.7, 1.5, 1.5, 0.1, "total",
              "random", "false", "random", "global", "random", 1)

swarm = Swarm(conf)
swarm.run()