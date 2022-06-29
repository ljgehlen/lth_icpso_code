
class config:
    def __init__(self, architecture, module, module_untrained, test_data, number_particles, iterations, learning, inertia,
                 pbest_ce, gbest_ce, percentage, percentage_mode, initialize, trained, formation, formation_reach, analyze, number_rand_masks):

        # data
        self.architecture = architecture
        self.module = module
        self.module_untrained = module_untrained
        self.test_data = test_data

        self.number_particles = number_particles
        self.iterations = iterations
        self.learning = learning

        self.inertia = inertia
        self.pbest_ce = pbest_ce
        self.gbest_ce = gbest_ce

        self.percentage = percentage
        self.percentage_mode = percentage_mode

        self.initialize = initialize
        self.trained = trained

        self.formation = formation
        self.formation_reach = formation_reach

        self.analyze = analyze
        self.number_rand_masks = number_rand_masks