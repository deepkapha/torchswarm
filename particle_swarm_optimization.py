#!/usr/bin/env python

# Library Imports
import torch
import time

# Main Class for Particle Swarm Optimization
class ParticleSwarmOptimizer(object):
    def __init__(self,swarm_size=100,options=None):
        if (options == None):
            options = [2,2,0.1,0.1,100]
        self.swarm_size = swarm_size
        self.c_cognitive = options[0]
        self.c_social = options[1]
        self.inertia_weight = options[2]
        self.velocity_limit = options[3]
        self.max_iterations = options[4]

    def optimize(self,function):
        self.fitness_function = function

    def search_space(self,upper_bound,lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.dimensionality = upper_bound.size()[0]    

    def populate(self):
        self.position = ((self.upper_bound - self.lower_bound)*torch.rand(1,self.swarm_size)) + self.lower_bound
        self.velocity = (2*self.velocity_limit*torch.rand(self.dimensionality,self.swarm_size)) - self.velocity_limit

    def enforce_bounds(self):
        upper_bound = self.upper_bound.view(self.dimensionality,1)
        lower_bound = self.lower_bound.view(self.dimensionality,1)
        self.position = torch.max(torch.min(self.position,upper_bound),lower_bound)
        self.velocity = torch.max(torch.min(self.velocity,torch.tensor(self.velocity_limit)),-torch.tensor(self.velocity_limit))

    def run(self,verbosity = True):
        self.current_fitness = self.fitness_function(self.position)
        self.particle_best = self.position
        self.particle_best_fitness = self.current_fitness
        self.global_best = self.position[:,self.particle_best_fitness.argmin()].view(self.dimensionality,1)
        self.global_best_fitness = self.particle_best_fitness.min()
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            self.velocity = ((self.inertia_weight*self.velocity) +
                            (self.c_cognitive*torch.rand(1)*(self.particle_best-self.position)) +
                            (self.c_social*torch.rand(1)*(self.global_best-self.position)))
            self.position = self.position + self.velocity
            self.enforce_bounds()
            self.current_fitness = self.fitness_function(self.position)
            local_mask = self.current_fitness<self.particle_best_fitness
            self.particle_best[:,local_mask] = self.position[:,local_mask]
            self.particle_best_fitness[local_mask] = self.current_fitness[local_mask]
            if (self.current_fitness.min() < self.global_best_fitness):
                self.global_best_fitness = self.current_fitness.min()
                self.global_best = self.position[:,self.current_fitness.argmin()].view(self.dimensionality,1)
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f} | current mean fitness {:.3f} | iteration time {:.3f}'
                .format(iteration + 1,self.global_best_fitness,self.current_fitness.mean(),toc-tic))

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example Usage
    Dimensions = int(1e6)
    SwarmSize = 10
    UpperBound = torch.ones(Dimensions,1)*10
    LowerBound = torch.ones(Dimensions,1)*(-10)
    def SumOfSquares(x):
        return (x**2).sum(dim=0)
    p = ParticleSwarmOptimizer(SwarmSize)
    p.optimize(SumOfSquares)
    p.search_space(UpperBound,LowerBound)
    p.populate()
    p.run()

