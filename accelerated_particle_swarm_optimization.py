#!/usr/bin/env python

# Library Imports
import torch
import time

# Main Class for Particle Swarm Optimization
class AcceleratedParticleSwarmOptimizer(object):
    def __init__(self,swarm_size=100,options=None):
        if (options == None):
            options = [0.5,0.7,100]
        self.swarm_size = swarm_size
        self.alpha = options[0]
        self.beta = options[1]
        self.max_iterations = options[2]

    def optimize(self,function):
        self.fitness_function = function

    def search_space(self,upper_bound,lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.dimensionality = upper_bound.size()[0]

    def populate(self):
        self.position = ((self.upper_bound - self.lower_bound)*torch.rand(1,self.swarm_size)) + self.lower_bound

    def enforce_bounds(self):
        upper_bound = self.upper_bound.view(self.dimensionality,1)
        lower_bound = self.lower_bound.view(self.dimensionality,1)
        self.position = torch.max(torch.min(self.position,upper_bound),lower_bound)

    def run(self,verbosity = True):
        self.current_fitness = self.fitness_function(self.position)
        self.global_best = self.position[:,self.current_fitness.argmin()].view(self.dimensionality,1)
        self.global_best_fitness = self.current_fitness.min()
        self.scaling = self.upper_bound - self.lower_bound
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            self.position = (((1 -self.beta)*self.position) + (self.beta*self.global_best) + ((self.alpha**iteration)*self.scaling*torch.randn(1)))
            self.enforce_bounds()
            self.current_fitness = self.fitness_function(self.position)
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
    p = AcceleratedParticleSwarmOptimizer(SwarmSize)
    p.optimize(SumOfSquares)
    p.search_space(UpperBound,LowerBound)
    p.populate()
    p.run()

