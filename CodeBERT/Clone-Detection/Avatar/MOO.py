import csv
import time
import random
import logging
import numpy as np

from tqdm import tqdm
from distill_utils import distill, hyperparams_convert
from surrogate import predictor
from flops import TransformerHparams
from reduction import reduction_z3

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

class Candidate(object):
    def __init__(self, candidates_vals):
        self.candidate_values = candidates_vals
        self.objective_values = []

    def get_candidate_values(self):
        return self.candidate_values

    def set_objective_values(self, objective_values):
        self.objective_values = objective_values

    def get_objective_values(self):
        return self.objective_values

    def set_candidate_values_at_index(self, indx, val):
        self.candidate_values[indx] = val

def generate_random_population(size, lb, ub):
    random_pop = []

    for i in range(size):
        while True:
            candidate_vals = []
            for index in range(len(lb)):
                if index == 5 or index == 8:
                    candidate_vals.append(round(random.uniform(lb[index], ub[index]+0.1), 1))
                else:
                    candidate_vals.append(int(random.uniform(lb[index], ub[index]+1)))
            if candidate_vals[3] % candidate_vals[7] == 0:
                break
        random_pop.append(Candidate(candidate_vals))

    return random_pop

def calculate_minimum_distance(candidate, random_pop):
    distance = 1e9

    for each_candidate in random_pop:
        vals = each_candidate.get_candidate_values()
        candidate_vals = candidate.get_candidate_values()
        dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
        if dist < distance:
            distance = dist

    return distance

def generate_adaptive_random_population(size, lb, ub):
    random_pop = []

    initial_pop = generate_random_population(10, lb, ub)[0]

    random_pop.append(initial_pop)

    while len(random_pop) < size:
        D = 0
        selected_candidate = None
        rp = generate_random_population(size, lb, ub)
        for each_candidate in rp:
            min_dis = calculate_minimum_distance(each_candidate, random_pop)
            if min_dis > D:
                D = min_dis
                selected_candidate = each_candidate

        random_pop.append(selected_candidate)

    return random_pop

def evaulate_population(population, surrogate_model):
    for each_candidate in population:
        candidate_values = each_candidate.get_candidate_values()
        model = TransformerHparams(candidate_values[3], candidate_values[2], candidate_values[9], candidate_values[1], candidate_values[6], candidate_values[7])
        size = abs(model.get_params()*4/1e6 - 3)
        flops = model.get_infer_flops()/1e9
        accuracy = surrogate_model.predict([candidate_values])[0]
        fitnesses = [size, accuracy, flops]
        each_candidate.set_objective_values(fitnesses)

def dominates(candidate1, candidate2):
    candidate1_objectives = candidate1.get_objective_values()
    candidate2_objectives = candidate2.get_objective_values()
    dominates = False
    if candidate1_objectives[0] < candidate2_objectives[0] and candidate1_objectives[1] > candidate2_objectives[1] and candidate1_objectives[2] < candidate2_objectives[2]:
        dominates = True
    return dominates

def update_archive(pop, archive):
    for each_candidate in pop:
        dominated = False
        for each_archive in archive:
            if dominates(each_archive, each_candidate):
                dominated = True
                break
        if not dominated:
            if len(archive) == 0:
                archive.append(each_candidate)
            else:
                to_remove = []
                for each_archive in archive:
                    if dominates(each_candidate, each_archive) or each_archive.get_candidate_values() == each_candidate.get_candidate_values():
                        to_remove.append(each_archive)

                for each_remove in to_remove:
                    archive.remove(each_remove)
                archive.append(each_candidate)

def partially_mapped_crossover(parent1, parent2):
    parent1_values = parent1.get_candidate_values()
    parent2_values = parent2.get_candidate_values()
    size = len(parent1_values)
    child1 = [-1] * size
    child2 = [-1] * size

    crossover_point1, crossover_point2 = sorted(random.sample(range(size), 2))

    for i in range(crossover_point1, crossover_point2 + 1):
        if parent2_values[i] not in child1:
            next_index = i
            while child1[next_index] != -1:
                next_index = parent2_values.index(parent1_values[next_index])
            child1[next_index] = parent2_values[i]

        if parent1_values[i] not in child2:
            next_index = i
            while child2[next_index] != -1:
                next_index = parent1_values.index(parent2_values[next_index])
            child2[next_index] = parent1_values[i]

    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2_values[i]
        if child2[i] == -1:
            child2[i] = parent1_values[i]

    return Candidate(child1), Candidate(child2)

def boundary_random_mutation(candidate, lb, ub, thresh):
    candidate_values = candidate.get_candidate_values()
    for index in range(len(candidate_values)):
        if random.uniform(0, 1) < thresh:
            if index == 5 or index == 8:
                candidate_values[index] = round(random.uniform(lb[index], ub[index]+0.1), 1)
            else:
                candidate_values[index] = int(random.uniform(lb[index], ub[index]+1))

    return candidate

def correct(pop, lb, ub):
    for indx in range(len(pop)):
        candidate = pop[indx]
        values = candidate.get_candidate_values()
        # for value_index in range(len(values)):
        #     pop[indx].set_candidate_values_at_index(value_index, int(pop[indx].get_candidate_values()[value_index]))
        #     while values[value_index] > ub[value_index] or values[value_index] < lb[value_index]:
        #         temp = generate_random_population(1, lb, ub)[0]
        #         pop[indx].set_candidate_values_at_index(value_index, int(temp.get_candidate_values()[value_index]))

        while values[3] % values[7] != 0:
            temp = generate_random_population(1, lb, ub)[0]
            pop[indx].set_candidate_values_at_index(3, int(temp.get_candidate_values()[3]))

    return pop

def select_best(tournament_candidates):
    best = tournament_candidates[0]
    for i in range(len(tournament_candidates)):
        candidate1 = tournament_candidates[i]
        for j in range(len(tournament_candidates)):
            candidate2 = tournament_candidates[j]
            if (dominates(candidate1, candidate2)):
                best = candidate1
    return best

def tournament_selection(pop, size):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates)
    return best

def generate_off_springs(pop, lb, ub):
    size = len(pop)
    population_to_return = []

    while len(population_to_return) < size:
        parent1 = tournament_selection(pop, 10)
        parent2 = tournament_selection(pop, 10)
        while parent1 == parent2:
            parent2 = tournament_selection(pop, 10)

        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= 0.60:
            parent1, parent2 = partially_mapped_crossover(parent1, parent2)
        child1 = boundary_random_mutation(parent1, lb, ub, 0.1)
        child2 = boundary_random_mutation(parent2, lb, ub, 0.1)

        population_to_return.append(child1)
        population_to_return.append(child2)

    return population_to_return

if __name__ == "__main__":
    start_time = time.time()
    # lb, ub = reduction_z3()
    # logging.info("Time taken for reduction: {}".format(time.time() - start_time))
    # logging.info("Lower Bound: {}".format(lb))
    # logging.info("Upper Bound: {}".format(ub))
    lb = [1, 1000, 1, 16, 1, 0.2, 32, 1, 0.2, 256, 1, 1, 1]
    ub = [4, 46000, 12, 256, 4, 0.5, 3072, 12, 0.5, 512, 3, 3, 2]

    pop = generate_adaptive_random_population(20, lb, ub)

    surrogate_data = []
    for each_pop in pop:
        surrogate_data.append(each_pop.get_candidate_values())

    accs = distill(surrogate_data, surrogate=True)

    with open("surrogate_train_data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob", "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length", "Position Embedding Type", "Learning Rate", "Batch Size", "Accuracy"])
        for d, acc in zip(surrogate_data, accs):
            writer.writerow(hyperparams_convert(d) + [acc])

    surrogate_model = predictor([surrogate_data, accs])

    # import ast
    # with open("accs.jsonl") as f:
    #     data = f.readlines()
    #     surrogate_data_acc = []
    #     data1 = []
    #     data2 = []
    #     for line in data:
    #         data1.append(ast.literal_eval(line.split("] ")[0]+"]"))
    #         data2.append(float(line.split("] ")[1]))

    # surrogate_data_acc = [data1, data2]
    # surrogate_model = predictor(surrogate_data_acc)

    evaulate_population(pop, surrogate_model)
    archive = []
    update_archive(pop, archive)

    iteration = 50
    for i in tqdm(range(iteration)):
        pop = generate_off_springs(pop, lb, ub)
        pop = correct(pop, lb, ub)
        evaulate_population(pop, surrogate_model)
        update_archive(pop, archive)

    logging.info("Time taken: {}".format(time.time() - start_time))
    logging.info("Number of solutions in the archive: {}".format(len(archive)))
    logging.info("Saving the archive to the file")
    with open("pareto_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob", "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length", "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS"])
        for each_archive in archive:
            writer.writerow(each_archive.get_candidate_values() + each_archive.get_objective_values())