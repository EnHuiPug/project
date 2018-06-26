import numpy as np

POPULATION_SIZE = 500
CROSS_RATE = 0.4
MUTATION_RATE = 0.05
GENERATIONS = 1000

N = 10  # number of students
CHROMOSOME_LENGTH = 15
replace_list = np.array([0, 2, 3, 5, 6, 8, 9, 11, 12, 14])
Target_schedule = np.zeros(shape=(N, CHROMOSOME_LENGTH), dtype=int)
# 学生编号 | 要求课时 | 要求上午（1）或下午（-1）
constrains = np.array([[0, 15, 0],
                       [1, 15, 1],
                       [2, 15, 0],
                       [3, 15, -1],
                       [4, 15, -1],
                       [5, 15, 0],
                       [6, 15, 1],
                       [7, 15, -1],
                       [8, 15, 1],
                       [9, 15, 0]])


class GA:
    def __init__(self, N, chromlen, cross_rate, mutation_rate, pop_size, constrains):
        self.students = N
        self.chromlen = chromlen
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.constrains = constrains

        self.pop = np.random.randint(2, size=(pop_size, N, chromlen))

    def fitness(self):
        result = np.zeros(self.pop_size) + 30
        flag = False
        for j in range(self.pop_size):
            table = self.pop[j]
            for i in range(N):
                line = table[i]
                totalhours = 0
                perference = 0
                timeslots = np.split(line, 5)
                for ary in timeslots:
                    hours = self.cal_timeslots(ary)
                    if hours != -1:
                        totalhours += hours
                    else:
                        totalhours += 1.25
                        result[j] -= 1
                    perference += self.cal_perference(ary)
                if perference/5 == self.constrains[i][2]:
                    result[j] += 1

                if totalhours == self.constrains[i][1]:
                    result[j] += 1
                elif totalhours < self.constrains[i][1]:
                    result[j] += 0.1
                else:
                    result[j] -= 1

            for col in table.T:
                if np.sum(col) > 26:
                    flag = True
                    break

            if flag:
                result[j] = 0
        result = result/10
        return result


    def convertArrary(self, ary):
        string = ''
        for num in ary:
            string += str(num)
        bin2int = int(string, 2)
        return bin2int

    def cal_perference(self, ary):
        bin2int = self.convertArrary(ary)
        if bin2int == 1 or bin2int == 3:
            result = -1
        elif bin2int == 6 or bin2int == 4:
            result = 1
        else:
            result = 0
        return result

    def cal_timeslots(self, ary):
        bin2int = self.convertArrary(ary)
        if bin2int == 0:
            result = 0
        elif bin2int == 1 or bin2int == 4:
            result = 2.5
        elif bin2int == 3 or bin2int == 6:
            result = 3.75
        elif bin2int == 5:
            result = 5
        elif bin2int == 7:
            result = 6.25
        else:
            result = -1
        return result


    def select(self):
        fitness = self.fitness() + 1e-4     # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            random_individual = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, self.students, round(self.students / 2))
            parent[cross_points] = pop[random_individual, cross_points]
        return parent

    def mutate(self, child):
        for point in replace_list:
            if np.random.rand() < self.mutate_rate:
                child.T[point] = abs(child.T[point] - 1)
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


if __name__ == '__main__':
    ga = GA(N=N, chromlen=CHROMOSOME_LENGTH, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, pop_size=POPULATION_SIZE, constrains=constrains)
    for generation in range(GENERATIONS):
        ga.evolve()

    fitness = ga.fitness()
    maxfitness = max(fitness)
    idxn = np.argmax(fitness)
    best = ga.pop[idxn]
    print("fitness")
    print(fitness)
    print("best index: " + str(idxn))
    print("best fitness: " + str(maxfitness))
    print(best)
    for line in best:
        x = 0
        timeslots = np.split(line, 5)
        for ary in timeslots:
            hours = ga.cal_timeslots(ary)
            if hours != -1:
                x += hours
            else:
                x += 1.25
        print(x, end=",")
    print()
    print("-------------------------------------------")
