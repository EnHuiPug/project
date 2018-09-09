__author__ = 'Sheng XU'

import unittest
import numpy as np

case1 = np.array([[0, 1, 1, 1, 0, 1, 0, 1, 1],
                  [1, 1, 1, 0, 0, 1, 1, 0, 1],
                  [0, 0, 1, 1, 0, 1, 1, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 1, 1],
                  [1, 1, 0, 0, 1, 1, 1, 0, 1]])

case1_fitness_score = 63.5

case2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])

case3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1]])

case4 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1]])

case5 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0]])


def fitness(table, constrains, col_constrain):
    for i in range(5):
        line = table[i]
        totalhours = 0
        perference = []
        timeslots = np.split(line, 5)
        for num in range(5):
            ary = timeslots[num]
            hours = cal_timeslots(ary)
            if hours != -1:
                totalhours += hours
            else:
                totalhours += 1.25
                if result - 10 > 0:
                    result -= 10
                else:
                    result = 0
            perference.append(10 * (num + 1) + cal_perference(ary))
        if constrains[i][2] != 0:
            if constrains[i][2] in perference:
                result += constrains[i][5]
        else:
            result += 1
        if totalhours == constrains[i][4]:
            result += 20
        elif totalhours < constrains[i][4]:
            result += 1
        else:
            if result - 5 > 0:
                result -= 5
            else:
                result = 0
    p = 0
    for col in table.T:
        if np.sum(col) + col_constrain[p] > 26:
            result = 0
        p = p + 1
    return result

# calculate the binary number like 101
def cal_perference(ary):
    bin2int = convertArrary(ary)
    if bin2int == 1 or bin2int == 3:
        result = 2
    elif bin2int == 6 or bin2int == 4:
        result = 1
    else:
        result = 0
    return result

# calculate the total hours of each child by the binary number like 101
def cal_timeslots(ary):
    bin2int = convertArrary(ary)
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

def convertArrary(ary):
    string = ''
    for num in ary:
        string += str(num)
    bin2int = int(string, 2)
    return bin2int


def crossover(parent1, parent2, cross_rate):
    if np.random.rand() < cross_rate:
        random_individual = parent2
        cross_points = [1, 3, 5]
        parent1[cross_points] = parent2[random_individual, cross_points]
    return parent1


def mutate(child, mutate_rate):
    if np.random.rand() < mutate_rate:
        child.T[3] = abs(child.T[3] - 1)
    return child

class TestGA(unittest.TestCase):

    def test_fitness(self):
        self.assertEquals(fitness(case1, constrians, col_constrains), case1_fitness_score)

    def test_convertArrary(self):
        self.assertEquals(convertArrary(np.array([1, 0, 1, 0, 1, 1])), '101011')

    def test_cal_perference(self):
        self.assertEquals(cal_perference(np.array([1, 0, 1])), 0)

    def test_cal_timeslots(self):
        self.assertEquals(cal_timeslots(np.array([1, 0, 1])), 5)

    def test_crossover(self):
        self.assertEquals(crossover(case2, case3), case4)

    def test_mutate(self):
        self.assertEquals(mutate(case2), case5)

if __name__ == '__main__':
    unittest.main()


