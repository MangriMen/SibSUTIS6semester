import copy
from gettext import find
import itertools
import math
from itertools import permutations
from my_fraction import Fraction

z = []
b = []
matrix = []
excluded = []
solution = []
solutions = []
basis = []
m = []
zero = Fraction(0, 1)


def print_matrix(matrix):
    for i in range(0, len(matrix[0])):
        if not(i in excluded):
            if matrix[0][i].numerator < 0:
                print(matrix[0][i], end="   ")
            else:
                print(matrix[0][i], end="    ")
    print()


def printZ():
    print("Z = ", end="")
    print_matrix(z)


def printM():
    print("M = ", end="")
    for i in range(0, len(m)):
        if not(i in excluded):
            if m[i].numerator < 0:
                print(m[i], end="   ")
            else:
                print(m[i], end="    ")
    print()


def print_full_matrix():
    for i in range(0, len(matrix)):
        print("    ", end="")
        for j in range(0, len(matrix[i])):
            if not(j in excluded):
                if matrix[i][j].numerator < 0:
                    print(matrix[i][j], end="   ")
                else:
                    print(matrix[i][j], end="    ")
        print(b[0][i])


def CreateNewMatrix():
    with open("data.txt") as f:
        zcontent = f.readline()
        bcontent = f.readline()
        matrixcontent = f.readlines()
        f.close()

    z.append([Fraction(x)
             for x in map(lambda item: int(item), zcontent.split(" "))])
    for i in range(0, len(z[0])-1):
        z[0][i] *= Fraction(-1, 1)
    b.append([Fraction(x)
             for x in map(lambda item: int(item), bcontent.split(" "))])

    for line in matrixcontent:
        matrix.append([Fraction(x)
                      for x in map(lambda item: int(item), line.split(" "))])

    print("Matrix")
    print_full_matrix()
    printZ()
    N = len(matrix)
    M = len(matrix[0])
    for i in range(0, N):
        for j in range(0, N):
            matrix[i] += [Fraction(0, 1)]
        matrix[i][M+i] = Fraction(1, 1)

    for i in range(0, len(matrix[0])-N):
        solution.append(Fraction(0, 1))
    solutions.append(solution)

    for i in range(0, N):
        basis.append((i, M+i))

    tmp = z[0][len(z[0])-1]
    z[0][len(z[0])-1] = Fraction(0, 1)
    for i in range(0, N):
        z[0] += [Fraction(0, 1)]

    z[0][len(z[0])-1] = tmp

    for j in range(0, M):
        tmp = Fraction(0, 1)
        for i in range(0, N):
            tmp += matrix[i][j]
        m.append(tmp*Fraction(-1, 1))

    for i in range(0, N):
        m.append(Fraction(0, 1))

    tmp = Fraction(0, 1)
    for i in range(0, N):
        tmp += b[0][i]
    m.append(tmp*Fraction(-1, 1))

    print("New matrix")
    print_full_matrix()
    printZ()
    printM()
    print(f"Basis: {basis}")
    print(f"X = {solutions[0]}")
    # , Z = {z[0][len(z[0])-1]}")


def negativeExist(array):
    for i in range(0, len(array)-1):
        if array[i] < Fraction(0, 1):
            return True

    return False


def findNegative(array):
    j = -1
    element = Fraction(1, 1)
    for i in range(0, len(array)-1):
        if array[i] < Fraction(0, 1):
            if array[i] < element:
                element = array[i]
                j = i

    return j


def findSO(j):
    so = Fraction(999, 1)
    i = -1
    for k in range(0, len(matrix)):
        if matrix[k][j] > Fraction(0, 1):
            if b[0][k]/matrix[k][j] < so:
                so = b[0][k]/matrix[k][j]
                i = k

    return i


def findprevious(i):
    for item in basis:
        if item[0] == i:
            return item


def jordan(el):
    i = el[0]
    j = el[1]
    element = matrix[i][j]
    for k in range(0, len(matrix[0])):
        matrix[i][k] /= element
    b[0][i] /= element

    for c in range(0, len(matrix)):
        if c != i:
            newel = matrix[c][j]
            for k in range(0, len(matrix[0])):
                matrix[c][k] -= matrix[i][k]*newel
            b[0][c] -= b[0][i]*newel

    newel = z[0][j]
    for c in range(0, len(z[0])-1):
        z[0][c] -= matrix[i][c] * newel
    z[0][len(z[0])-1] -= b[0][i]*newel

    newel = m[j]
    for c in range(0, len(m)-1):
        m[c] -= matrix[i][c] * newel
    m[len(m)-1] -= b[0][i]*newel


def makesolution(bs):
    N = len(matrix[0]) - len(matrix)
    solution = []
    for i in range(0, N):
        solution.append(Fraction(0, 1))

    for item in bs:
        if item[1] < N:
            solution[item[1]] = b[0][item[0]]

    solutions.append(solution)


def checkM():
    for i in range(0, len(m) - len(matrix) - 1):
        if m[i] != Fraction(0, 1):
            return False

    return True


def SolveByM():
    while 1:
        if negativeExist(m):
            j = findNegative(m)
            i = findSO(j)
            bas = findprevious(i)
            basis.remove(bas)
            excluded.append(bas[1])
            basis.append((i, j))
            basis.sort()
            jordan((i, j))
            print("\nMatrix")
            print_full_matrix()
            printZ()
            printM()
            print(f"Basis: {basis}")
            makesolution(basis)
            print(f"X = {solutions[len(solutions)-1]}")
        else:
            if checkM() == True:
                return True
            else:
                return False


def findNegativeZ(array):
    j = -1
    element = Fraction(1, 1)
    for i in range(0, len(array[0])-len(matrix)-1):
        if array[0][i] < Fraction(0, 1):
            if array[0][i] < element:
                element = array[0][i]
                j = i

    return j


def negativeExistZ(array):
    for i in range(0, len(array[0])-len(matrix)-1):
        if array[0][i] < Fraction(0, 1):
            return True

    return False


def included(i):
    for item in basis:
        if item[1] == i:
            return True

    return False


def doubleCheck():
    pj = -1
    for i in range(0, len(z[0])-len(matrix)-1):
        if z[0][i] == Fraction(0, 1) and not(included(i)):
            pj = i

    return pj


def table():
    for item in solutions:
        print(item)


def SolveByZ():
    print("\nMatrix")
    print_full_matrix()
    printZ()
    print(f"Basis: {basis}")
    print(f"X = {solutions[len(solutions)-1]}")
    while 1:
        if negativeExistZ(z):
            j = findNegativeZ(z)
            i = findSO(j)
            bas = findprevious(i)
            basis.remove(bas)
            basis.append((i, j))
            basis.sort()
            jordan((i, j))
            print("\nMatrix")
            print_full_matrix()
            printZ()
            print(f"Basis: {basis}")
            makesolution(basis)
            print(f"X = {solutions[len(solutions)-1]}")
        else:
            j = doubleCheck()
            if j == -1:
                print(f"Z = {z[0][len(z[0])-1]}")
                # table()
                break
            i = findSO(j)
            bas = findprevious(i)
            basis.remove(bas)
            basis.append((i, j))
            basis.sort()
            jordan((i, j))
            print("\nMatrix")
            print_full_matrix()
            printZ()
            print(f"Basis: {basis}")
            makesolution(basis)
            print(f"X = {solutions[len(solutions)-1]}")
            sol = []
            for i in range(0, len(solutions[0])):
                sol.append(
                    Fraction(-1, 1)*solutions[len(solutions)-2][i] + solutions[len(solutions)-1][i])
            print("Solution = [", end="")
            for i in range(0, len(sol)):
                if sol[i] > Fraction(0, 1):
                    print(
                        f"{solutions[len(solutions)-2][i]} + {sol[i]}k", end="")
                else:
                    print(
                        f"{solutions[len(solutions)-2][i]} - {Fraction(-1,1)*sol[i]}k", end="")
                if i < len(sol) - 1:
                    print("; ", end="")
            print("]")
            print(f"Z = {z[0][len(z[0])-1]}")
            break


def Solve() -> None:
    if SolveByM() == False:
        print("Constraint system is inconsistent")
        return
    SolveByZ()


def main():
    CreateNewMatrix()
    Solve()


if __name__ == "__main__":
    main()
