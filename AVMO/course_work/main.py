from traceback import StackSummary
from main import CreateNewMatrix
from my_fraction import Fraction

isCompact = True


def getBasisIndexFromRow(matrix: list, excluded: list, row: int) -> int:
    """
    Returns
    -------
    int
        index of the column containing the basis element
    """
    index = [j for j, el in enumerate(
        matrix[row]) if el == 1 and j not in excluded]
    basis_index = -1

    for ind in index:
        isBasis = True
        for i, _ in enumerate(matrix):
            if i != row:
                if matrix[i][ind] != 0:
                    isBasis = False
        if isBasis:
            basis_index = ind
            break

    return basis_index


def getXFromIndex(index):
    if index < 0:
        return ""
    return f"x{index + 1}"


def createLineSplitter(cell_lenght, count, side=""):
    delimeter = "┼"

    if side == "top":
        delimeter = "┬"
    elif side == "bottom":
        delimeter = "┴"

    left_segment = delimeter + "─" * cell_lenght + delimeter
    next_segment = ("─" * cell_lenght + delimeter) * (count - 1)

    return left_segment + next_segment


def createCell(value="", width=1, isFirstLeft=False):
    cell = ("│" if isFirstLeft else "") +\
        '{:>{}}'.format(str(value), int(max(width - 1, 0))) + " │"
    return cell


def createHeaderRow(matrix, excluded, cell_width):
    out = createCell("б.п.", cell_width, True) + createCell("1", cell_width)
    for x_index in range(0, len(matrix[0])):
        if x_index not in excluded:
            out += createCell(f"x{x_index+1}", cell_width)
    return out


def createTableRows(matrix, solution, excluded, cell_width, column_count):
    rows = ""

    for i, _ in enumerate(matrix):
        rows += createCell(getXFromIndex(
            getBasisIndexFromRow(matrix, excluded, i)), cell_width, True)
        rows += createCell(solution[i], cell_width)
        for j in range(0, len(matrix[i])):
            if not (j in excluded):
                rows += createCell(matrix[i][j], cell_width)
        rows += "\n"
        if not isCompact:
            rows += createLineSplitter(cell_width, column_count) + "\n"

    return rows


def createFunctionRow(functionZ, excluded, cell_width, name="F"):
    out = createCell(name, cell_width, True) + createCell(
        functionZ[-1], cell_width)
    for i, x in enumerate(functionZ[:-1]):
        if i not in excluded:
            out += createCell(x, cell_width)
    return out


def getCellWidth(matrix, solution, functionZ, functionM, excluded=[]):
    width = 0

    if solution:
        width = max(width, len(str(max(solution, key=lambda x: len(str(x))))))
    if functionZ:
        width = max(width, len(str(max(functionZ, key=lambda x: len(str(x))))))
    if functionM:
        width = max(width, len(str(max(functionM, key=lambda x: len(str(x))))))

    for i, _ in enumerate(matrix):
        for j in range(0, len(matrix[i])):
            if j not in excluded:
                width = max(width, len(str(matrix[i][j])))

    return width


class DualSimplexMethod:
    def __init__(self, filename="") -> None:
        self.matrix = []
        self.functionZ = []
        self.function_m = []
        self.free_members = []

        self.excluded = []
        self.solution = []
        self.solutions = []
        self.basis = []

        self.loadFromFile(filename)

    def printFullMatrix(self, title=""):
        if title:
            print(f" {title}")

        cell_width = max(5, getCellWidth(
            self.matrix, self.free_members, self.functionZ, self.function_m, self.excluded)) + 1
        column_count = len(self.matrix[0]) + 2 - len(self.excluded)

        print(createLineSplitter(cell_width, column_count, "top"))
        print(createHeaderRow(self.matrix, self.excluded, cell_width))
        print(createLineSplitter(cell_width, column_count))

        print(createTableRows(self.matrix, self.free_members, self.excluded,
                              cell_width, column_count), end='')

        print(createFunctionRow(self.functionZ, self.excluded, cell_width, "Z"))

        if self.function_m and [x for i, x in enumerate(self.function_m) if x != 0 and i not in self.excluded]:
            if not isCompact:
                print(createLineSplitter(cell_width, column_count))
            print(createFunctionRow(self.function_m,
                  self.excluded, cell_width, "M"))
            print(createLineSplitter(cell_width, column_count, "bottom"))
        else:
            print(createLineSplitter(cell_width, column_count, "bottom"))

    def loadFromFile(self, filename):
        with open(filename, "r") as fileIn:
            function_line = fileIn.readline()
            b_line = fileIn.readline()
            matrix_lines = fileIn.readlines()

        self.functionZ = [Fraction(int(x) * -1)
                          for x in function_line.split(" ")]
        self.free_members = [Fraction(int(x)) for x in b_line.split(" ")]

        for matrix_line in matrix_lines:
            self.matrix.append([Fraction(int(x))
                               for x in matrix_line.split(" ")])

        self.printFullMatrix()

        N = len(self.matrix)
        M = len(self.matrix[0])
        for i in range(0, N):
            for j in range(0, N):
                self.matrix[i] += [Fraction(0, 1)]
            self.matrix[i][M+i] = Fraction(1, 1)

        for i in range(0, len(self.matrix[0])-N):
            self.solution.append(Fraction(0, 1))
        self.solutions.append(self.solution)

        for i in range(0, N):
            self.basis.append((i, M+i))

        tmp = self.functionZ[len(self.functionZ)-1]
        self.functionZ[len(self.functionZ)-1] = Fraction(0, 1)
        for i in range(0, N):
            self.functionZ += [Fraction(0, 1)]

        self.functionZ[len(self.functionZ)-1] = tmp

        for j in range(0, M):
            tmp = Fraction(0, 1)
            for i in range(0, N):
                tmp += self.matrix[i][j]
            self.function_m.append(tmp*Fraction(-1, 1))

        for i in range(0, N):
            self.function_m.append(Fraction(0, 1))

        tmp = Fraction(0, 1)
        for i in range(0, N):
            tmp += self.free_members[i]
        self.function_m.append(tmp*Fraction(-1, 1))

        self.printFullMatrix()
        print(f"X = {self.solutions[0]}")

    @ staticmethod
    def negativeExist(array):
        for i in range(0, len(array)-1):
            if array[i] < Fraction(0, 1):
                return True

        return False

    @ staticmethod
    def findNegative(array):
        j = -1
        element = Fraction(1, 1)
        for i in range(0, len(array)-1):
            if array[i] < Fraction(0, 1):
                if array[i] < element:
                    element = array[i]
                    j = i

        return j

    def findSO(self, j):
        so = Fraction(999, 1)
        i = -1
        for k, _ in enumerate(self.matrix):
            if self.matrix[k][j] > Fraction(0, 1):
                if self.free_members[k]/self.matrix[k][j] < so:
                    so = self.free_members[k]/self.matrix[k][j]
                    i = k

        return i

    def findprevious(self, i):
        for item in self.basis:
            if item[0] == i:
                return item

    def jordan(self, el):
        i = el[0]
        j = el[1]
        element = self.matrix[i][j]
        for k in range(0, len(self.matrix[0])):
            self.matrix[i][k] /= element
        self.free_members[i] /= element

        for c, _ in enumerate(self.matrix):
            if c != i:
                newel = self.matrix[c][j]
                for k in range(0, len(self.matrix[0])):
                    self.matrix[c][k] -= self.matrix[i][k]*newel
                self.free_members[c] -= self.free_members[i]*newel

        newel = self.functionZ[j]
        for c in range(0, len(self.functionZ)-1):
            self.functionZ[c] -= self.matrix[i][c] * newel
        self.functionZ[len(self.functionZ)-1] -= self.free_members[i]*newel

        newel = self.function_m[j]
        for c in range(0, len(self.function_m)-1):
            self.function_m[c] -= self.matrix[i][c] * newel
        self.function_m[len(self.function_m)-1] -= self.free_members[i]*newel

    def makesolution(self, bs):
        N = len(self.matrix[0]) - len(self.matrix)
        solution = []
        for i in range(0, N):
            solution.append(Fraction(0, 1))

        for item in bs:
            if item[1] < N:
                solution[item[1]] = self.free_members[item[0]]

        self.solutions.append(solution)

    def checkM(self):
        for i in range(0, len(self.function_m) - len(self.matrix) - 1):
            if self.function_m[i] != Fraction(0, 1):
                return False

        return True

    def solveByM(self):
        while 1:
            if self.negativeExist(self.function_m):
                j = self.findNegative(self.function_m)
                i = self.findSO(j)
                bas = self.findprevious(i)
                self.basis.remove(bas)
                self.excluded.append(bas[1])
                self.basis.append((i, j))
                self.basis.sort()
                self.jordan((i, j))
                self.printFullMatrix()

                self.makesolution(self.basis)
                print(f"X = {self.solutions[len(self.solutions)-1]}")
            else:
                return self.checkM()

    def findNegativeZ(self, array):
        j = -1
        element = Fraction(1, 1)
        for i in range(0, len(array)-len(self.matrix)-1):
            if array[i] < Fraction(0, 1):
                if array[i] < element:
                    element = array[i]
                    j = i

        return j

    def negativeExistZ(self, array):
        for i in range(0, len(array)-len(self.matrix)-1):
            if array[i] < Fraction(0, 1):
                return True

        return False

    def included(self, i):
        for item in self.basis:
            if item[1] == i:
                return True

        return False

    def doubleCheck(self):
        pj = -1
        for i in range(0, len(self.functionZ)-len(self.matrix)-1):
            if self.functionZ[i] == Fraction(0, 1) and not(self.included(i)):
                pj = i

        return pj

    def solveByZ(self):
        self.printFullMatrix()
        print(f"X = {self.solutions[len(self.solutions)-1]}")
        while 1:
            if self.negativeExistZ(self.functionZ):
                j = self.findNegativeZ(self.functionZ)
                i = self.findSO(j)
                bas = self.findprevious(i)
                self.basis.remove(bas)
                self.basis.append((i, j))
                self.basis.sort()
                self.jordan((i, j))

                self.printFullMatrix()

                self.makesolution(self.basis)
                print(f"X = {self.solutions[len(self.solutions)-1]}")
            else:
                j = self.doubleCheck()
                if j == -1:
                    print(
                        f"Z = {self.functionZ[len(self.functionZ)-1]}")
                    break
                i = self.findSO(j)
                bas = self.findprevious(i)
                self.basis.remove(bas)
                self.basis.append((i, j))
                self.basis.sort()
                self.jordan((i, j))

                self.printFullMatrix()

                self.makesolution(self.basis)
                print(f"X = {self.solutions[len(self.solutions)-1]}")
                sol = []
                for i in range(0, len(self.solutions[0])):
                    sol.append(
                        Fraction(-1, 1)*self.solutions[len(self.solutions)-2][i] + self.solutions[len(self.solutions)-1][i])
                print("Solution = [", end="")
                for i, _ in enumerate(sol):
                    if sol[i] > Fraction(0, 1):
                        print(
                            f"{self.solutions[len(self.solutions)-2][i]} + {sol[i]}λ", end="")
                    else:
                        print(
                            f"{self.solutions[len(self.solutions)-2][i]} - {Fraction(-1,1)*sol[i]}λ", end="")
                    if i < len(sol) - 1:
                        print("; ", end="")
                print("]")
                print(f"Z = {self.functionZ[len(self.functionZ)-1]}")
                break

    def solve(self) -> None:
        if not self.solveByM():
            print("Constraint system is inconsistent")
            return
        self.solveByZ()


def main():
    DualSimplexMethodSolver = DualSimplexMethod("data.txt")
    DualSimplexMethodSolver.solve()


if __name__ == "__main__":
    main()
