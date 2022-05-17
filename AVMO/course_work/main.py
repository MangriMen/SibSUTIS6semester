from array import array
from pprint import pprint
from tkinter.messagebox import NO
from my_fraction import Fraction

isCompact = True


def getBasisIndexFromRow(matrix: list, excluded: list, row: int) -> int:
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
        self.function_z = []
        self.function_m = []
        self.free_members = []

        self.excluded = []
        self.solution = []
        self.solutions = []
        self.solutionsZ = []
        self.basis = []

        self.loadFromFile(filename)

    def __printSimplexTable(self, title=""):
        if title:
            print(f" {title}")

        cell_width = max(5, getCellWidth(
            self.matrix, self.free_members, self.function_z, self.function_m, self.excluded)) + 1
        column_count = len(self.matrix[0]) + 2 - len(self.excluded)

        print(createLineSplitter(cell_width, column_count, "top"))
        print(createHeaderRow(self.matrix, self.excluded, cell_width))
        print(createLineSplitter(cell_width, column_count))

        print(createTableRows(self.matrix, self.free_members, self.excluded,
                              cell_width, column_count), end='')

        print(createFunctionRow(self.function_z, self.excluded, cell_width, "Z"))

        if self.function_m and [x for i, x in enumerate(self.function_m) if x != 0 and i not in self.excluded]:
            if not isCompact:
                print(createLineSplitter(cell_width, column_count))
            print(createFunctionRow(self.function_m,
                  self.excluded, cell_width, "M"))

        print(createLineSplitter(cell_width, column_count, "bottom"))

    def __printSolution(self):
        print(f"X = {self.solutions[-1]}")

    def __readFromFile(self, filename):
        with open(filename, "r") as fileIn:
            function_line = fileIn.readline()
            b_line = fileIn.readline()
            matrix_lines = fileIn.readlines()

        self.function_z = [Fraction(int(x) * -1)
                           for x in function_line.split(" ")]
        self.function_z[-1] *= -1
        self.free_members = [Fraction(int(x)) for x in b_line.split(" ")]
        self.matrix = [[Fraction(int(x)) for x in matrix_line.split(" ")]
                       for matrix_line in matrix_lines]

    def __complementToTheBasis(self):
        N = len(self.matrix)
        M = len(self.matrix[0])

        for i in range(0, N):
            for j in range(0, N):
                self.matrix[i] += [Fraction(0, 1)]
            self.matrix[i][M+i] = Fraction(1, 1)

        self.solution = [Fraction(0, 1)
                         for _ in range(len(self.matrix[0]) - N)]
        self.solutions.append(self.solution)

        self.basis = [(i, M + i) for i in range(N)]
        self.function_z = [*self.function_z[:-1], *
                           [Fraction(0, 1) for _ in range(N)], self.function_z[-1]]

        for j in range(0, M):
            tmp = Fraction(0, 1)
            for i in range(0, N):
                tmp += self.matrix[i][j]
            self.function_m.append(tmp * Fraction(-1, 1))

        self.function_m.extend([Fraction(0, 1) for _ in range(N)])

        self.function_m.append(
            sum(self.free_members, Fraction(0, 0)) * Fraction(-1, 1))

    def loadFromFile(self, filename):
        self.__readFromFile(filename)
        self.__printSimplexTable("Original matrix")

        self.__complementToTheBasis()
        self.__printSimplexTable("Complement to the basis")
        self.__printSolution()

    def __findNegative(self, array, isCutRange=False):
        j = -1
        element = Fraction(1, 1)
        for i in range(0, len(array) - (len(self.matrix) if isCutRange else 0) - 1):
            if array[i] < Fraction(0, 1):
                if array[i] < element:
                    element = array[i]
                    j = i

        return j

    def __negativeExist(self, array, isCutRange=False):
        for i in range(0, len(array) - (len(self.matrix) if isCutRange else 0) - 1):
            if array[i] < Fraction(0, 1):
                return True

        return False

    def __findSR(self, j):
        sr = Fraction(999, 1)
        i = -1
        for k, _ in enumerate(self.matrix):
            if self.matrix[k][j] > Fraction(0, 1):
                if (calcSR := self.free_members[k]/self.matrix[k][j]) < sr:
                    sr = calcSR
                    i = k

        return i

    def __findprevious(self, i):
        for item in self.basis:
            if item[0] == i:
                return item

    def __jordan(self, el):
        (i, j) = el

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

        newel = self.function_z[j]
        for c in range(0, len(self.function_z)-1):
            self.function_z[c] -= self.matrix[i][c] * newel
        self.function_z[len(self.function_z)-1] -= self.free_members[i]*newel

        newel = self.function_m[j]
        for c in range(0, len(self.function_m)-1):
            self.function_m[c] -= self.matrix[i][c] * newel
        self.function_m[len(self.function_m)-1] -= self.free_members[i]*newel

    def __makesolution(self, bs):
        N = len(self.matrix[0]) - len(self.matrix)
        solution = []
        for i in range(0, N):
            solution.append(Fraction(0, 1))

        for item in bs:
            if item[1] < N:
                solution[item[1]] = self.free_members[item[0]]

        self.solutions.append(solution)

    def __checkM(self):
        for i in range(0, len(self.function_m) - len(self.matrix) - 1):
            if self.function_m[i] != 0:
                return False
        return True

    def __included(self, i):
        for item in self.basis:
            if item[1] == i:
                return True

        return False

    def __doubleCheck(self):
        pj = -1
        for i in range(0, len(self.function_z)-len(self.matrix)-1):
            if self.function_z[i] == Fraction(0, 1) and not(self.__included(i)):
                pj = i

        return pj

    def __printZSolution(self):
        print(f"Z = {self.function_z[-1]}")

    def __printMultipleSolutions(self):
        sol = [(Fraction(-1, 1) * self.solutions[-2][i]) + self.solutions[-1][i]
               for i, _ in enumerate(self.solutions[0])]

        print("Solution = [", end="")
        for i, _ in enumerate(sol):
            print(
                f"{self.solutions[-2][i]} {'+' if sol[i] > Fraction(0, 1) else '-'} {abs(sol[i])}λ", end="")
            if i < len(sol) - 1:
                print("; ", end="")
        print("]")

    def __print_simplex_table_with_z(self):
        cell_width = len(self.solutions[0]) * 5
        print(createLineSplitter(cell_width, 2, "top"))
        for solution, z_value in zip(self.solutions, self.solutionsZ):
            out = ""
            out += createCell(solution, cell_width, True)
            out += createCell(z_value, cell_width)
            print(out)
        print(createLineSplitter(cell_width, 2, "bottom"))

    def __changeBasis(self, i, j, toExclude=False):
        bas = self.__findprevious(i)

        if toExclude:
            self.excluded.append(bas[1])

        self.basis.remove(bas)
        self.basis.append((i, j))
        self.basis.sort()

    def __step(self, i, j, isExcludeBasis=False):
        self.solutionsZ.append(self.function_z[-1])
        self.__changeBasis(i, j, isExcludeBasis)
        self.__jordan((i, j))
        self.__makesolution(self.basis)

        self.__printSimplexTable()
        self.__printSolution()

    def __solveByM(self):
        print("\n---Solve by function M---\n")
        while True:
            if self.__negativeExist(self.function_m):
                j = self.__findNegative(self.function_m)
                if (i := self.__findSR(j)) == -1:
                    return False

                self.__step(i, j, True)
            else:
                return self.__checkM()

    def __solveByZ(self):
        print("\n---Solve by function Z---\n")
        while True:
            if self.__negativeExist(self.function_z, True):
                j = self.__findNegative(self.function_z, True)
                if (i := self.__findSR(j)) == -1:
                    return False

                self.__step(i, j, False)
            else:
                if (j := self.__doubleCheck()) == -1:
                    self.__printZSolution()
                    self.solutionsZ.append(self.function_z[-1])
                    break
                if (i := self.__findSR(j)) == -1:
                    return False

                self.__step(i, j, False)

                self.__printMultipleSolutions()
                self.__printZSolution()
                break

    def solve(self) -> None:
        if not self.__solveByM():
            print("\n---Constraint system is inconsistent---")
            return
        if not self.__solveByZ():
            print("\n---Constraint system is inconsistent---")
            return

        try:
            self.__print_simplex_table_with_z()
        except:
            pass


def main() -> None:
    DualSimplexMethodSolver = DualSimplexMethod(
        input("Enter filename(default \"data.txt\"): ") or "data.txt")
    DualSimplexMethodSolver.solve()


if __name__ == "__main__":
    main()
