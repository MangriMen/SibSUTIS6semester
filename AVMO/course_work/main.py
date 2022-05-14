from array import array
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
        self.functionZ = []
        self.function_m = []
        self.free_members = []

        self.excluded = []
        self.solution = []
        self.solutions = []
        self.basis = []

        self.loadFromFile(filename)

    def __printSimplexTable(self, title=""):
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

    def __printSolution(self):
        print(f"X = {self.solutions[-1]}")

    def __readFromFile(self, filename):
        with open(filename, "r") as fileIn:
            function_line = fileIn.readline()
            b_line = fileIn.readline()
            matrix_lines = fileIn.readlines()

        self.functionZ = [Fraction(int(x) * -1)
                          for x in function_line.split(" ")]
        self.functionZ[-1] *= -1
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

    def __findSO(self, j):
        so = Fraction(999, 1)
        i = -1
        for k, _ in enumerate(self.matrix):
            if self.matrix[k][j] > Fraction(0, 1):
                if self.free_members[k]/self.matrix[k][j] < so:
                    so = self.free_members[k]/self.matrix[k][j]
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

        newel = self.functionZ[j]
        for c in range(0, len(self.functionZ)-1):
            self.functionZ[c] -= self.matrix[i][c] * newel
        self.functionZ[len(self.functionZ)-1] -= self.free_members[i]*newel

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
        for i in range(0, len(self.functionZ)-len(self.matrix)-1):
            if self.functionZ[i] == Fraction(0, 1) and not(self.__included(i)):
                pj = i

        return pj

    def __changeBasis(self, i, j, toExclude=False):
        bas = self.__findprevious(i)

        if toExclude:
            self.excluded.append(bas[1])

        self.basis.remove(bas)
        self.basis.append((i, j))
        self.basis.sort()

    def __step(self, i, j, isExcludeBasis=False):
        self.__changeBasis(i, j, isExcludeBasis)
        self.__jordan((i, j))
        self.__makesolution(self.basis)

        self.__printSimplexTable()
        self.__printSolution()

    def __solveByM(self):
        while True:
            if self.__negativeExist(self.function_m):
                j = self.__findNegative(self.function_m)
                i = self.__findSO(j)

                self.__step(i, j, True)
            else:
                return self.__checkM()

    def __solveByZ(self):
        while True:
            if self.__negativeExist(self.functionZ, True):
                j = self.__findNegative(self.functionZ, True)
                i = self.__findSO(j)

                bas = self.__findprevious(i)
                self.basis.remove(bas)
                self.basis.append((i, j))
                self.basis.sort()
                self.__jordan((i, j))

                self.__printSimplexTable()

                self.__makesolution(self.basis)
                self.__printSolution()
            else:
                if (j := self.__doubleCheck()) == -1:
                    print(f"Z = {self.functionZ[-1]}")
                    break
                i = self.__findSO(j)

                bas = self.__findprevious(i)
                self.basis.remove(bas)
                self.basis.append((i, j))
                self.basis.sort()
                self.__jordan((i, j))

                self.__printSimplexTable()

                self.__makesolution(self.basis)
                print(f"X = {self.solutions[len(self.solutions)-1]}")

                sol = []
                for i in range(0, len(self.solutions[0])):
                    sol.append(Fraction(-1, 1) * self.solutions[len(
                        self.solutions)-2][i] + self.solutions[len(self.solutions)-1][i])

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
        if not self.__solveByM():
            print("Constraint system is inconsistent")
            return
        self.__solveByZ()


def main() -> None:
    DualSimplexMethodSolver = DualSimplexMethod(
        input("Enter filename: ") or "data.txt")
    DualSimplexMethodSolver.solve()


if __name__ == "__main__":
    main()
