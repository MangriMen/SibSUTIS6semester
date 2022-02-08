import sys
from my_fraction import Fraction


class Solver:
    def __init__(self, filename=""):
        if filename == "":
            print("Error. Enter filename.")
            exit(-1)

        try:
            with open(filename, "r") as fileIn:
                lines = fileIn.readlines()
        except FileNotFoundError:
            print(f"Error. {filename} not found")
            exit(-1)

        self.matrix = []
        for line in lines:
            self.matrix.append([Fraction(x, 1) for x in map(
                lambda item: int(item), line.split(" "))])

    def solve(self):
        self.print_matrix()

        for row in range(len(self.matrix)):
            nuls = False
            if self.matrix[row][row] == 0:
                nuls = True
                for item in range(row + 1, len(self.matrix)):
                    if self.matrix[item][item] != 0:
                        self.matrix[item][item], self.matrix[row][row] = self.matrix[row][row], self.matrix[item][item]
                        nuls = False

            if nuls:
                continue

            for col in range(len(self.matrix[row])):
                if col == row:
                    continue
                self.matrix[row][col] /= self.matrix[row][row]

            self.matrix[row][row] = Fraction(1, 1)

            for i in range(len(self.matrix)):
                if i == row:
                    continue
                factor = self.matrix[i][row] / self.matrix[row][row]

                for col in range(row, len(self.matrix[i])):
                    self.matrix[i][col] -= self.matrix[row][col] * factor
                    self.matrix[i][col].normalize()

            print()
            self.print_matrix()

        if not self.has_answers():
            print("\nNo answer")
            return

        self.print_answer()

    def is_one(self, fraction):
        return abs(fraction.numerator) == 1 and abs(fraction.denominator) == 1

    def has_answers(self):
        has_answers = True
        for i in range(len(self.matrix)):
            max_el = max(self.matrix[i][:-1], key=lambda p: p.numerator)
            min_el = min(self.matrix[i][:-1], key=lambda p: p.numerator)

            if max_el.numerator == 0 and min_el.numerator == 0:
                if self.matrix[i][-1] == 0:
                    del self.matrix[i]
                    i -= 1
                    print()
                    self.print_matrix()
                else:
                    has_answers = False
        return has_answers

    def print_matrix(self):
        for line in self.matrix:
            for item in line:
                print(str(item) + " ", end="")
            print()

    def print_answer(self):
        print("\nAnswer:")
        for i in range(len(self.matrix)):
            number_of_x = 0
            for j in range(len(self.matrix[i]) - 1):
                if self.matrix[i][j].numerator:
                    print(f"{' + ' if number_of_x else ''}", end="")

                    print(
                        f"{self.matrix[i][j].neg_sign() if self.is_one(self.matrix[i][j]) else self.matrix[i][j]}x{j + 1}", end="")

                    number_of_x += 1

            print(f" = {self.matrix[i][-1]}")


def main():
    if len(sys.argv) > 1 and sys.argv[1]:
        solver = Solver(sys.argv[1])
    else:
        solver = Solver(input("Input data filename: "))
    solver.solve()


if __name__ == "__main__":
    main()
