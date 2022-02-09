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
            if self.matrix[row][row] != Fraction(0, 1):
                for calc_row in range(len(self.matrix)):
                    for calc_col in range(row + 1, len(self.matrix[row])):
                        if calc_row != row:
                            self.matrix[calc_row][calc_col] -= (
                                self.matrix[row][calc_col] * self.matrix[calc_row][row]) / self.matrix[row][row]

                for col in range(len(self.matrix[row])):
                    if col != row:
                        self.matrix[row][col] /= self.matrix[row][row]

                self.matrix[row][row] = Fraction(1, 1)

                for del_row in range(len(self.matrix)):
                    if del_row != row:
                        self.matrix[del_row][row] = Fraction(0, 1)
            else:
                continue

            print()
            self.print_matrix()

        if not self.has_answers():
            print("\nNo answer")
            return

        if self.print_answer() > len(self.matrix[0][:-1]):
            self.print_general_answer()

    def is_one(self, fraction):
        return abs(fraction.numerator) == 1 and abs(fraction.denominator) == 1

    def has_answers(self):
        has_answers = True
        i = 0
        while i < len(self.matrix):
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

            i += 1
        return has_answers

    def print_matrix(self):
        max_lenght = [0 for _ in range(len(self.matrix[0]))]
        for i in range(len(self.matrix[0])):
            for j in range(len(self.matrix)):
                if len(str(self.matrix[j][i])) > max_lenght[i]:
                    max_lenght[i] = len(str(self.matrix[j][i]))

        max_l = max(max_lenght)

        for i, line in enumerate(self.matrix):
            for item in line:
                print(("%" + str(max_l + 2) + "s") % item, end="")
            print()

    def print_answer(self):
        print("\nAnswer:")
        all_number_of_x = 0
        for i in range(len(self.matrix)):
            number_of_x = 0
            for j in range(len(self.matrix[i]) - 1):
                if self.matrix[i][j].numerator:
                    print(f"{' + ' if number_of_x else ''}", end="")

                    print(
                        f"{self.matrix[i][j].neg_sign() if self.is_one(self.matrix[i][j]) else self.matrix[i][j]}x{j + 1}", end="")

                    number_of_x += 1

            print(f" = {self.matrix[i][-1]}")
            all_number_of_x += number_of_x

        return all_number_of_x

    def print_general_answer(self):
        print("\nGeneral answer:")
        simple_x_indexes = dict()
        for i in range(len(self.matrix)):
            number_of_x = 0
            simple_x_indexes[i] = 0
            for j in range(len(self.matrix[i]) - 1):
                if self.matrix[i][j].numerator:
                    if self.is_one(self.matrix[i][j]):
                        simple_x_indexes[i] = j
                    number_of_x += 1

            if number_of_x > 1:
                number_of_x = 0
                print(
                    f"{self.matrix[i][simple_x_indexes[i]].neg_sign()}x{simple_x_indexes[i] + 1} = ", end="")
                for j in range(len(self.matrix[i]) - 1):
                    if self.matrix[i][j].numerator and j != simple_x_indexes[i]:
                        self.matrix[i][j] *= Fraction(-1, 1)
                        print(f"{' + ' if number_of_x else ''}", end="")

                        print(
                            f"{self.matrix[i][j].neg_sign() if self.is_one(self.matrix[i][j]) else self.matrix[i][j]}x{j + 1}", end="")

                        number_of_x += 1

                if self.matrix[i][-1].numerator != 0:
                    print(f" + {self.matrix[i][-1]}")
                else:
                    print()

            else:
                print(
                    f"{self.matrix[i][j].neg_sign()}x{simple_x_indexes[i] + 1} = {self.matrix[i][-1]}")

        any_str = ""
        for j in range(len(self.matrix[0]) - 1):
            if j not in simple_x_indexes.values():
                any_str += f"x{j + 1}, "

        if any_str:
            print(f"{any_str[:-2]} - any")


def main():
    if len(sys.argv) > 1 and sys.argv[1]:
        solver = Solver(sys.argv[1])
    else:
        solver = Solver(input("Input data filename: "))
    solver.solve()


if __name__ == "__main__":
    main()
