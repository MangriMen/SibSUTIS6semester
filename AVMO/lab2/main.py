import copy
import itertools
import math
from re import A
import sys
from itertools import permutations
from tkinter.messagebox import NO
from turtle import RawPen
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

        variables_count = len(self.matrix[0][:-1])
        rang = len(self.matrix)
        variants_count = int(math.factorial(
            variables_count) / (math.factorial(rang) * math.factorial(variables_count - rang)))

        print("\nAll possible inclusions of variables:")
        numbers = ''.join([str(x) for x in range(len(self.matrix[0][:-1]))])
        variants_count_list = [[int(x) for x in tup]
                               for tup in itertools.combinations(numbers, rang)]

        for arr in variants_count_list:
            print(
                ''.join(map(lambda item: f"x{item}", [str(x+1) for x in arr])))

        if len(variants_count_list) > variants_count:
            raise RuntimeError("Practice variants count no equals teoretical")

        print()

        for inclusion in variants_count_list:
            copy_matrix = copy.deepcopy(self.matrix)
            print(
                ''.join(map(lambda item: f"x{item}", [str(x+1) for x in inclusion])))

            is_no_answer = False
            for row in range(len(copy_matrix)):
                result = all(
                    element == copy_matrix[row][0] for element in [copy_matrix[row][x] for x in inclusion])
                if result:
                    print("Answer: ∅\n")
                    is_no_answer = True
                    break

            if is_no_answer:
                continue

            need_to_recalculate = True
            free_cash = [0 for _ in range(len(copy_matrix))]
            for col in inclusion:
                for row in range(len(copy_matrix)):
                    free_cash[row] |= (1 if copy_matrix[row]
                                       [col] == 1 else 0)

            print()
            self.print_matrix(copy_matrix)

            while need_to_recalculate:
                counter_row = 0
                for col in inclusion:
                    need_to_recalculate = False
                    for row in range(len(copy_matrix)):
                        if copy_matrix[row][col] != 1 and copy_matrix[row][col] != 0:
                            need_to_recalculate = True
                            try:
                                counter_row = free_cash.index(0)
                            except:
                                break
                            break
                    if need_to_recalculate:
                        if counter_row >= len(copy_matrix):
                            counter_row = 0
                        for calc_row in range(len(copy_matrix)):
                            for calc_col in range(col + 1, len(copy_matrix[counter_row])):
                                if calc_row != counter_row:
                                    copy_matrix[calc_row][calc_col] -= (
                                        copy_matrix[counter_row][calc_col] * copy_matrix[calc_row][col]) / copy_matrix[counter_row][col]

                        for col_in in range(len(copy_matrix[counter_row])):
                            if col_in != col:
                                copy_matrix[counter_row][col_in] /= copy_matrix[counter_row][col]

                        copy_matrix[counter_row][col] = Fraction(1, 1)

                        for del_row in range(len(copy_matrix)):
                            if del_row != counter_row:
                                copy_matrix[del_row][col] = Fraction(
                                    0, 1)

                        free_cash[counter_row] = 1

                        print()
                        self.print_matrix(copy_matrix)

            ans_string = "("
            for col in range(len(copy_matrix[0][:-1])):
                if col in inclusion:
                    for row in range(len(copy_matrix)):
                        if copy_matrix[row][col] == 1:
                            ans_string += str(copy_matrix[row][-1]) + ";"
                else:
                    ans_string += "0;"
            ans_string = ans_string[:-1] + ")"

            print(f"Answer: {ans_string}")

            print()

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

    def print_matrix(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        max_lenght = [0 for _ in range(len(matrix[0]))]
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                if len(str(matrix[j][i])) > max_lenght[i]:
                    max_lenght[i] = len(str(matrix[j][i]))

        max_l = max(max_lenght)

        for i, line in enumerate(matrix):
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
        copy_matrix = copy.deepcopy(self.matrix)
        print("\nGeneral answer:")
        simple_x_indexes = dict()
        for i in range(len(copy_matrix)):
            number_of_x = 0
            simple_x_indexes[i] = 0
            for j in range(len(copy_matrix[i]) - 1):
                if copy_matrix[i][j].numerator:
                    if self.is_one(copy_matrix[i][j]):
                        simple_x_indexes[i] = j
                    number_of_x += 1

            if number_of_x > 1:
                number_of_x = 0
                print(
                    f"{copy_matrix[i][simple_x_indexes[i]].neg_sign()}x{simple_x_indexes[i] + 1} = ", end="")
                for j in range(len(copy_matrix[i]) - 1):
                    if copy_matrix[i][j].numerator and j != simple_x_indexes[i]:
                        copy_matrix[i][j] *= Fraction(-1, 1)
                        print(f"{' + ' if number_of_x else ''}", end="")

                        print(
                            f"{copy_matrix[i][j].neg_sign() if self.is_one(copy_matrix[i][j]) else copy_matrix[i][j]}x{j + 1}", end="")

                        number_of_x += 1

                if copy_matrix[i][-1].numerator != 0:
                    print(f" + {copy_matrix[i][-1]}")
                else:
                    print()

            else:
                print(
                    f"{copy_matrix[i][j].neg_sign()}x{simple_x_indexes[i] + 1} = {copy_matrix[i][-1]}")

        any_str = ""
        for j in range(len(copy_matrix[0]) - 1):
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
