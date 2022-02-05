import math


class Fractions:
    def __init__(self, numerator=0, denominator=1):
        self.numerator = numerator
        self.denominator = denominator

    def normalize(self):
        if self.numerator == 0:
            self.denominator = 1
            return

        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator /= gcd
        self.denominator /= gcd

        if self.denominator < 0:
            self.denominator *= -1
            self.numerator *= -1

    def __add__(self, other):
        temp = Fractions()
        temp.numerator = self.numerator * other.denominator \
                         + self.denominator * other.numerator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __sub__(self, other):
        temp = Fractions()
        temp.numerator = self.numerator * other.denominator \
                         - self.denominator * other.denominator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __mul__(self, other):
        temp = Fractions()
        temp.numerator = self.numerator * other.numerator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __truediv__(self, other):
        temp = Fractions()
        temp.numerator = self.numerator * other.denominator
        temp.denominator = self.denominator * other.numerator
        temp.normalize()

        return temp

    def __eq__(self, other):
        if isinstance(other, Fractions):
            if self.numerator == other.numerator \
                    and self.denominator == other.denominator:
                return True
        elif isinstance(other, int):
            if self.numerator == other and self.denominator == 1:
                return True

        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Solver:
    def __init__(self, filename=""):
        if filename == "":
            exit(-1)

        with open(filename, "r", ) as fileIn:
            lines = fileIn.readlines()

        self.matrix = []
        for line in lines:
            self.matrix.append([Fractions(x, 1) for x in map(lambda item: int(item), line.split(" "))])

    def solve(self):
        pass

def main():
    pass


if __name__ == "__main__":
    main()
