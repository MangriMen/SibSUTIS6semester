import math


class Fraction:
    def __init__(self, numerator=0, denominator=1):
        self.numerator = int(numerator)
        self.denominator = int(denominator)

    def normalize(self):
        if self.numerator == 0:
            self.denominator = 1
            return

        gcd = math.gcd(int(self.numerator), int(self.denominator))
        self.numerator /= gcd
        self.denominator /= gcd

        if self.denominator < 0:
            self.denominator *= -1
            self.numerator *= -1

    def sign(self):
        if self.numerator < 0:
            return "-"

        return "+"

    def __add__(self, other):
        temp = Fraction()
        temp.numerator = self.numerator * other.denominator \
            + self.denominator * other.numerator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __sub__(self, other):
        temp = Fraction()
        temp.numerator = self.numerator * other.denominator \
            - self.denominator * other.numerator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __mul__(self, other):
        temp = Fraction()
        temp.numerator = self.numerator * other.numerator
        temp.denominator = self.denominator * other.denominator
        temp.normalize()

        return temp

    def __truediv__(self, other):
        temp = Fraction()
        temp.numerator = self.numerator * other.denominator
        temp.denominator = self.denominator * other.numerator
        temp.normalize()

        return temp

    def __eq__(self, other):
        if isinstance(other, Fraction):
            if self.numerator == other.numerator \
                    and self.denominator == other.denominator:
                return True
        elif isinstance(other, int):
            if self.numerator == other and self.denominator == 1:
                return True

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.denominator == 1:
            return f"{int(self.numerator)}"
        return f"{int(self.numerator)}/{int(self.denominator)}"

    def __repr__(self):
        if self.denominator == 1:
            return f"{int(self.numerator)}"
        return f"{int(self.numerator)}/{int(self.denominator)}"
