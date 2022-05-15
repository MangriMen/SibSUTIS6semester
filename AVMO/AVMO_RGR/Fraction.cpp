#include "Fraction.h"

long long gcd(long long a, long long b) {
    while (b > 0) {
        long long c = a % b;
        a = b;
        b = c;
    }

    return a;
}

long long lcm(long long a, long long b) {
    return (a * b) / gcd(a, b);
}

Fraction summarize(Fraction a, Fraction b) {
    long long unionDenom = lcm(a.denominator, b.getDenom());
    long long relNum = a.numerator * unionDenom / a.denominator;
    long long mulNum = b.numerator * unionDenom / b.denominator;
    long long numerator = relNum + mulNum;
    long long denominator = unionDenom;

    Fraction res(numerator, denominator);
    res.reduce();

    return res;
}

Fraction subtract(Fraction a, Fraction b) {
    long long numerator = a.numerator * b.denominator - b.numerator * a.denominator;
    long long denominator = a.denominator * b.denominator;

    Fraction res(numerator, denominator);
    res.reduce();

    return res;
}

Fraction divide(Fraction a, Fraction b) {
    a.numerator *= b.getDenom();
    a.denominator *= b.getNom();

    if (a.denominator < 0) {
        a.numerator *= -1;
        a.denominator *= -1;
    }

    Fraction res(a.numerator, a.denominator);
    res.reduce();

    return res;
}

Fraction multiply(Fraction a, Fraction b) {
    a.numerator *= b.getNom();
    a.denominator *= b.getDenom();

    Fraction res(a.numerator, a.denominator);
    res.reduce();

    return res;
}