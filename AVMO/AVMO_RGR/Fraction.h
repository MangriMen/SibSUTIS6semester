#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

long long gcd(long long a, long long b);
long long lcm(long long a, long long b);

#define FRACTION_MAX "999999999"

class Fraction {
public:
    long long numerator;
    long long denominator;

    Fraction(long long num = 1, long long denom = 1) : numerator(num), denominator(denom) {}
    Fraction() : numerator(0), denominator(1) {}
    Fraction(const string& str) {
        int pos = str.find("/");

        if (pos == str.npos) {
            numerator = stoi(str);
            denominator = 1;
        } else {
            numerator = stoi(str.substr(0, pos));
            denominator = stoi(str.substr(pos + 1, str.length()));
        }
    }

    long long getNom() {
        return numerator;
    }

    long long getDenom() {
        return denominator;
    }

    Fraction ABS() {
        return Fraction(abs(numerator), abs(denominator));
    }

    string toString() {
        string frac = "";

        if (numerator == 0) {
            frac.append("0");
            return frac;
        }

        frac.append(to_string(numerator));
        if (denominator != 1) {
            frac.append("/");
            frac.append(to_string(denominator));
        }

        return frac;
    }

    void reduce() {
        long long GCD = gcd(abs(numerator), denominator);

        if (GCD != 1 && GCD != 0) {
            numerator /= GCD;
            denominator /= GCD;
        }
    }
};

Fraction summarize(Fraction a, Fraction b);
Fraction subtract(Fraction a, Fraction b);
Fraction divide(Fraction a, Fraction b);
Fraction multiply(Fraction a, Fraction b);