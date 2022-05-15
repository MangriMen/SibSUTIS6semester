#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include "Fraction.h"
using namespace std;

bool isEqual(Fraction a, Fraction b);
bool greaterThen(Fraction a, Fraction b);
bool lessThen(Fraction a, Fraction b);
void printMatrix(vector<vector<Fraction>> matrix);
void jordanGauss(vector<vector<Fraction>>& matrix, pair<int, int> basic);
void findAndMove(vector<vector<Fraction>>& matrix, int i, int j);
void createBasis(vector<vector<Fraction>>& matrix, vector<int> basic, vector<pair<int, int>>& currentBasicVariables);
int zFindNegative(vector<vector<Fraction>>& matrix);
int simplexRatio(vector<vector<Fraction>>& matrix, int j);
void deletePreviousBasic(vector<pair<int, int>>& currentBasicVariables, int i);
void printBasicVariables(vector<pair<int, int>>& currentBasicVariables);
bool isNotBasic(vector<pair<int, int>>& currentBasicVariables, int j);
int multipleSolutionCount(vector<vector<Fraction>>& matrix, vector<pair<int, int>>& currentBasicVariables);
vector<Fraction> findSolution(vector<vector<Fraction>>& matrix, vector<pair<int, int>>& currentBasicVariables);
void solve(vector<vector<Fraction>>& matrix, vector<vector<Fraction>>& startingMatrix, vector<pair<int, int>>& currentBasicVariables);

int main() {
    setlocale(LC_ALL, "Russian");

    ifstream dataFileIn("data.txt");
    int rows = 4;
    int cols = 6;

    vector<vector<Fraction>> matrix(rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            string temp;
            dataFileIn >> temp;
            Fraction xVariable(temp);
            
            matrix[i].push_back(xVariable);
        }
    }
    dataFileIn.close();

    cout << left;
    cout << endl << "Input data:" << endl << endl;
    printMatrix(matrix);

    for (int i = 0; i < cols - 1; ++i) {
        matrix[rows - 1][i] = multiply(matrix[rows - 1][i], Fraction(-1, 1));
    }

    cout << endl << "Z to canonical form:" << endl << endl;
    printMatrix(matrix);
    vector<vector<Fraction>> startingMatrix(matrix);

    vector<int> basic;
    cout << endl << "Enter starting basic: ";
    for (int i = 0; i < matrix.size() - 1; ++i) {
        int basicXVariable;
        cin >> basicXVariable;

        basic.push_back(basicXVariable);
    }

    vector<pair<int, int>> currentBasicVariables;
    cout << endl << "-------------------------------------------" << endl;
    createBasis(matrix, basic, currentBasicVariables);
    printBasicVariables(currentBasicVariables);

    solve(matrix, startingMatrix, currentBasicVariables);

    return 0;
}

bool isEqual(Fraction a, Fraction b) {
    return (a.numerator == b.numerator && a.denominator == b.denominator);
}

bool greaterThen(Fraction a, Fraction b) {
    return (a.numerator * b.denominator > b.numerator * a.denominator);
}

bool lessThen(Fraction a, Fraction b) {
    return (a.numerator * b.denominator < b.numerator * a.denominator);
}

void printMatrix(vector<vector<Fraction>> matrix) {
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {
            cout << setw(6) << matrix[i][j].toString() << " ";
        }
        cout << endl;
    }
}

void jordanGauss(vector<vector<Fraction>>& matrix, pair<int, int> basic) {
    int i = basic.first;
    int j = basic.second;

    Fraction el = matrix[i][j];

    for (int k = 0; k < matrix[i].size(); k++) {
        matrix[i][k] = divide(matrix[i][k], el);
    }

    for (int k = 0; k < matrix.size(); k++) {
        if (k == i) {
            continue;
        }

        Fraction zeroMultiplier = matrix[k][j];

        for (int l = 0; l < matrix[k].size(); l++) {
            matrix[k][l] = subtract(matrix[k][l], multiply(matrix[i][l], zeroMultiplier));
        }
    }
}

void findAndMove(vector<vector<Fraction>>& matrix, int i, int j) {
    Fraction currentElement = matrix[i][j];
    int index = i;
    int prevIndex = i;

    while (i < matrix.size()) {
        if (greaterThen(matrix[i][j], currentElement)) {
            currentElement = matrix[i][j];
            index = i;
        }

        i += 1;
    }

    if (prevIndex != index) {
        swap(matrix[prevIndex], matrix[index]);
    }
}

void createBasis(vector<vector<Fraction>>& matrix, vector<int> basic, vector<pair<int, int>>& currentBasicVariables) {
    for (int i = 0; i < basic.size(); ++i) {
        basic[i] -= 1;
    }
    
    for (int i = 0; i < basic.size(); ++i) {
        if (isEqual(matrix[i][basic[i]], Fraction("0"))) {
            findAndMove(matrix, i, basic[i]);

            if (isEqual(matrix[i][basic[i]], Fraction(0, 1))) {
                cout << "Basic error" << endl;
                system("pause");
                exit(0);
            }
        }
        
        currentBasicVariables.push_back({ i, basic[i] });
        jordanGauss(matrix, currentBasicVariables[i]);
        printMatrix(matrix);
        
        if (i < basic.size() - 1) {
            cout << endl;
        }
    }

    for (int i = 0; i < matrix.size() - 1; ++i) {
        if (lessThen(matrix[i][matrix[i].size() - 1], Fraction("0"))) {
            cout << "Reference basis failure" << endl;
            exit(0);
        }
    }
}

int zFindNegative(vector<vector<Fraction>>& matrix) {
    int negativeIndex = -1;
    int ZN = matrix.size() - 1;
    Fraction el("0");

    for (int k = 0; k < matrix[0].size() - 1; k++) {
        if (lessThen(matrix[ZN][k], el)) {
            el = matrix[ZN][k];
            negativeIndex = k;
        }
    }

    return negativeIndex;
}

int simplexRatio(vector<vector<Fraction>>& matrix, int j) {
    int result = -1;
    Fraction min(FRACTION_MAX);
    
    for (int i = 0; i < matrix.size() - 1; ++i) {
        if (greaterThen(matrix[i][j], Fraction("0"))) {
            const Fraction temp = divide(matrix[i][matrix[0].size() - 1], matrix[i][j]);

            if (lessThen(temp, min)) {
                min = temp;
                result = i;
            }
        }
    }

    return result;
}

void deletePreviousBasic(vector<pair<int, int>>& currentBasicVariables, int i) {
    for (int k = 0; k < currentBasicVariables.size(); k++) {
        if (currentBasicVariables[k].first == i) {
            currentBasicVariables.erase(currentBasicVariables.begin() + k);
            break;
        }
    }
}

void printBasicVariables(vector<pair<int, int>>& currentBasicVariables) {
    cout << endl << "Current basic: ";
    for (int i = 0; i < currentBasicVariables.size(); ++i) {
        cout << "x" << currentBasicVariables[i].second + 1 << " ";
    }

    cout << endl << "-------------------------------------------" << endl;
}

bool isNotBasic(vector<pair<int, int>>& currentBasicVariables, int j) {
    for (int i = 0; i < currentBasicVariables.size(); ++i) {
        if (currentBasicVariables[i].second == j) {
            return false;
        }
    }

    return true;
}

int multipleSolutionCount(vector<vector<Fraction>>& matrix, vector<pair<int, int>>& currentBasicVariables) {
    int result = -1;
    int ZN = matrix.size() - 1;

    for (int i = 0; i < matrix[0].size() - 1; ++i) {
        if (isEqual(matrix[ZN][i], Fraction("0")) && isNotBasic(currentBasicVariables, i)) {
            result = i;
        }
    }

    return result;
}

vector<Fraction> findSolution(vector<vector<Fraction>>& matrix, vector<pair<int, int>>& currentBasicVariables) {
    vector<Fraction> solution;
    
    for (int i = 0; i < matrix[0].size() - 1; ++i) {
        solution.push_back(Fraction("0"));
    }

    for (int i = 0; i < currentBasicVariables.size(); ++i) {
        solution[currentBasicVariables[i].second] = matrix[currentBasicVariables[i].first][matrix[0].size() - 1];
    }

    return solution;
}

void solveStep(vector<pair<int, int>>& currentBasicVariables, vector<vector<Fraction>>& matrix, int i, int j, vector<vector<Fraction>> &solutions) {
    deletePreviousBasic(currentBasicVariables, i);
    currentBasicVariables.push_back({ i, j });

    jordanGauss(matrix, { i, j });

    solutions.push_back(findSolution(matrix, currentBasicVariables));

    printMatrix(matrix);
    printBasicVariables(currentBasicVariables);
}

void solve(vector<vector<Fraction>>& matrix, vector<vector<Fraction>>& startingMatrix, vector<pair<int, int>>& currentBasicVariables) {
    vector<vector<Fraction>> solutions;
    solutions.push_back(findSolution(matrix, currentBasicVariables));
    
    while (zFindNegative(matrix) != -1) {
        int j = zFindNegative(matrix);
        int i = simplexRatio(matrix, j);

        if (i == -1) {
            cout << "Function not limited" << endl;
            system("pause");
            exit(0);
        }
       
        solveStep(currentBasicVariables, matrix, i, j, solutions);
    }

    if (multipleSolutionCount(matrix, currentBasicVariables) != -1) {
        int j = multipleSolutionCount(matrix, currentBasicVariables);
        int i = simplexRatio(matrix, j);
        
        solveStep(currentBasicVariables, matrix, i, j, solutions);
    }

    cout << endl << "Solution: ";
    if (multipleSolutionCount(matrix, currentBasicVariables) != -1) {
        vector<Fraction> solutionFinal;
        for (int i = 0; i < solutions[0].size(); ++i) {
            solutionFinal.push_back(summarize(
                multiply(solutions[solutions.size() - 2][i], Fraction("-1")),
                solutions[solutions.size() - 1][i])
            );
        }

        for (int i = 0; i < solutions[0].size(); ++i) {
            if (isEqual(solutionFinal[i], Fraction("0")) && !isEqual(solutions[solutions.size() - 2][i], Fraction("0"))) {
                cout << "x" << i + 1 << " = ";
                cout << solutions[solutions.size() - 2][i].toString();
            } else if (!isEqual(solutionFinal[i], Fraction("0")) && isEqual(solutions[solutions.size() - 2][i], Fraction("0"))) {
                cout << "x" << i + 1 << " = ";
                cout << solutionFinal[i].toString() << "h";
            } else if (isEqual(solutionFinal[i], Fraction("0")) && isEqual(solutions[solutions.size() - 2][i], Fraction("0"))) {
                cout << "x" << i + 1 << " = 0";
            } else {
                cout << "x" << i + 1 << " = ";
                cout << solutions[solutions.size() - 2][i].toString();
                if (lessThen(solutionFinal[i], Fraction("0"))) {
                    cout << " - " << multiply(solutionFinal[i], Fraction("-1")).toString() << "h";
                } else {
                    cout << " + " << solutionFinal[i].toString() << "h";
                }
            }
            cout << endl;
        }
    } else {
        for (int i = 0; i < solutions[0].size(); ++i) {
            cout << "x" << i + 1 << " = " << solutions[solutions.size() - 1][i].toString() << " ";
        }
    }

    Fraction answer("0");
    cout << endl << endl << "Answer: ";
    for (int i = 0; i < startingMatrix[startingMatrix.size() - 1].size(); ++i) {
        if (isEqual(startingMatrix[startingMatrix.size() - 1][i], Fraction(0, 1))) {
            continue;
        }

        if (i != 0) { cout << " + "; }

        cout << startingMatrix[startingMatrix.size() - 1][i].toString() << " * "
            << solutions[solutions.size() - 1][i].toString();

        answer = summarize(answer, multiply(startingMatrix[startingMatrix.size() - 1][i], solutions[solutions.size() - 1][i]));
    }
    cout << " = " << answer.toString() << endl;

}