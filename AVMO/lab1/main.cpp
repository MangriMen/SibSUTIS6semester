
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

long long my_gcd(long long a, long long b)
{
	return ((b == 0) ? a : my_gcd(b, a % b));
}

// eazi fraktur
class ef
{
	long long up;
	long long down;

public:
	ef(long long u, long long d) : up(u), down(d)
	{
		normalize();
	}
	ef() : ef(0, 1) {}

	string to_string()
	{
		if (down == 1)
			return std::to_string(up);
		return std::to_string(up) + "/" + std::to_string(down);
	}

	string sign()
	{
		if (up < 0)
			return "-";
		return "+";
	}

	std::pair<long long, long long> get() const
	{
		return {up, down};
	}

	ef operator+(const ef &other) const
	{
		ef temp{};
		temp.up = up * other.down + down * other.up;
		temp.down = down * other.down;
		temp.normalize();
		return temp;
	}

	ef operator-(const ef &other) const
	{
		ef temp{};
		temp.up = up * other.down - down * other.up;
		temp.down = down * other.down;
		temp.normalize();
		return temp;
	}

	ef operator*(const ef &other)
	{
		ef temp{};
		temp.up = up * other.up;
		temp.down = down * other.down;
		temp.normalize();
		return temp;
	}

	ef operator/(const ef &other)
	{
		ef temp{};
		temp.up = up * other.down;
		temp.down = down * other.up;
		temp.normalize();
		return temp;
	}

	bool operator==(long long num) const
	{
		ef temp;
		temp.up = up;
		temp.down = down;
		temp.normalize();
		if (temp.down == 1 && temp.up == num)
			return true;
		return false;
	}

	bool operator!=(long long num)
	{
		return !this->operator==(num);
	}

	void normalize()
	{
		if (up == 0)
		{
			down = 1;
			return;
		}

		auto gcd = my_gcd(up, down);
		up /= gcd, down /= gcd;
		if (down < 0)
			down *= -1, up *= -1;
	}
};

// linear system
class solver
{
	vector<vector<ef>> matrix;

public:
	// n - rows
	// m - columns
	void init()
	{
		cout << "enter size (N [space] M [enter])" << endl;
		long long n, m;
		cin >> n >> m;
		cin.ignore(_MAX_PATH, '\n');

		cout << "enter matrix" << endl;
		string line{};
		matrix.clear();
		matrix.resize(n);
		for (auto &i : matrix)
			i.resize(m);
		for (int i = 0; i < n; i++)
		{
			getline(cin, line, '\n');
			line += " ";
			vector<string> parsed{};
			string temp{};
			for (int j = 0; j < line.length(); j++)
				if (line[j] == ' ' || line[j] == ',' || line[j] == ';')
				{
					if (temp.length() > 0)
						parsed.push_back(temp);
					temp = "";
				}
				else
					temp += line[j];

			if (parsed.size() != m)
			{
				cout << "parsed.size() (" << parsed.size() << ") != m (" << m << ")" << endl;
				return;
			}
			for (int j = 0; j < parsed.size(); j++)
			{
				parsed[j] += " ";
				long long tmp = 0ll;
				vector<long long> u_and_d{};
				bool minus_flag = false;
				for (int k = 0; k < parsed[j].length(); k++)
				{
					if (parsed[j][k] == '/' || parsed[j][k] == ':' || parsed[j][k] == ' ')
					{
						if (minus_flag)
							u_and_d.push_back(-tmp);
						else
							u_and_d.push_back(tmp);
						tmp = 0ll;
					}
					else if (parsed[j][k] == '-')
						minus_flag = true;
					else
						tmp *= 10, tmp += static_cast<long long>(parsed[j][k] - '0');
				}
				if (u_and_d.size() < 2 || u_and_d[1] == 0)
					matrix[i][j] = {u_and_d[0], 1};
				else
					matrix[i][j] = {u_and_d[0], u_and_d[1]};
			}
		}
	}

	void solve()
	{
		// example:
		// a b c | d
		// e f g | h
		// i j k | l

		for (int row = 0; row < matrix.size(); row++)
		{
			bool nuls = false;
			if (matrix[row][row] == 0)
			{
				nuls = true;
				for (int i = row + 1; i < matrix.size(); i++)
					if (matrix[i][i] != 0)
					{
						swap(matrix[i][i], matrix[row][row]);
						nuls = false;
					}
			}

			if (nuls)
				continue;

			// normalize row
			for (int col = 0; col < matrix[row].size(); col++)
			{
				if (col == row)
					continue;
				matrix[row][col] = matrix[row][col] / matrix[row][row];
			}
			matrix[row][row] = {1, 1};

			// other rows
			for (int i = 0; i < matrix.size(); i++)
			{
				if (i == row)
					continue;
				ef factor = matrix[i][row] / matrix[row][row];
				for (int col = row; col < matrix[i].size(); col++)
				{
					matrix[i][col] = matrix[i][col] - matrix[row][col] * factor;
					matrix[i][col].normalize();
				}
			}

			cout << endl;
			output_matrix();
		}

		// case 0x(m-1) 1
		bool no_answers = false;
		for (int i = 0; i < matrix.size(); i++)
		{
			auto mx = max_element(matrix[i].begin(), matrix[i].end() - 1, [](const ef &a, const ef &b)
								  { return a.get().first < b.get().first; });
			auto mn = min_element(matrix[i].begin(), matrix[i].end() - 1, [](const ef &a, const ef &b)
								  { return a.get().first < b.get().first; });
			if (mx->get().first == 0 && mn->get().first == 0)
			{
				if (matrix[i].back() != 0)
					no_answers = true;
				else
				{
					matrix.erase(matrix.begin() + i--);
					cout << endl;
					output_matrix();
				}
			}
		}

		if (no_answers)
		{
			cout << endl
				 << "no answers" << endl;
			return;
		}

		// other cases
		cout << endl
			 << "answer:" << endl;
		for (int i = 0; i < matrix.size(); i++)
		{
			for (int j = 0, x_count = 0; j < matrix[i].size() - 1; j++)
			{
				if (matrix[i][j].get().first)
				{
					if (x_count)
						cout << " " << matrix[i][j].sign() << " ";

					if (abs(matrix[i][j].get().first) != 1ll)
						cout << matrix[i][j].to_string() << " x" << j + 1;
					else
						cout << "x" << j + 1;

					x_count++;
				}
			}
			cout << " = " << matrix[i].back().to_string() << endl;
		}
	}

	void output_matrix()
	{
		for (int i = 0; i < matrix.size(); i++)
		{
			for (int j = 0; j < matrix[0].size(); j++)
				cout << matrix[i][j].to_string() << " ";
			cout << endl;
		}
	}
};

int main()
{
	solver s;
	s.init();
	s.solve();
}