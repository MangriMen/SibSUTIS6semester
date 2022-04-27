import sys
import copy


def getMin(matrix, r, line_count, column_count):
    min_element = stl_index = str_index = sys.maxsize

    for i in range(0, line_count):
        for j in range(0, column_count):
            if matrix[i][j] < min_element or min_element == sys.maxsize:
                if i in r['str_index'] or j in r['stl_index']:
                    continue
                min_element = matrix[i][j]
                str_index = i
                stl_index = j

    return str_index, stl_index


def findMin(matrix, i, j, line_count, column_count, t):
    min_element = stl_index = str_index = sys.maxsize

    for k in range(0, column_count):
        if min_element > matrix[i][k] and not(k in t['stl_index']):
            min_element = matrix[i][k]
            stl_index = k
            str_index = i

    for k in range(0, line_count):
        if min_element > matrix[k][j] and not(k in t['str_index']):
            min_element = matrix[k][j]
            stl_index = j
            str_index = k

    return str_index, stl_index


def solver(transport_matrix, reserves, needs, line_count, column_count):
    t = {'str_index': [], 'stl_index': []}
    r = []

    for i in range(0, len(transport_matrix)):
        r.append([])
        for j in range(0, len(transport_matrix[i])):
            r[i].append(-1)

    for p in range(0, line_count*column_count):
        i, j = getMin(transport_matrix, t, line_count, column_count)

        try:
            info = (reserves[i], needs[j])
        except:
            return r

        if info[0] > info[1]:
            r[i][j] = info[1]
            reserves[i] -= info[1]
            t['stl_index'].append(j)
        elif info[0] < info[1]:
            r[i][j] = info[0]
            needs[j] -= info[0]
            t['str_index'].append(i)
        else:
            r[i][j] = info[1]
            needs[j] -= info[0]
            reserves[i] -= info[1]
            t['stl_index'].append(j)
            t['str_index'].append(i)
            i1, j1 = findMin(transport_matrix, i, j,
                             line_count, column_count, t)
            try:
                r[i1][j1] = 0
            except:
                return r

    return r


def countCost(matrix, transport_matrix):
    tmp_result = 0

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if matrix[i][j] != -1:
                tmp_result += matrix[i][j]*transport_matrix[i][j]

    return tmp_result


def main():
    file = open('test.txt', 'r')

    line = file.readline()
    reserves = [int(item) for item in line.split(" ")]

    line = file.readline()
    init_needs = [int(item) for item in line.split(" ")]
    needs = copy.deepcopy(init_needs)

    lines = file.readlines()

    transport_matrix = []

    for line in lines:
        transport_matrix.append([int(item) for item in line.split(" ")])

    reserves_sum = needs_sum = 0

    for item in reserves:
        reserves_sum += item

    for item in init_needs:
        needs_sum += item

    extra_provider = False

    if needs_sum > reserves_sum:
        extra_provider = True
        print("Additional provider is needed\n")
        reserves.append(abs(needs_sum - reserves_sum))
        tmpm = []
        for i in range(0, len(transport_matrix[0])):
            tmpm.append(0)
        transport_matrix.append(tmpm)

    extra_consumer = False

    if needs_sum < reserves_sum:
        extra_consumer = True
        print("Additional consumer is needed\n")
        reserves.append(abs(needs_sum - reserves_sum))

    print("Provider info")
    for item in reserves:
        print(item, end=' ')
    print("\n")

    print("Consumer info")
    for item in init_needs:
        print(item, end=' ')
    print("\n")

    print("Transport cost")
    for row in transport_matrix:
        for item in row:
            print('{0:5}'.format(item), end=' ')
        print(end="\n")

    line_count = len(transport_matrix) - int(extra_provider)
    column_count = len(transport_matrix[0]) - int(extra_consumer)

    result_matrix = solver(transport_matrix, reserves,
                           init_needs, line_count, column_count)

    element_count = 0
    for i in range(0, len(result_matrix[0])):
        temp_sum = 0

        for j in range(0, len(result_matrix)):
            if result_matrix[j][i] != -1:
                temp_sum += result_matrix[j][i]

        if needs[i] != temp_sum:
            result_matrix[len(result_matrix) - 1][i] = needs[i] - temp_sum

    for i in range(0, len(result_matrix[0])):
        for j in range(0, len(result_matrix)):
            if result_matrix[j][i] != -1:
                element_count += 1

    print("\nTraffic distribution")
    for row in result_matrix:
        for item in row:
            print('{0:5}'.format(item), end=' ')
        print(end='\n')
    print(
        f"{len(result_matrix)} + {len(result_matrix[0])} - 1 == {element_count}")
    print(f"Cost is {countCost(result_matrix, transport_matrix)}")


if __name__ == "__main__":
    main()
