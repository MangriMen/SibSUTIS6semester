import sys


def getMin(a, r, ex1, ex2):
    min = stl = str = sys.maxsize
    for i in range(0, ex1):
        for j in range(0, ex2):
            if a[i][j] < min or min == sys.maxsize:
                if i in r['str'] or j in r['stl']:
                    continue
                min = a[i][j]
                str = i
                stl = j
    return str, stl


def findMin(a, i, j, ex1, ex2, t):
    min = stl = str = sys.maxsize
    for k in range(0, ex2):
        if min > a[i][k] and not(k in t['stl']):
            min = a[i][k]
            stl = k
            str = i

    for k in range(0, ex1):
        if min > a[k][j] and not(k in t['str']):
            min = a[k][j]
            stl = j
            str = k

    return str, stl


def func(a, provider_info, consumer_info, extra1, extra2):
    t = {'str': [], 'stl': []}
    r = []
    for i in range(0, len(a)):
        r.append([])
        for j in range(0, len(a[i])):
            r[i].append(-1)

    for p in range(0, extra1*extra2):
        i, j = getMin(a, t, extra1, extra2)
        try:
            info = (provider_info[i], consumer_info[j])
        except:
            return r
        if info[0] > info[1]:
            r[i][j] = info[1]
            provider_info[i] -= info[1]
            t['stl'].append(j)
        elif info[0] < info[1]:
            r[i][j] = info[0]
            consumer_info[j] -= info[0]
            t['str'].append(i)
        else:
            r[i][j] = info[1]
            consumer_info[j] -= info[0]
            provider_info[i] -= info[1]
            t['stl'].append(j)
            t['str'].append(i)
            i1, j1 = findMin(a, i, j, extra1, extra2, t)
            #print(f"{i1} {j1}")
            try:
                r[i1][j1] = 0
            except:
                return r
    return r


def countCost(a, b):
    tmp_result = 0
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            if a[i][j] != -1:
                tmp_result += a[i][j]*b[i][j]
    return tmp_result


def main():
    file = open('test.txt', 'r')

    line = file.readline()
    provider_info = [int(item) for item in line.split(" ")]

    line = file.readline()
    consumer_info = [int(item) for item in line.split(" ")]
    extraconsumerinfo = [int(item) for item in consumer_info]

    lines = file.readlines()
    b = []
    for line in lines:
        b.append([int(item) for item in line.split(" ")])

    providersum = consumersum = 0
    for item in provider_info:
        providersum += item

    for item in consumer_info:
        consumersum += item

    extraprovider = False
    if consumersum > providersum:
        extraprovider = True
        print("Need another provider\n")
        provider_info.append(abs(consumersum - providersum))
        tmpm = []
        for i in range(0, len(b[0])):
            tmpm.append(0)
        b.append(tmpm)

    extraconsumer = False
    if consumersum < providersum:
        extraconsumer = True
        print("Need another consumer\n")
        provider_info.append(abs(consumersum - providersum))

    print("Provider info")
    for item in provider_info:
        print(item, end=' ')
    print("\n")

    print("Consumer info")
    for item in consumer_info:
        print(item, end=' ')
    print("\n")

    print("Transport cost")
    for row in b:
        for item in row:
            print('{0:5}'.format(item), end=' ')
        print()

    if extraprovider:
        strcnt = len(b) - 1
    else:
        strcnt = len(b)

    if extraconsumer:
        clmcnt = len(b[0]) - 1
    else:
        clmcnt = len(b[0])

    r = func(b, provider_info, consumer_info, strcnt, clmcnt)
    el = 0
    for i in range(0, len(r[0])):
        tmpsum = 0
        for j in range(0, len(r)):
            if r[j][i] != -1:
                tmpsum += r[j][i]
        if extraconsumerinfo[i] != tmpsum:
            r[len(r)-1][i] = extraconsumerinfo[i] - tmpsum

    for i in range(0, len(r[0])):
        for j in range(0, len(r)):
            if r[j][i] != -1:
                el += 1

    print("\nTraffic distribution")
    for row in r:
        for item in row:
            print('{0:5}'.format(item), end=' ')
        print()
    print(f"{len(r)} + {len(r[0])} - 1 == {el}")
    print(f"Cost is {countCost(r,b)}")


if __name__ == "__main__":
    main()
