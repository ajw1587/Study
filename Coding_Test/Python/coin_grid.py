n = 1260
count = 0
list = [500, 100, 50, 10]
for coin in list:
    print('coin: ', coin)
    count += n // coin
    n %= coin
print(count)