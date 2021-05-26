# 거스름돈 0원 만들기 경우의 수

a = 3000
b = 5000
budget = 23000

print(budget // a)
print(budget // b)
count = 0
for i in range(budget // a + 1):
    for j in range(budget // b + 1):
        money = i*a + j*b
        if money == budget:
            count += 1
print(count)

# array = [[] for i in range()]
