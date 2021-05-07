# n: 배열의 크기
# m: 몇번 더할 수 있는지
# k: 연속적으로 몇번 더할 수 있는지
# n, m, k 세 수를 입력 받는다.
# n, m, k = map(int, input().split())
n, m, k = 5, 7, 2

# N개의 수를 공백을 기준으로 입력받는다.
# data = list(map(int, input().split()))
data = list([3, 4, 3, 4, 3])

data.sort()
first = data[-1]
second = data[-2]

print('first: ', first)
print('second: ', second)

first_number = m // k
print(first_number)
first_number = first_number * k

second_number = m % k

result = first * first_number + second * second_number
print(result)