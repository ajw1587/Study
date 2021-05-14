# 구간 합 (Interval Sum)
# N개의 정수로 구성된 수열이 있다.
# M개의 쿼리(Query) 정보가 주어진다.
#   - 각 쿼리는 Left와 Right으로 구성된다.
#   - 각 쿼리에 대하여 [Left, Right] 구간에 포함된 데이터들의 합을 출력해야 한다.
# 수행 시간 제한은 O(N + M)이다.
# 접두사 합(Prefix Sum): 배열의 맨 앞부터 특정 위치까지의 합을 미리 구해 놓은것
# 주어진 배열 = 10 / 20 / 30 / 40 / 50
# 접두사 배열(P) = 0 / 10 / 30 / 60 / 100 / 150
# 구간합: P[Right] - P[Left - 1]

n = 5
data = [10, 20, 30, 40, 50]

sum_value = 0
prefix_sum = [0]
for i in data:
    sum_value += i
    prefix_sum.append(sum_value)

left = 3
right = 4
print(prefix_sum[right] - prefix_sum[left - 1])
