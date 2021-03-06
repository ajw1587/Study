# N명의 병사가 무작위로 나열되어 있다.
# 각 병사는 특정한 값의 전투력을 보유하고 있다.
# 전투력이 높은 병사가 앞쪽에 오도록 내림차순으로 배치를 하고자 한다.
# 즉, 앞쪽에 있는 병사의 전투력이 뒤쪽에 있는 병사보다 높다.
# 배치 과정에서 특정한 위치에 있는 병사를 열외시키는 방법을 이용한다.
# 남아있는 병사의 수가 최대가 되도록 해라.
# 첫째줄: N (1<= N <= 2000), 둘째줄: 각 병사의 전투력

n = int(input())
array = list(map(int, input().split()))

dp = [1] * n
for i in range(1, n):
  for j in range(i):
    if array[j] > array[i]:
      dp[i] = max(dp[i], dp[j] + 1)

print(dp[-1])