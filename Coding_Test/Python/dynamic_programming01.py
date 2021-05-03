# https://www.youtube.com/watch?v=rWbjQphRE9A&list=PLVsNizTWUw7H9_of5YCB0FmsSc-K44y81&index=28
# 동적 계획법
# 아래 두 조건을 만족시 사용 가능
# 1. 최적 부분 구조 (Optimal Substructure)
#   - 큰 문제를 작은 문제로 나눌 수 있으며 작은 문제의 답을 모아서 큰 문제를 해결할 수 있다.
# 2. 중복되는 부분 문제 (overlapping Subproblem)
#   - 동일한 작은 문제를 반복적으로 해결해야 한다.

# Fibonacci Function
def fibo01(x):
    if x == 1 or x == 2:
        return 1
    return fibo(x - 1) + fibo(x - 2)
# 여기서의 시간 복잡도 = O(2^n)

# 다이나믹 프로그래밍 사용하기
# 메모이제이션 (Memoization)
d = [0] * 100
# 피보나치 함수를 재귀함수로 구현(탑다운 다이나믹 프로그래밍)
def fibo02(x):
    # 종료 조건
    if x == 1 or x == 2:
        return 1
    # 이미 계산한 적 있는 문제라면 그대로 반환
    if d[x] != 0:
        return d[x]
    # 아직 계산하지 않은 문제라면 점화식에 따라서 피보나치 결과 반환
    d[x] = fibo(x - 1) + fibo(x - 2)
    return d[x]

# 보텀업 방식
d = [0] * 100
d[1] = 1
d[2] = 1
n = 99

for i in range(3, n + 1):
    d[i] = d[i - 1] + d[i - 2]
