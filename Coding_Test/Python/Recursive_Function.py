def recursive_function(i):
    # 100번째 호출시 종료
    if i == 100:
        return
    print(i, '번째 재귀함수에서', i + 1, '번째 재귀함수를 호출합니다.')
    recursive_function(i + 1)
    print(i, '번째 재귀함수를 종료합니다.')
recursive_function(1)



# 재귀함수를 사용하지 않은 팩토리얼
def factorial_iterative(n):
    result = 1
    # 1부터 n까지의 수를 차례대로 곱하기
    for i in range(1, n + 1):
        result *= i
    return result

# 재귀함수를 사용한 팩토리얼
def factorial_recursive(n):
    if n <= 1:
        return 1
    # n! = n * (n - 1) 그대로 적기
    return n * factorial_recursive(n - 1)


# 유클리드 호제법 (최대공약수)
def gcd(a, b):
    if a % b == 0:
        return b
    else:
        return gcd(b, a % b)
