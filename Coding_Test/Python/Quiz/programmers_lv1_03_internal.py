# https://programmers.co.kr/learn/courses/30/lessons/70128
def solution(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] * b[i])
    answer = sum(result)
    return answer