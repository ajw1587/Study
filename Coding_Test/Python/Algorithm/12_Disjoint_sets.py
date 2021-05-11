# 서로소 집합 == 합치기 찾기(Union find)자료구조:
# 공통 원소가 없는 두 집합
# ex) {1, 2}, {3, 4}는 서로소 관계이다.
# ex) {1, 2}, {2, 3}은 서로소 관계가 아니다.

# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v, e = map(int, input().split())
parent = [0] * (v + 1)  # 부모 테이블 초기화하기

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i

# Union 연산을 각각 수행
for i in range(e):
    a, b = map(int, input().split())
    union_parent(parent, a, b)

# 각 원소가 속한 집합 출력하기
print('각 원소가 속한 집합: ', end = '')
for i in range(1, v + 1):
    print(find_parent(parent, i), end = ' ')
print()

# 부모 테이블 내용 출력하기
print('부모 테이블: ', end = '')
for i in range(1, v + 1):
    print(parent[i], end = ' ')
