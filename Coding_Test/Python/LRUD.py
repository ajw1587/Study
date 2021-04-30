n = list(map(str, input().split()))
print(type(n))
print(n)

x, y = 1, 1
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]
move_types = ['L', 'R', 'U', 'D']

for i in range(len(n)):
    for j in range(len(move_types)):
        if n[i] == move_types[j]:
            nx = x + dx[j]
            ny = y + dy[j]
            print(type(nx))
    if nx < 1 or ny < 1 or nx > n or ny > n:
        continue
    x = nx
    y = ny
print(x, y)