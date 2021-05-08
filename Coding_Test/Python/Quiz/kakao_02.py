from collections import deque

def P_location(array):
    p_location = []
    for i in range(5):
        for j in range(5):
            if array[i][j] == 1:
                p_location.append((i, j))
    return p_location

def distance(x, y, array):
    queue = deque()
    queue.append((x, y))
    
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    while queue:
        x2, y2 = queue.popleft()
        for i in range(4):
            nx = x2 + dx[i]
            ny = y2 + dy[i]
            if nx < 0 or ny < 0 or nx >= 5 or ny >= 5:
                continue
            if array[nx][ny] == 0:
                queue.append((nx, ny))
                array[nx][ny] = 2
            elif array[nx][ny] == 1:
                if (abs(x - nx) + abs(y - ny)) <= 2:
                    array[x2][y2] = 2
                    return True
    return False

def change_array(array):
    num_array = []
    for i in range(5):
        sub_array = []
        for j in range(5):
            if array[i][j] == 'P':
                sub_array.append(1)
            elif array[i][j] == 'O':
                sub_array.append(0)
            else:
                sub_array.append(2)
        num_array.append(sub_array)
    return num_array

def solution(places):
    answer = [1] * 5
    for i in range(len(places)):
        array = change_array(places[i])
        p_location = P_location(array)
        if len(p_location) == 0:
            continue
        for j in range(len(p_location)):
            x = p_location[j][0]
            y = p_location[j][1]
            array[x][y] = 2
            
            if distance(x, y, array) == True:
                answer[i] = 0
                break
    return answer