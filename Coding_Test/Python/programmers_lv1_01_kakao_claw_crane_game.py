# https://programmers.co.kr/learn/challenges
# board = 5x5 이상 30x30이하의 배열
# moves = 인형 이동
# 인형의 종류 1 ~ 1000, 0은 빈 공간

def solution(board, moves):
    count = 0
    stack = []
    for i in moves:
        i -= 1
        for j in range(len(board)):
            if board[j][i] != 0:
                stack.append(board[j][i])
                board[j][i] = 0

                if len(stack) >= 2:
                    if stack[-1] == stack[-2]:
                        stack.pop()
                        stack.pop()
                        count += 2
                break
    answer = count
    return answer

board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]
print(solution(board, moves))