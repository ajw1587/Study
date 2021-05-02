# 계수정렬
# https://www.youtube.com/watch?v=65Ui3RNibRA&list=PLVsNizTWUw7H9_of5YCB0FmsSc-K44y81&index=24
# 공간 복잡도가 높다.

# 모든 원소의 값이 0보다 크다고 가정
array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
# 모든 범위를 포함하는 리스트 생성
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end = ' ')