# 마라톤에 참여한 선수들의 이름이 담긴 배열 participant와
# 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때, 
# 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.

def solution(participant, completion):
    participant.sort()
    completion.sort()
    idx = 0
    for i in range(len(participant)):
        if participant[i] != completion[i]:
            idx = i
            break    

    answer = participant[i]
    return answer

parti = ["leo", "kiki", "eden"]
com = ["eden", "kiki"]

print(solution(parti, com))