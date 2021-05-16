import argparse

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description = '사용법 테스트')

# 입력받을 인자값 등록
parser.add_argument('--target', '-t', required = True, help = '어느것을 요구')
parser.add_argument('--env', required = False, default = 'dev', help = '실행환경')

# 입력받은 인자값을 args에 저장
args = parser.parse_args()

# 입력받은 인자값 출력
print(args.target)
print(args.env)
