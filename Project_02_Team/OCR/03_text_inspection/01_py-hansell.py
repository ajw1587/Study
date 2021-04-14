from hanspell import spell_checker

result = spell_checker.check(u'맛춤법 검사')
print(result.as_dict()['checked'])

result = spell_checker.check(u'예찬이의여자칱구만들기프로젝트')
print(result.as_dict()['checked'])

result = spell_checker.check(u'외않되?')
print(result.as_dict()['checked'])