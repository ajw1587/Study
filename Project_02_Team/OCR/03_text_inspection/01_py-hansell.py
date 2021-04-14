from hanspell import spell_checker

result = spell_checker.check(u'예찬이눈썹밀자')
print(result.as_dict())