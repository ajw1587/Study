from hanspell import spell_checker

result = spell_checker.check(u'영어는안되는구나!')
print(result.as_dict())