from hanspell import spell_checker

result = spell_checker.check(u'아버지가방에들어가신다.')
print(result.as_dict())