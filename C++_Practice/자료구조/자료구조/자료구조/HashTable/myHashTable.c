#include "myHashTable.h"

hashTable* initTable(HashFunc hashf, int BucketSize)
{
	hashTable* temp = (hashTable*)malloc(sizeof(hashTable));
	
	temp->BucketSize = BucketSize;
	temp->Bucket = (List**)calloc(BucketSize, sizeof(List*));
	temp->hashf = &hashf;

	return temp;
}

void appendTable(hashTable* _table, const char* const _text)
{
	int hashResult = _table->hashf(_text, _table->BucketSize);

	if (_table->Bucket[hashResult] == NULL)
		_table->Bucket[hashResult] = initLIst();

	push_back(_table->Bucket[hashResult], _text);
}

void viewTable(const hashTable* _table)
{
	List* cursor = NULL;

	for (int i = 0; i < _table->BucketSize; i++)
	{
		printf("bucket[%d]-> ", i);
		if (_table->Bucket[i] == NULL)
		{
			printf("(NULL)\n");
		}
		else
		{
			viewListElements(_table->Bucket[i]);
			printf("\n");
		}
	}
}

void SearchData(const hashTable* _table, const char* const _text)
{
	int index = _table->hashf(_text, _table->BucketSize);
	node* cursor = _table->Bucket[index]->head->pNext;
	int locate = 1;

	while (strcmp(cursor->Name, _text) != 0 && cursor != _table->Bucket[index]->tail)
	{
		cursor = cursor->pNext;
		locate++;
	}

	if (cursor != _table->Bucket[index]->tail)
	{
		printf("%s Locate => Nucket[%d]'s %dth Data", _text, index, locate);
	}
	else
	{
		printf("[%s] is Not Found\n", _text);
	}
}