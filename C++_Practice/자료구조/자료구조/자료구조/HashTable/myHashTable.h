#pragma once
#include "HashTable.h"

//HashFunc type means function to return int data and has const char ptr and integer Parameter
typedef int HashFunc(const char* const _str, int keyRange);

typedef struct _HasHTable
{
	List** Bucket;
	int BucketSize;
	HashFunc* hashf;
}hashTable;

hashTable* initTable(HashFunc hashf, int BucketSize);
void appendTable(hashTable* _table, const char* const _text);
void viewTable(const hashTable* _table);
void SearchData(const hashTable* _table, const char* const _text);