#include "myArray.h"

void* setArray(int Size, int Count)
{
	return calloc(Count, Size);
}

int* SetArrRange(int* _block, int _Size, int _Min, int _Max)
{
	int range = _Max - _Min + 1;
	if (_block == NULL) _block = (int*)setArray(_Size, sizeof(int));

	for (int i = 0; i < _Size; i++)
	{
		_block[i] = rand() % range + _Min;
		
		for (int j = 0; j < i; j++)
		{
			if (_block[i] == _block[j])
			{
				i--;
				break;
			}
		}
	}
	return _block;
}

void printArray(const int* _block, int _Count)
{
	for (int i = 0; i < _Count; i++)
	{
		printf("%3d ", _block[i]);
	}
	printf("\n");
}