#include "mySort.h"

/* 기수 정렬 */
void RadixSort(int* block, int Size, int Radix) {
	a_queue** Bucket = (a_queue**)calloc(Radix, sizeof(a_queue*));
	for (int i = 0; i < Radix; i++) {
		Bucket[i] = initArrayQueue(Size);
	}
	int num = 0;
	int bi = 0;
	int d = 0;
	bool bEnd = false;
	while (1) {
		for (int i = 0; i < Size; i++) {
			num = block[i] / pow(10, d) % 10;
			enQueue(Bucket[num], block[i]);
		}
		if (isQFull(Bucket[0])) {
			bEnd = true;
			break;
		}
		else {
			for (int i = 0, bi = 0; i < Radix; i++) {
				while (!isQEmpty(Bucket[i])) {
					block[bi++] = dequeue(Bucket[i]);
				}
			}
			d++;
		}
	}
}

/* 버블 정렬 */
void BubbleSort(int* block, int Size)
{
	for (int i = 0; i < Size - 1; i++)
	{
		for (int j = 0; j < Size - 1 - i; j++)
		{
			if (block[j] > block[j + 1])
			{
				SWAP(block[j], block[j + 1]);
			}
		}
	}
}

/* 선택 정렬 */
void SelectSort(int* block, int Size)
{
	int Small = 0;
	for (int i = 0; i < Size - 1; i++)
	{
		Small = i;
		for (int j = i+1; j < Size; j++)
		{
			if (block[Small] > block[j])
				Small = j;
		}
		SWAP(block[i], Small);
	}
}

/* 삽입 정렬 */
void insertSort(int* block, int Size)
{
	int value;
	
	for (int i = 1; i < Size; i++)
	{
		value = block[i];
		for (int k = i - 1; k >= 0; k--)
		{
			if (value < block[k])
			{
				block[k + 1] = block[k];
			}
			else // value >= block[k]
			{
				block[k + 1] = value;
				break;
			}
		}
	}
}

/* 퀵 정렬 */
void QuickSort(int* block, int front, int rear)
{
	int pivot = 0;
	pivot = getPivot(block, front,rear);
	if (front < rear)
	{
		QuickSort(block, front, pivot - 1);
		QuickSort(block, pivot + 1, rear);
	}
	else return;
	return;
}

int getPivot(int* block, int front, int rear)
{
	int p =front;
	int low = front;
	int high = rear + 1;

	do
	{
		do
		{
			low++;
		} while (block[low] < block[p] && low <= rear);
		do
		{
			high++;
		} while (block[high] > block[p] && high <= front);

		if (low < high)
		{
			SWAP(block[low], block[high]);
		}
	} while (low < high);

	SWAP(block[p], block[high]);

	return high;
}

