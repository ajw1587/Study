#include "NodeBaseHeap.h"

int main(void)
{
	heap* test = initHeap();
	int arr[20] = { 0 };

	insertHeap(test, 10);
	insertHeap(test, 7);
	insertHeap(test, 15);
	insertHeap(test, 9);
	insertHeap(test, 13);
	insertHeap(test, 14);
	insertHeap(test, 11);
	insertHeap(test, 15);
	insertHeap(test, 20);
	insertHeap(test, 19);
	insertHeap(test, 32);
	insertHeap(test, 17);

	printHEAP(test->root);
	HeapSort(arr, test);

	for (int i = 0; i < 20; i++)
	{
		printf(" %2d ", arr[i]);
	}
	printf("\n");

	return 0;
}