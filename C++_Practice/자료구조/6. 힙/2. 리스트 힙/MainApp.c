#include "NodeBaseHeap.h"

int main() {

	heap* test = initHeap();
	int arr[10] = { 0 };
	insertHeap(test, 10);
	insertHeap(test, 7);
	insertHeap(test, 15);
	insertHeap(test, 9);
	insertHeap(test, 13);
	insertHeap(test, 14);
	insertHeap(test, 11);

	printHEAP(test->root);
	HeapSort(arr, test);
	printf("\n");
	for (int i = 0; i < 10; i++) {
		printf(" %2d ", arr[i]);
	}
	printf("\n");

	return 0;
}