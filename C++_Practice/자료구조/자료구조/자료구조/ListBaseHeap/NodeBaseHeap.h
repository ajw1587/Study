#pragma once
#include "btnode.h"

typedef struct _HeapTree
{
	btnode* root;
}heap;

heap* initHeap(void);
void insertHeap(heap* block, int item);
void printHEAP(btnode* cursor);
void HeapSort(int*, heap*);
int heapf(heap*);