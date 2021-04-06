#pragma once
#include "framework.h"

typedef struct _Heap {
	/*¢¯IAuAIA©ª ¨¡¢ç¢¬¢çAC ¡¾¢¬Co*/
	int * Hstorage;
	int depth;
	int cur;
}heap;


heap* createHeap(int Tree_depth);
void addData(heap* block, int item);
int mvLeftChild(int locate);
int mvRightChild(int locate);
int mvParent(int locate);

int popData(heap * block);