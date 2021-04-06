#include "myHeap.h"

heap* createHeap(int Tree_depth) {
	int Capacity = 1 << Tree_depth;
	heap* temp = (heap*)malloc(sizeof(heap));
	temp->depth = Tree_depth;
	temp->Hstorage = (int*)calloc(Capacity, sizeof(int));
	temp->cur = 1;// rootData
	return temp;
}

bool isHeapFull(const heap* block) {
	return block->cur == 1 << (block->depth);
}
bool isHeapEmpty(const heap* block) {
	return block->cur == 1;
}

void addData(heap* block, int item) {
	int index = 0;
	int parent = 0;
	int temp = 0;
	if (isHeapFull(block)) return;
	else {
		index = block->cur;
		// 현재 입력된 위치와 그 부모인덱스를 비교해서 값을 교환 
		block->Hstorage[block->cur] = item;
		while (index != 1) {
			parent = mvParent(index);
			if (block->Hstorage[parent] > block->Hstorage[index]) {
				temp = block->Hstorage[parent];
				block->Hstorage[parent] = block->Hstorage[index];
				block->Hstorage[index] = temp;
				index = parent;
			}
			else {
				break;
			}
		}
		block->cur += 1;
		return;
	}
}
int popData(heap* block) {
	int data = block->Hstorage[1];
	
	int index = 0;
	int parent = 1;
	int last = block->cur - 1;
	int temp = 0;

	block->Hstorage[parent] = block->Hstorage[last];
	block->Hstorage[last] = 0;
	block->cur -= 1;
	last = block->cur - 1;

	while (1){
		if (block->Hstorage[mvRightChild(parent)] == 0) {
			index = mvLeftChild(parent);
		}
		else if(block->Hstorage[mvRightChild(parent)] > block->Hstorage[mvLeftChild(parent)])
			index = mvLeftChild(parent);
		else {
			index = mvRightChild(parent);
		}

		if (index > last) {
			break;
		}
		if (block->Hstorage[parent] > block->Hstorage[index]) {
			temp = block->Hstorage[parent];
			block->Hstorage[parent] = block->Hstorage[index];
			block->Hstorage[index] = temp;
			parent = index;
		}
		else {
			break;
		}
	}
	

	return data;
}


int mvLeftChild(int locate)		{return locate * 2;		}
int mvRightChild(int locate)	{return locate * 2 + 1;	}
int mvParent(int locate)		{return locate / 2;		}