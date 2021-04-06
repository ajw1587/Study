#pragma once
#include "btnode.h"

typedef struct _FULL_BINARY_TREE {
	btnode* Root;
	btnode* last;

}fbtree;

fbtree* initFBtree(void);
void addItemp(fbtree* block, int item);