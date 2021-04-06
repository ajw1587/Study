#pragma once
#include "Tnode.h"

typedef struct _TreeStruct {
	tnode* root;
	tnode* now;
}tree;

tree* initTree(const char* value);

void addNode(tnode * block, const char * value);

tnode* moveNode(tnode** pNow);

void printChildren(tnode* pNow);

void viewElements(tree* pTree);
