#pragma once

#include "Tnode.h"

typedef struct _TreeStruct
{
	tnode* root;
	tnode* now;
}tree;

tree* initTree(const char* value);

void addNode(tree* block, char* value);

tnode* moveNode(tnode** pNow);

void printChildren(tnode* pNow);

void viewElements(tree* pTree);

void innerViewer(const tnode* cursor);