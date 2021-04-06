#pragma once
#include "btNode.h"

typedef struct binarySearchTree {
	btnode* root;
}bst;

typedef void orderFunc(btnode* cursor);




bst* initBSTree(void);
bool bst_isThere(bst* block, int item);
void bst_addNode(bst* block, int item);
void bst_PreOrder (btnode* cursor);
void bst_PostOrder(btnode* cursor);
void bst_InOrder(btnode* cursor);

void bst_Orderring(const bst *, orderFunc);

void uninstallBST(bst*);
