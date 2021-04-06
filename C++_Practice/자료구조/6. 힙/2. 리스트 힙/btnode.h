#pragma once
#include "framework.h"

typedef struct _BinaryTreeNode {
	int Key;
	struct _BinaryTreeNode* Left;
	struct _BinaryTreeNode* Right;
	struct _BinaryTreeNode* Parent;
}btnode;

btnode* initbtNode();
void setKey(btnode * target, int item);
btnode* setbtNode(int item);
btnode* copybtNode(btnode * target, const btnode* source);
void printbtNode(const btnode* target);

int countChild(const btnode* target);