#pragma once
#include "../LinkedList/framework.h"

typedef struct BTNode
{
	int Key;
	struct BTNode* Left;
	struct BTNode* Right;
	struct BTNode* Parent;

}btnode;

btnode* initbtNode(void);
void setKey(btnode* block, int item);
btnode* setbtNode(int item);
btnode* copyNode(btnode* target, const btnode* source);
void printbtNode(const btnode* target);

int countChild(const btnode* target);