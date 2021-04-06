#pragma once

#include "framework.h"

typedef struct _Node
{
	int m_nData;
	struct _Node* m_pNext;
}node;

bool isNonAlloc(const node* _cpNode);
node* initNode(void);
bool setNodeData(node*, int);
int deleteNode(node*);
void viewNode(const node*);