#pragma once
#include "framework.h"

typedef struct _Node {
	int m_nData;
	struct _Node* m_pNext;
}node;

node* initNode(void); 
bool setNodeData(node*, int);
int deleteData(node*); 
void viewNode(const node*);