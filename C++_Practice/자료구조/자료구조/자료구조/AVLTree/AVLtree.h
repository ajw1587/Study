#pragma once

// AVLtree는 이진탐색트리의 개선형
#include "myNode.h"

typedef struct _AVLTREE
{
	node* root;
}Avltree;

Avltree* initAvl(void);

node* LL_Rotate(node*);
node* RR_Rotate(node*);
node* LR_Rotate(node*);
node* RL_Rotate(node*);

int getHeight(node*);
int heightDiff(node*);

void ReBalance(Avltree*);
void insertData(Avltree*, int);

typedef void orderfunc(node*);

void orderring(Avltree*, orderfunc order);

void PreOrder(node*);
void PostOrder(node*);
void InOrder(node*);