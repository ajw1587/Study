#pragma once
#include "myNode.h"

typedef struct _AVLTREE {
	node* root;

}Avltree;

Avltree* initAvl(void);

node* LL_Rotate(node*);
node* RR_Rotate(node*);
node* LR_Rotate(node*);
node* RL_Rotate(node*);

void ReBalance(Avltree*);
void insertData(Avltree*, int);

int getHeight(node*);
int heightDiff(node*);

typedef void orderfunc(node*);

void orderring(Avltree*, orderfunc order);

void PreOrder(node*);
void PostOrder(node*);
void InOrder(node*);