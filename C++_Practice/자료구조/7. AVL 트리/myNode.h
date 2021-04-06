#pragma once
#include "framework.h"
typedef struct _NODE {
	int data; 
	int height;
	struct _NODE * left;
	struct _NODE* right;
}node;

node* initNode(void);
void setData(node*, int);
//node* createNode(int);
void viewData(const node*);