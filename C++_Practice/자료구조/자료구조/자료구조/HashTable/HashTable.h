#pragma once
#include "myNode.h"

typedef struct
{
	node* head;
	node* tail;
}List;

List* initList(void);
void push_back(List*, const char* const);
const char* pop_back(List*);
void push_front(List*, const char* const);
const char* pop_front(List*);

void innerFL(node* snode, node* lnode);
bool SaveList(FILE*, const List*);
void freeList(List*);

void viewListElements(const List* const);