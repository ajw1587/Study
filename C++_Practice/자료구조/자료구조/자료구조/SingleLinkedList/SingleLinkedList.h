#pragma once

#include "iNode.h"

typedef struct _SingleLinkedList
{
	node* phead;
	node* pTail;
	size_t m_nCapacity;
	size_t m_nSize;
}slist;

slist* createList(int);
bool ListifEmpty(const slist*);
bool ListisFull(const slist*);

void appendList(slist*, int);
void viewNode(slist* cplist);
int FindDelNode(slist* cplist);

void insertlist(slist*, int, int);