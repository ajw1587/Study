#pragma once

#include "dlNode.h" 

typedef struct _DoubleLinkedList
{
	node* m_pHead;
	node* m_pTail;
	size_t m_nCapacity;
	size_t m_nSize;
}dlist;

dlist* createList(int);
bool ListifEmpty(const dlist*);
bool ListisFull(const dlist*);

void appendList(dlist*, int);
void viewNode(dlist* cplist);
int FindDelNode(dlist* cplist);

void insertlist(dlist*, int, int);