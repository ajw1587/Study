#include "DoubleLinkedList.h"

dlist* createList(int capacity)
{
	dlist* locallist = (dlist*)malloc(sizeof(dlist));
	locallist->m_pHead = initNode();
	locallist->m_pTail = initNode();

	return locallist;
}
bool ListifEmpty(const dlist*)
bool ListisFull(const dlist*)

void appendList(dlist*, int)
void viewNode(dlist* cplist)
int FindDelNode(dlist* cplist)

void insertlist(dlist*, int, int)