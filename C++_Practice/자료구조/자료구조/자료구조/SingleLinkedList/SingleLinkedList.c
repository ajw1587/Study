#include "SingleLinkedList.h"

slist* createList(int capacity)
{
	slist* localList = (slist*)malloc(sizeof(slist));
	localList->phead = initNode();
	localList->pTail = initNode();
	localList->m_nCapacity = capacity;
	localList->m_nSize = 0;

	localList->phead->m_pNext = localList->pTail;

	return localList;
}
bool ListifEmpty(const slist* cplist)
{
	if (cplist->m_nSize == 0) return true;
	if (cplist->phead->m_pNext == cplist->pTail)
		return true;
	else return false;
}
bool ListisFull(const slist* cplist)
{
	if (cplist->m_nSize == cplist->m_nCapacity)
		return true;
	else return false;
}

void appendList(slist* cplist, int value)
{
	node* lcursor = NULL;
	node* newnode = (node*)malloc(sizeof(node));

	lcursor = cplist->phead->m_pNext;

	while (lcursor->m_pNext != cplist->pTail)
	{
		lcursor = lcursor->m_pNext;
	}
	newnode->m_pNext = lcursor->m_pNext;
	lcursor->m_pNext = newnode;
	
}

void viewNode(slist* cplist)
{
	int data = 0;
	node* lcursor = NULL;

	lcursor = cplist->phead->m_pNext;

	while (lcursor->m_pNext != cplist->pTail)
	{
		printf("%5d \n", lcursor->m_nData);
		lcursor = lcursor->m_pNext;
	}
}

int FindDelNode(slist* cplist)
{
	node* lcursor = NULL;
	node* delnode = NULL;
	int data;

	printf("삭제할 데이터를 입력하여 주십시오: ");
	scanf(&data);

	lcursor = cplist->phead;

	while (lcursor->m_pNext->m_nData != data)
	{
		lcursor = lcursor->m_pNext;
	}

	delnode = lcursor->m_pNext;
	lcursor->m_pNext = delnode->m_pNext;

	data = delnode->m_nData;

	(cplist->m_nSize)--;
	free(delnode);
	return data;
}

void insertlist(slist* plist , int position, int value)
{
	node* lcursor = NULL;
	node* newnode = (node*)malloc(sizeof(node));

	setNodeData(newnode, value);

	lcursor = plist->phead;
	for(int i = 0; i < position-1; i++)
		lcursor = lcursor->m_pNext;

	newnode->m_pNext = lcursor->m_pNext;
	lcursor->m_pNext = newnode;

}