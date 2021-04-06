#include "SingleLinkedList.h"

slist* createList(int capacity) {
	slist* localList = (slist*)malloc(sizeof(slist));
	localList->pHead = initNode();
	localList->pTail = initNode();
	localList->m_nCapacity = capacity;
	localList->m_nSize = 0;

	localList->pHead->m_pNext = localList->pTail;

	return localList;
}
bool ListisEmpty(const slist* cplist) {
	if (cplist->m_nSize == 0) return true;
	if (cplist->pHead->m_pNext == cplist->pTail)
		return true;
	else return false;
}
bool ListisFull(const slist* cplist) {
	if (cplist->m_nSize == cplist->m_nCapacity)
		return true;
	else return false;
}

bool appendList(slist* _plist, int _value) {
	node* lcursor = NULL;
	node* newNode = NULL;
	if (ListisFull(_plist)) return false;

	newNode = initNode();
	setNodeData(newNode, _value);
	lcursor = _plist->pHead;

	while (lcursor->m_pNext != _plist->pTail)
		lcursor = lcursor->m_pNext;
	

	newNode->m_pNext = lcursor->m_pNext;
	lcursor->m_pNext = newNode;

	_plist->m_nSize+= 1;
	return true;
}
void viewList(const slist* cplist) {
	node* cursor = NULL;
	if (ListisEmpty(cplist))return;
	cursor = cplist->pHead->m_pNext;
	while (cursor != cplist->pTail) {
		viewNode(cursor);
		cursor = cursor->m_pNext;
	}
	printf("\n");
	return;
}
int deleteNode(slist* plist, int position) {
	node* posCursor = NULL;
	node* delCursor = NULL;
	int delData = 0;
	if (position > plist->m_nSize) return -1;
	posCursor = plist->pHead;
	for (int i = 0; i < position - 1; i++)
		posCursor = posCursor->m_pNext;

	delCursor = posCursor->m_pNext;
	delData = delCursor->m_nData;
	
	posCursor->m_pNext = delCursor->m_pNext;
	deleteData(delCursor);
	free(delCursor);

	plist->m_nSize -= 1;
	return delData;
}


int seekList(const slist* cplist, int value) {
	node* cursor = cplist->pHead->m_pNext;
	int count = 1;
	while (cursor->m_nData != value && cursor != cplist->pTail) {
		cursor = cursor->m_pNext;
		count++;
	}
	if (count > cplist->m_nSize) return -1;
	else  return count;
}
int pickList(const slist* cplist, int position) {
	node* cursor = NULL;
	if (position > cplist->m_nSize || position <= 0) return -1;
	cursor = cplist->pHead;
	for (int i = 0; i < position; i++) {
		cursor = cursor->m_pNext;
	}
	return cursor->m_nData;
}


bool insertNode(slist* plist, int position, int value) {
	node* newNode = NULL;
	node* cursor = NULL;
	if (ListisEmpty(plist))
		return appendList(plist, value);
	if (position > plist->m_nCapacity || position <= 0) return false;

	newNode = initNode();
	setNodeData(newNode, value);
	cursor = plist->pHead;
	for (int i = 0; i < position - 1; i++)
		cursor = cursor->m_pNext;

	newNode->m_pNext = cursor->m_pNext;
	cursor->m_pNext = newNode;

	plist->m_nSize += 1;
	
	return true;
}
void deAllocList(slist* plist) {
	innerDel(plist->pHead);
}
void innerDel(node* pNode) {
	if (pNode->m_pNext != NULL)
		innerDel(pNode->m_pNext);
	deleteData(pNode);
	free(pNode);

}