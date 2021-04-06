#include "ListBaseStack.h"
l_stack* initStackList(int Capacity) {
	l_stack* temp = (l_stack*)malloc(sizeof(l_stack));
	temp->m_nCapacity = Capacity;
	temp->m_nSize = 0;
	temp->m_pHead = initNode();
	temp->m_pTail = initNode();
	temp->m_pHead->m_pNext = temp->m_pTail;
	temp->m_pTail->m_pPrev = temp->m_pHead;
	return temp;
}


void Push_Back(l_stack* _pStack, int _Value) {
	node* Cursor = _pStack->m_pTail->m_pPrev;
	node* newNode = initNode();
	setNodeData(newNode, _Value);

	newNode->m_pPrev = Cursor;
	newNode->m_pNext = Cursor->m_pNext; //(TAIL)

	Cursor->m_pNext->m_pPrev = newNode;
	Cursor->m_pNext = newNode;
	_pStack->m_nSize += 1;

	return;
}

int Pop_Back(l_stack* pStack) {
	node* cursor = pStack->m_pTail->m_pPrev;
	int data = cursor->m_nData;

	cursor->m_pNext->m_pPrev = cursor->m_pPrev;
	cursor->m_pPrev->m_pNext = cursor->m_pNext;
	free(cursor);
	return data;
}
int List_Seek(const l_stack* cpstack) {
	return cpstack->m_pTail->m_pPrev->m_nData;
}