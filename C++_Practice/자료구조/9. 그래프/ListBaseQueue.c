#include "ListBaseQueue.h"
l_queue* initQueue(int Capacity) {
	l_queue* temp = (l_queue*)malloc(sizeof(l_queue));
	temp->m_nCapacity = Capacity;
	temp->m_nSize = 0;
	temp->m_pHead = initNode();
	temp->m_pTail = initNode();
	temp->m_pHead->m_pNext = temp->m_pTail;
	temp->m_pTail->m_pPrev = temp->m_pHead;
	return temp;
}


void QPush_Back(l_queue* _pQ, int _Value) {
	node* Cursor = _pQ->m_pTail->m_pPrev;
	node* newNode = initNode();
	setNodeData(newNode, _Value);

	newNode->m_pPrev = Cursor;
	newNode->m_pNext = Cursor->m_pNext; //(TAIL)

	Cursor->m_pNext->m_pPrev = newNode;
	Cursor->m_pNext = newNode;
	_pQ->m_nSize += 1;

	return;
}
int QPop_front(l_queue* pQ) {
	node* cursor = pQ->m_pHead->m_pNext;
	int data = cursor->m_nData;

	cursor->m_pNext->m_pPrev = cursor->m_pPrev;
	cursor->m_pPrev->m_pNext = cursor->m_pNext;
	free(cursor);
	return data;

}