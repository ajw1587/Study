#pragma once
#include "../myheader/dlNode.h"

typedef struct _ListBasedQueue {
	node* m_pHead;
	node* m_pTail;
	int m_nSize;
	int m_nCapacity;
}l_queue; // list based stack

/*	Queue  : First In First Out
*/
l_queue* initQueue(int);
void QPush_Back(l_queue*, int);
int QPop_front(l_queue*);