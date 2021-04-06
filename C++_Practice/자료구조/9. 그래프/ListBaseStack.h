#pragma once
#include "dlNode.h"

typedef struct _ListBasedStack {
	node* m_pHead;
	node* m_pTail;
	int m_nSize;
	int m_nCapacity;
}l_stack; // list based stack

/*	Stack  : Last In First Out 
*/

l_stack* initStackList(int );
void Push_Back(l_stack*, int);
int Pop_Back(l_stack *);
int List_Seek(const l_stack*);