#pragma once
#include "btnode.h"

typedef struct _QNode
{
	btnode** data;
	struct _QNode* next;
	struct _QNode* prev;

}qnode;

typedef struct _ListBaseQ
{
	qnode* head;
	qnode* tail;

}LbQUEUE;

qnode* initNode(void);
LbQUEUE* initQueue(void);
void printQueue(const LbQUEUE* block);
void pushBack(LbQUEUE* block, btnode**);
btnode** popFront(LbQUEUE*);
void clearQueue(LbQUEUE*);
void uninstall(LbQUEUE*);

void innerUninstall(qnode* start, qnode* end);

btnode** SeekFirst(const LbQUEUE* block);