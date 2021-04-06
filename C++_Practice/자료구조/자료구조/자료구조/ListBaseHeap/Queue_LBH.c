#include "Queue_LBH.h"

qnode* initNode(void)
{
	qnode* temp = (qnode*)malloc(sizeof(qnode));
	temp->data = NULL;
	temp->next = NULL;
	temp->prev = NULL;

	return temp;
}
LbQUEUE* initQueue(void)
{
	LbQUEUE* temp = (LbQUEUE*)malloc(sizeof(LbQUEUE));
	temp->head = initNode();
	temp->tail = initNode();

	temp->head->next = temp->tail;
	temp->tail->prev = temp->head;

	return temp;
}
void printQueue(const LbQUEUE* block)
{
	qnode* cursor = block->head->next;
	while (cursor != block->tail)
	{
		printbtNode(*(cursor->data));
		printf(" - ");
		cursor = cursor->next;
	}
}
void pushBack(LbQUEUE* block, btnode** item)
{
	qnode* cursor = block->tail->prev;
	qnode* newNode = initNode();

	newNode->data = item;

	newNode->next = cursor->next;
	newNode->prev = cursor;
	cursor->next->prev = newNode;
	cursor->next = newNode;
}

btnode** popFront(LbQUEUE* block)
{
	btnode** rt = NULL;
	qnode* cursor = block->head->next;
	if (cursor == block->tail) { return NULL; }
	rt = cursor->data;

	cursor->prev->next = cursor->next;
	cursor->next->prev = cursor->prev;
	
	free(cursor);

	return rt;
}

void clearQueue(LbQUEUE* block)
{
	innerUninstall(block->head->next, block->tail->prev);
}

void uninstall(LbQUEUE* block)
{
	innerUninstall(block->head, block->tail);
}

void innerUninstall(qnode* start, qnode* end)
{
	if (start != end)
	{
		innerUninstall(start->next, end);
	}
	
	free(start);
}

btnode** SeekFirst(const LbQUEUE* block)
{
	if (block->head->next != block->tail)
	{
		return block->head->next->data;
	}
	else return NULL;
}