#include "Stack_LBH.h"

snode* initSNode() {
	snode* temp = (snode*)malloc(sizeof(snode));
	temp->data = NULL;
	temp->next = NULL;
	temp->prev = NULL;
	return temp;
}
LbStack* initStack(void) {
	LbStack* temp = (LbStack*)malloc(sizeof(LbStack));
	temp->head = initSNode();
	temp->tail = initSNode();

	temp->head->next = temp->tail;
	temp->tail->prev = temp->head;
	return temp;
}

void SpushBack(LbStack* block, btnode** item) {
	snode* cursor = block->tail->prev;
	snode* newNode = initSNode();

	newNode->data = item;

	newNode->next = cursor->next;
	newNode->prev = cursor;
	cursor->next->prev = newNode;
	cursor->next = newNode;
}

btnode** SpopBack(LbStack* block) {
	btnode** rt = NULL;
	snode* cursor = block->tail->prev;
	if (cursor == block->tail) { return NULL; }
	rt = cursor->data;

	cursor->prev->next = cursor->next;
	cursor->next->prev = cursor->prev;

	free(cursor);

	return rt;
}
void clearStack(LbStack* block) {
	innerUninstall(block->head->next, block->tail->prev);
}
void uninstallStack(LbStack* block) {
	innerUninstall(block->head, block->tail);
}

void stackinnerUninstall(snode* start, snode* end) {
	if (start != end) {
		innerUninstall(start->next, end);
	}
	free(start);
}