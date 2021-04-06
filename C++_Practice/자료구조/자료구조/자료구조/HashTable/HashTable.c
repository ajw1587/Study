#include "HashTable.h"

List* initList(void)
{
	List* temp = (List*)malloc(sizeof(List));
	temp->head = initNode();
	temp->tail = initNode();

	temp->head->pNext = temp->tail;
	temp->tail->pPrev = temp->head;

	return temp;
}
void push_back(List* block, const char* const _name)
{
	node* newNode = initNode();
	node* cursor = block->tail->pPrev;

	setNode(newNode, _name);

	newNode->pNext = cursor->pNext;
	newNode->pPrev = cursor;

	cursor->pNext->pPrev = newNode;
	cursor->pNext = newNode;
}

void push_front(List* block, const char* const _name)
{
	node* newNode = initNode();
	node* cursor = block->head->pPrev;

	setNode(newNode, _name);

	newNode->pNext = cursor;
	newNode->pPrev = cursor->pPrev;

	cursor->pPrev->pNext = newNode;
	cursor->pPrev = newNode;
}
const char* pop_back(List* block)
{
	node* dcur = block->tail->pPrev;
	char* rt = (char*)malloc(strlen(dcur->Name) + 1);
	strcpy(rt, dcur->Name);

	dcur->pNext->pPrev = dcur->pNext;
	dcur->pPrev->pNext = dcur->pPrev;
	freeNode(dcur);

	return rt;
}
const char* pop_front(List* block)
{
	node* dcur = block->head->pNext;
	char* rt = (char*)malloc(strlen(dcur->Name) + 1);
	strcpy(rt, dcur->Name);

	dcur->pNext->pPrev = dcur->pNext;
	dcur->pPrev->pNext = dcur->pPrev;
	freeNode(dcur);

	return rt;
}

bool SaveList(FILE* fout, const List* block)
{
	if (fout == NULL || block == NULL) return false;
	node* cur = block->head->pNext;
	while(cur != block->tail)
	{
		pushFile(fout, cur);
	}
	return true;
}
void freeList(List* block)
{
	innerFL(block->head, block->tail);
}

void innerFL(node* snode, node* lnode)
{
	if (snode != lnode)
	{
		innerFL(snode->pNext, lnode);
	}
	freeNode(snode);
}
void viewListElements(const List* const clist)
{
	node* prt = clist->head->pNext;
	while (prt != clist->tail)
	{
		viewData(prt);
		prt = prt->pNext;
	}
	return;
}