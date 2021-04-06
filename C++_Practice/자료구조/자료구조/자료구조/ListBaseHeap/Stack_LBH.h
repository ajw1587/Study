#pragma once
#include "btnode.h"

typedef struct _SNode
{
	btnode** data;
	struct _SNode* next;
	struct _SNode* prev;
}snode;

typedef struct _ListBaseStack
{
	snode* head;
	snode* tail;
}LbStack;

snode* initSNode(void);
LbStack* initStack(void);

void SpushBack(LbStack* block, btnode**);
btnode** SpopBack(LbStack*);
void clearStack(LbStack*);
void uninstallStack(LbStack*);

void StackinnerUninstall(snode* start, snode* end);