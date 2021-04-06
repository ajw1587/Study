#pragma once
#include "btnode.h"

typedef struct _SNode {
	btnode** data;
	struct _SNode* next;
	struct _SNode* prev;

}snode;

typedef struct _ListBaseStack {
	snode* head;
	snode* tail;

}LbStack;

snode* initSNode();
LbStack* initStack(void);

void SpushBack(LbStack*, btnode**);
btnode** SpopBack(LbStack*);
void clearStack(LbStack*);
void uninstallStack(LbStack*);

void stackinnerUninstall(snode* start, snode* end);