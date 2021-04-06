#pragma once
#include "../Project01_SingleLinkedList/framework.h"

typedef struct _TNode {
	char m_sData[30];
	struct _TNode* m_pParent;
	struct _TNode* m_pSibling;
	struct _TNode* m_pChild;
}tnode;

tnode* initNode();
void setData(tnode * block , const char * Value);
void viewData(const tnode* block);

tnode* getParent(const tnode* block);
tnode* getSibling(const tnode* block);
tnode* getChild(const tnode* block);