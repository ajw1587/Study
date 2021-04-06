#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct _TNode
{
	char m_sData[30];
	struct _TNode* m_pParent;
	struct _TNode* m_pSibling;
	struct _TNode* m_pChild;
}tnode;

tnode* initNode();
void setData(tnode* block, char* value);
void viewData(const tnode* block);

tnode* getParent(const tnode* block);
tnode* getSibling(const tnode* block);
tnode* getchild(const tnode* block);