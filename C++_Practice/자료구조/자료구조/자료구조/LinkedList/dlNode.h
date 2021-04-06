#pragma once

#include "framework.h"

typedef struct dlNode
{
	int m_nData;
	struct dlNode* m_pNext; // 다음 원소의 시작 주소
	struct dlNode* m_pPrev; // 이전 원소의 시작 주소
}node;

bool isNonAlloc(const node* _cpNode);
node* initNode(void);
bool setNodeData(node* pnode, int value);
int deleteNode(node* pnode);
void viewNode(const node* cpnode);