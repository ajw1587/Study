#pragma once

#include "framework.h"

typedef struct dlNode
{
	int m_nData;
	struct dlNode* m_pNext; // ���� ������ ���� �ּ�
	struct dlNode* m_pPrev; // ���� ������ ���� �ּ�
}node;

bool isNonAlloc(const node* _cpNode);
node* initNode(void);
bool setNodeData(node* pnode, int value);
int deleteNode(node* pnode);
void viewNode(const node* cpnode);