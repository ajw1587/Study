#pragma once

#include "../LinkedList/framework.h"

/*
	���� Ž�� Ʈ�� (Binary Search Tree)
	- ��� ���Ҵ� ���� �ٸ� ������ �� (KEY)�� ���´�.
	- ���� ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� �۴�.
	- ������ ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� ũ��.
	- �� ����Ʈ�� ���� ���� Ž�� Ʈ�� �����̴�.
*/

typedef struct Binart_Tree_Node
{
	int Key;
	struct Binary_Tree_Node* Left;
	struct Binary_Tree_Node* Right;
}btnode;

btnode* initNode();
void setData(btnode* block, int item);
btnode* setNode(int item);