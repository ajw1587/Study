#pragma once

/*
	���� Ž�� Ʈ�� (Binary Search Tree)
	- ��� ���Ҵ� ���� �ٸ� ������ �� (KEY)�� ���´�.
	- ���� ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� �۴�.
	- ������ ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� ũ��.
	- �� ����Ʈ�� ���� ���� Ž�� Ʈ�� �����̴�.

*/

#include "btNode.h"

typedef struct binarySearchTree
{
	btnode* root;
}bst;

typedef void orderFunc(btnode* cursor); // �ڷ��� ����: void�� ���Ͻ�Ű�� �Ű������� ����� �Լ��� �����ϴ� Ÿ�� orderFunc�� Ÿ�Ը��� �ȴ�.

bst* initBStree(void);
bool bst_isThere(bst* block, int item);
void bst_addNode(bst* block, int item);

void bst_PreOrder(const btnode* cursor); // ���� ��ȸ
void bst_PostOrder(const btnode* cursor); // ���� ��ȸ
void bst_InOrder(const btnode* cursor); // ���� ��ȸ

void bst_Orderring(const bst* block, orderFunc);

void uninstallBST(bst* block); // Ʈ�� ��ü ����
void postOrderDelete(btnode* cursor);