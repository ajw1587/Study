#include "NodeBaseTree.h"

/*
	���� Ž�� Ʈ�� (Binary Search Tree)
	- ��� ���Ҵ� ���� �ٸ� ������ �� (KEY)�� ���´�.
	- ���� ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� �۴�.
	- ������ ����Ʈ���� ������ ���� �׻� �� ��Ʈ �� ���� ũ��.
	- �� ����Ʈ�� ���� ���� Ž�� Ʈ�� �����̴�.

*/

bst* initBStree(void)
{
	bst* temp = (bst*)malloc(sizeof(bst));

	temp->root = NULL;

	return temp;
}

bool bst_isThere(bst* block, int item)
{
	btnode* cursor = block->root;
	while (true)
	{
		if (cursor->Key == item)
		{
			return true;
		}
		if (cursor->Key < item)
		{
			if (cursor->Right == NULL)
				return false;
			cursor = cursor->Right;
		}
		else
		{
			if (cursor->Left == NULL)
				return false;
			cursor = cursor->Left;
		}
	}
}

void bst_addNode(bst* block, int item)
{
	btnode* cursor = block->root;
	btnode* newNode = NULL;

	if (block->root == NULL)							//�뵵�� �޶� if���� ���� ����
	{													//�뵵�� �޶� if���� ���� ����
		newNode = setNode(item);						//�뵵�� �޶� if���� ���� ����
		block->root = newNode;							//�뵵�� �޶� if���� ���� ����
		return;											//�뵵�� �޶� if���� ���� ����
	}													//�뵵�� �޶� if���� ���� ����
	if (bst_isThere(block, item))						//�뵵�� �޶� if���� ���� ����
	{													//�뵵�� �޶� if���� ���� ����
		printf("%d already exist Tree \n", item);		//�뵵�� �޶� if���� ���� ����
		return;											//�뵵�� �޶� if���� ���� ����
	}													//�뵵�� �޶� if���� ���� ����
	// �´� ��ġ�� ��带 ã�ư��� �Է�
	newNode = setNode(item);
	cursor = block->root;

	while (1)
	{
		if (cursor->Key < item)
		{
			if (cursor->Right == NULL)
			{
				cursor->Right = newNode;
				return;
			}
			cursor = cursor->Right;
		}
		else //cursor->Key > item
		{
			if (cursor->Left == NULL)
			{
				cursor->Left = newNode;
				return;
			}
			cursor = cursor->Left;
		}
	}
}

void bst_PreOrder(const btnode* cursor) // ���� ��ȸ
{
	printf("%d   ", cursor->Key);
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	return;
}
void bst_PostOrder(const btnode* cursor) // ���� ��ȸ
{
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	printf("%d   ", cursor->Key);
	return;
}
void bst_InOrder(const btnode* cursor) // ���� ��ȸ
{
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	printf("%d   ", cursor->Key);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	return;
}

void bst_Orderring(const bst* block, orderFunc order) // �Լ������� define ���, � ��ȸ����� �������
{
	order(block->root);
}

void uninstallBST(bst* block)
{
	postOrderDelete(block->root);
}

void postOrderDelete(btnode* cursor) // Ʈ�� ��ü �Ҵ� ���� �Լ�
{
	if (cursor->Right != NULL) postOrderDelete(cursor->Right);
	if (cursor->Left != NULL) postOrderDelete(cursor->Left);
	free(cursor);
}