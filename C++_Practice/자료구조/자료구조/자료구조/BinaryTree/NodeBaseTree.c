#include "NodeBaseTree.h"

/*
	이진 탐색 트리 (Binary Search Tree)
	- 모든 원소는 서로 다른 유일한 값 (KEY)를 갖는다.
	- 왼쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 작다.
	- 오른쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 크다.
	- 각 서브트리 또한 이진 탐색 트리 구조이다.

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

	if (block->root == NULL)							//용도가 달라 if문을 따로 생성
	{													//용도가 달라 if문을 따로 생성
		newNode = setNode(item);						//용도가 달라 if문을 따로 생성
		block->root = newNode;							//용도가 달라 if문을 따로 생성
		return;											//용도가 달라 if문을 따로 생성
	}													//용도가 달라 if문을 따로 생성
	if (bst_isThere(block, item))						//용도가 달라 if문을 따로 생성
	{													//용도가 달라 if문을 따로 생성
		printf("%d already exist Tree \n", item);		//용도가 달라 if문을 따로 생성
		return;											//용도가 달라 if문을 따로 생성
	}													//용도가 달라 if문을 따로 생성
	// 맞는 위치의 노드를 찾아가서 입력
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

void bst_PreOrder(const btnode* cursor) // 전위 순회
{
	printf("%d   ", cursor->Key);
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	return;
}
void bst_PostOrder(const btnode* cursor) // 후위 순회
{
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	printf("%d   ", cursor->Key);
	return;
}
void bst_InOrder(const btnode* cursor) // 중위 순회
{
	if (cursor->Left != NULL) bst_PostOrder(cursor->Left);
	printf("%d   ", cursor->Key);
	if (cursor->Right != NULL) bst_PostOrder(cursor->Right);
	return;
}

void bst_Orderring(const bst* block, orderFunc order) // 함수포인터 define 사용, 어떤 순회방법을 사용할지
{
	order(block->root);
}

void uninstallBST(bst* block)
{
	postOrderDelete(block->root);
}

void postOrderDelete(btnode* cursor) // 트리 전체 할당 해제 함수
{
	if (cursor->Right != NULL) postOrderDelete(cursor->Right);
	if (cursor->Left != NULL) postOrderDelete(cursor->Left);
	free(cursor);
}