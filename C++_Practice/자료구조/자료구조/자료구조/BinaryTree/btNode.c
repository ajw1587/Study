/*
	이진 탐색 트리 (Binary Search Tree)
	- 모든 원소는 서로 다른 유일한 값 (KEY)를 갖는다.
	- 왼쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 작다.
	- 오른쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 크다.
	- 각 서브트리 또한 이진 탐색 트리 구조이다.

*/

#include "btNode.h"

btnode* initNode()
{
	btnode* temp = (btnode*)malloc(sizeof(btnode));
	temp->Left = NULL;
	temp->Right = NULL;
	temp->Key = 0;

	return temp;
}

void setData(btnode* block, int item)
{
	block->Key = item;
}

btnode* setNode(int item)
{
	btnode* temp = initNode();
	setData(temp, item);
	return temp;
}