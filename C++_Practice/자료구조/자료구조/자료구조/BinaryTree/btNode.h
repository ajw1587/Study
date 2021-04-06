#pragma once

#include "../LinkedList/framework.h"

/*
	이진 탐색 트리 (Binary Search Tree)
	- 모든 원소는 서로 다른 유일한 값 (KEY)를 갖는다.
	- 왼쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 작다.
	- 오른쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 크다.
	- 각 서브트리 또한 이진 탐색 트리 구조이다.
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