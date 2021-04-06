#pragma once

/*
	이진 탐색 트리 (Binary Search Tree)
	- 모든 원소는 서로 다른 유일한 값 (KEY)를 갖는다.
	- 왼쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 작다.
	- 오른쪽 서브트리의 원소의 값은 항상 그 루트 값 보다 크다.
	- 각 서브트리 또한 이진 탐색 트리 구조이다.

*/

#include "btNode.h"

typedef struct binarySearchTree
{
	btnode* root;
}bst;

typedef void orderFunc(btnode* cursor); // 자료형 정의: void를 리턴시키고 매개변수가 어떤거인 함수를 저장하는 타입 orderFunc가 타입명이 된다.

bst* initBStree(void);
bool bst_isThere(bst* block, int item);
void bst_addNode(bst* block, int item);

void bst_PreOrder(const btnode* cursor); // 전위 순회
void bst_PostOrder(const btnode* cursor); // 후위 순회
void bst_InOrder(const btnode* cursor); // 중위 순회

void bst_Orderring(const bst* block, orderFunc);

void uninstallBST(bst* block); // 트리 자체 삭제
void postOrderDelete(btnode* cursor);