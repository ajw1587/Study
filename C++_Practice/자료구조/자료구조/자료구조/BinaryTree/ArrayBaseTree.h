#pragma once

#include "../LinkedList/framework.h"

#define TREE_SIZE 16;
typedef int Node;

typedef struct _BinaryTree
{
	Node* storage; // 1~15 인덱스 까지 15개의 데이터를 저장
	int cur; // 마지막 입력된 인덱스
}btree;

btree* initBTree(int Size);
void addData(btree* block, Node value);

void erase_Tree(btree* block);
void FreeTree(btree* block);

int getLeftChild(int Pos);
int getRightChild(int Pos);

//순회 기법: Root 위치에 따라 명칭이 달라짐
void PreOrder(const btree* target, int index); // 전위 순회: Root -> Left -> Right
void PostOrder(const btree* target, int index); // 후위 순회: Left -> Right -> Root
void InOrder(const btree* target, int index); // 후위 순회: Left -> Root -> Right