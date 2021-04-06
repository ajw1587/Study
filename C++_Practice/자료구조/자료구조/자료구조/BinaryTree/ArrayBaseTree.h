#pragma once

#include "../LinkedList/framework.h"

#define TREE_SIZE 16;
typedef int Node;

typedef struct _BinaryTree
{
	Node* storage; // 1~15 �ε��� ���� 15���� �����͸� ����
	int cur; // ������ �Էµ� �ε���
}btree;

btree* initBTree(int Size);
void addData(btree* block, Node value);

void erase_Tree(btree* block);
void FreeTree(btree* block);

int getLeftChild(int Pos);
int getRightChild(int Pos);

//��ȸ ���: Root ��ġ�� ���� ��Ī�� �޶���
void PreOrder(const btree* target, int index); // ���� ��ȸ: Root -> Left -> Right
void PostOrder(const btree* target, int index); // ���� ��ȸ: Left -> Right -> Root
void InOrder(const btree* target, int index); // ���� ��ȸ: Left -> Root -> Right