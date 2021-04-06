#include "ArrayBaseTree.h"

btree* initBTree(int Size)
{
	btree* temp = (btree*)malloc(sizeof(btree));
	temp->storage = (Node*)calloc(TREE_SIZE, sizeof(Node));
	temp->cur = 1;
	return temp;
}

void addData(btree* block, Node value)
{
	if (block->cur == TREE_SIZE)
	{
		printf("Tree is Full \n");
		return;
	}
	block->storage[block->cur] = value;
	block->cur += 1;
}

void erase_Tree(btree* block)
{
	while (block->cur != 1)
	{
		block->cur -= 1;
	}
}

void FreeTree(btree* block)
{
	erase_Tree(block);
	free(block->storage);
}

int getLeftChild(int Pos)
{
	return Pos * 2;
}

int getRightChild(int Pos)
{
	return Pos * 2 + 1;
}

//순회 기법: Root 위치에 따라 명칭이 달라짐
void PreOrder(const btree* target, int index) // 전위 순회: Root -> Left -> Right, index = 1;
{
	printf("%d   ", target->storage[index]);

	if (target->storage[getLeftChild(index)] != 0 && getLeftChild(index) < TREE_SIZE) // 왼쪽 자식 노드 출력
	{
		PreOrder(target, getLeftChild(index));
	}
	if (target->storage[getLeftChild(index)] != 0 && getRightChild(index) < TREE_SIZE) // 오른쪽 자식 노드 출력
	{
		PreOrder(target, getRightChild(index));
	}
}

void PostOrder(const btree* target, int index) // 후위 순회: Left -> Right -> Root
{
	if (target->storage[getLeftChild(index)] != 0 && getLeftChild(index) < TREE_SIZE) // 왼쪽 자식 노드 출력
	{
		PreOrder(target, getLeftChild(index));
	}
	if (target->storage[getLeftChild(index)] != 0 && getRightChild(index) < TREE_SIZE) // 오른쪽 자식 노드 출력
	{
		PreOrder(target, getRightChild(index));
	}
	printf("%d   ", target->storage[index]);
}

void InOrder(const btree* target, int index) // 중위 순회: Left -> Root -> Right
{
	if (target->storage[getLeftChild(index)] != 0 && getLeftChild(index) < TREE_SIZE) // 왼쪽 자식 노드 출력
	{
		InOrder(target, getLeftChild(index));
	}

	printf("%d   ", target->storage[index]);

	if (target->storage[getLeftChild(index)] != 0 && getRightChild(index) < TREE_SIZE) // 오른쪽 자식 노드 출력
	{
		InOrder(target, getRightChild(index));
	}
}