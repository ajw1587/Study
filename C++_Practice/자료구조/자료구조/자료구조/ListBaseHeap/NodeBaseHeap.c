#include "NodeBaseHeap.h"
#include "Queue_LBH.h"
#include "Stack_LBH.h"

heap* initHeap(void)
{
	heap* temp = (heap*)malloc(sizeof(heap));

	temp->root = NULL;
	return temp;
}
void insertHeap(heap* block, int item)
{
	btnode* cursor = NULL;
	btnode* newNode = NULL;
	btnode* child = NULL;
	btnode* parent = NULL;
	LbQUEUE* Level = initQueue();

	if (block->root == NULL)
	{
		newNode = setbtNode(item);
		block->root = newNode;
		return;
	}

	cursor = block->root;
	newNode = setbtNode(item);
	pushBack(Level, &cursor);

	while (1)
	{
		cursor = *popFront(Level); // 레벨에서 데이터를 뽑는다

		if (countChild(cursor) == 0)
		{
			cursor->Left = newNode;
			newNode->Parent = cursor;
			child = cursor->Left;
			uninstall(Level);
			break;
		}
		else if (countChild(cursor) == 1)
		{
			cursor->Right = newNode;
			newNode->Parent = cursor;
			child = cursor->Right;
			uninstall(Level);
			break;
		}
		else //countChild(cursor) == 2
		{
			pushBack(Level, &cursor->Left);
			pushBack(Level, &cursor->Right);
		}
	}

	// 힙에서 데이터 정렬하는 부분
	parent = cursor;
	int temp = 0;
	while (parent != NULL && child->Key < parent->Key)
	{
			temp = child->Key;
			child->Key = parent->Key;
			parent->Key = temp;
			child = parent;
			parent = parent->Parent;
	}
}

void printHEAP(btnode* cursor)
{
	printbtNode(cursor);
	
	if (cursor->Left != NULL)
	{
		printf(" { ");
		printHEAP(cursor->Left);
	}
	if (cursor->Right != NULL)
	{
		printf(" , ");
		printHEAP(cursor->Right);
		printf(" } ");
	}
}



//////////////Stack Heap Sort//////////////
void HeapSort(int* arr, heap* block)
{
	int result = 0;
	int index = 0;

	while (1)
	{
		result = heapf(block);
		if (result == -1) break;
		arr[index] = result;
		index++;
	}

	return;
}
int heapf(heap* block)
{
	if (block->root == NULL) return -1;

	int rt = 0;
	btnode* cursor = block->root;
	LbStack* stk = initStack();
	LbQUEUE* que = initQueue();
	btnode* del = NULL;

	if (block->root->Left == NULL && block->root->Right == NULL)
	{
		rt = block->root->Key;
		free(block->root);
		block->root = NULL;
		return rt;
	}

	rt = cursor->Key;
	pushBack(que, &cursor);

	do
	{
		cursor = *popFront(que);
		SpushBack(stk, &cursor);

		if (cursor->Left != NULL) pushBack(que, &cursor->Left);
		else break;
		if (cursor->Right != NULL) pushBack(que,& cursor->Right);
		else break;

	} while (1);

	//////////////////////////////////////////////////////// 마지막에 들어온 데이터 뽑기
	while (que->head->next != que->tail)
	{
		SpushBack(stk, popFront(que));
	}
	uninstall(que);

	// 마지막 노드값을 root -> Key 에 저장 후 노드 해제
	del = (*SpopBack(stk));
	block->root->Key = (*SpopBack(stk))->Key;
	uninstallStack(stk);
	if (countChild(del->Parent) == 2)
		del->Parent->Right = NULL;
	else if (countChild(del->Parent) == 1)
		del->Parent->Left = NULL;
	free(del);
	//////////////////////////////////////////////////////////////////////////////////
	

	int temp = 0;
	btnode* child = NULL;
	cursor = block->root;

	while (cursor != NULL)
	{
		if (cursor->Right != NULL)
		{
			child = cursor->Left->Key < cursor->Right->Key ? cursor->Left : cursor->Right;
		}
		else if (cursor->Left == NULL)
		{
			break;
		}
		else
		{
			child = cursor->Left;
		}

		if (cursor->Key > child->Key)
		{
			temp = cursor->Key;
			cursor->Key = child->Key;
			child->Key = temp;
			cursor = child;
		}
		else break;
	}

	return rt;
}