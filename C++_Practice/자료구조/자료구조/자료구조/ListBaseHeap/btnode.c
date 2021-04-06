#include "btnode.h"

btnode* initbtNode(void)
{
	btnode* temp = (btnode*)malloc(sizeof(btnode));

	temp->Key = 0;
	temp->Left = NULL;
	temp->Right = NULL;
	temp->Parent = NULL;

	return temp;
}
void setKey(btnode* block, int item)
{
	block->Key = item;
}
btnode* setbtNode(int item)
{
	btnode* block = initbtNode();

	block->Key = item;

	return block;
}
btnode* copyNode(btnode* target, const btnode* source)
{
	if (target != NULL)
	{
		free(target);
	}
	
	target = initbtNode();
	target->Key = source->Key;
	target->Left = source->Left;
	target->Right = source->Right;
	target->Parent = source->Parent;
	
	return target;
}
void printbtNode(const btnode* target)
{
	printf("%2d", target->Key);
}

int countChild(const btnode* target)
{
	if (target->Left == NULL) return 0;
	else if (target->Right == NULL) return 1;
	else return 2;
}