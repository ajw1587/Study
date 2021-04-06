#include "myNode.h"

// AVLtree는 이진탐색트리의 개선형
node* initNode(void)
{
	node* temp = (node*)malloc(sizeof(node));

	temp->data = 0;
	temp->height = 0;
	temp->left = NULL;
	temp->right = NULL;

	return temp;
}
void setData(node* block, int _key)
{
	block->data = _key;
}
void viewData(const node* cpnode)
{
	printf("%d", cpnode->data);
}