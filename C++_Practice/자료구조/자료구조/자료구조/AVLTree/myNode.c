#include "myNode.h"

// AVLtree�� ����Ž��Ʈ���� ������
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