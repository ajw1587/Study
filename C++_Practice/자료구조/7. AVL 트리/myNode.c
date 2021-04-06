#include "myNode.h"

node* initNode(void) {
	node* temp = (node*)malloc(sizeof(node));
	temp->data = 0; 
	temp->height = 0; 
	temp->left = NULL;
	temp->right = NULL;
	return temp;
}
void setData(node* _block, int _key) {
	_block->data = _key;
}

void viewData(const node* cpnode) {
	printf("%d", cpnode->data);
}