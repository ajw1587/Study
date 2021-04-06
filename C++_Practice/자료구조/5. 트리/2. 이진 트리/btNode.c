#include "btNode.h"

btnode* initNode() {
	btnode* temp = (btnode*)malloc(sizeof(btnode));
	temp->Left = NULL;
	temp->Right = NULL;
	temp->Key = 0;
	return temp;
}
void setData(btnode* block, int item) {
	block->Key = item;
}
btnode* setNode(int item) {
	btnode* temp = initNode();
	setData(temp, item);
	return temp;
}