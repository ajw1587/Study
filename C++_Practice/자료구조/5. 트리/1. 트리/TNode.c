#include "Tnode.h"

tnode* initNode() {
	tnode* tempNode = (tnode*)malloc(sizeof(tnode));
	memset(tempNode->m_sData, 0 , 30);

	tempNode->m_pParent = NULL;
	tempNode->m_pSibling = NULL;
	tempNode->m_pChild = NULL;
	return tempNode;
}
void setData(tnode* block, const char * Value) {
	if (block == NULL) return;
	else {
		strcpy(block->m_sData,Value);
		return;
	}
}

void viewData(const tnode* block) {
	printf("%s, ", block->m_sData);
}
tnode* getParent(const tnode* block) {
	if (block->m_pParent == NULL) return block;
	else return block->m_pParent;
}
tnode* getSibling(const tnode* block) {
	if (block->m_pSibling == NULL) return block;
	else return block->m_pSibling;
}
tnode* getChild(const tnode* block) {
	if (block->m_pChild == NULL) return block;
	else return block->m_pChild;
}