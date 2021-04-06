#include "Tnode.h"

tnode* initNode(void)
{
	tnode* temp = (tnode*)malloc(sizeof(tnode));
	memset(temp->m_sData, 0, 30);

	temp->m_pChild = NULL;
	temp->m_pParent = NULL;
	temp->m_pSibling = NULL;

	return temp;
}

void setData(tnode* block, char* value)
{
	if (block == NULL) return;
	else
	{
		strcpy(block->m_sData, value);
		return;
	}
}

void viewData(const tnode* block)
{
	printf("%s", block->m_sData);
}

tnode* getParent(const tnode* block)
{
	if (block->m_pParent == NULL) return block;
	else return block->m_pParent;
}
tnode* getSibling(const tnode* block)
{
	if (block->m_pSibling == NULL) return block;
	else return block->m_pSibling;
}
tnode* getchild(const tnode* block)
{
	if (block->m_pChild == NULL) return block;
	else return block->m_pChild;
}