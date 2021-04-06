#include "myNode.h"

node* initNode(void)
{
	node* temp = (node*)malloc(sizeof(node));
	memset(temp->Name, 0, 30);
	temp->pNext = NULL;
	temp->pPrev = NULL;
	return temp;
}

node* setNode(node* _block, const char* const _str)
{
	strcpy(_block->Name, _str);
	return _block;
}

void freeNode(node* block)
{
	if (block != NULL)
		free(block);
}

void viewData(const node* _cblock)
{
	printf("%-30", _cblock->Name);
}

bool pushFile(FILE* fout, const node* _cblock)
{
	if (fout == NULL || _cblock == NULL) return false;
	fwrite(_cblock->Name, strlen(_cblock->Name), 1, fout);
	return true;
}