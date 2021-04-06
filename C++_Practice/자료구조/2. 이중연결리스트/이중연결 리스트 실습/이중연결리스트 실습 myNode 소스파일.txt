#include "myNode.h"

node* initNode() {
	node* temp = (node*)malloc(sizeof(node));
	memset(temp->m_sPage, 0, PageSize);
	temp->next = NULL;
	temp->prev = NULL;

	return temp;
}

bool setNodeData(node* pnode, FILE * fin) {
	int i = 0;
	while (i < PageSize-1 && !feof(fin)) {
		pnode->m_sPage[i] = fgetc(fin);
		/*
		if (i > 0) {
			if (strcmp(pnode->m_sPage + (i - 1), "\n\n") == 0)break;
		}*/
		i++;
	}
	return true;
}
void deleteData(node* pnode) {
	free(pnode);
	return;
}
void viewNode(const node* pnode) {
	printf("%s", pnode->m_sPage);
}