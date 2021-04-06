#include "dlNode.h"


bool isNonAlloc(const node* _cpNode) {
	return _cpNode == NULL;
}
node* initNode(void) {
	node* localNode = NULL;
	if (isNonAlloc(localNode)) {
		localNode = (node*)malloc(sizeof(node));
		localNode->m_nData = 0;
		localNode->m_pNext = NULL;
		//double linked list
		localNode->m_pPrev = NULL;
		return localNode;
	}
	else return NULL;
}




bool setNodeData(node* pnode, int value) {
	if (isNonAlloc(pnode)) return false;
	pnode->m_nData = value;
	return true;
}
int deleteData(node* pnode) {
	if (isNonAlloc(pnode)) return false;
	int data = 0;
	data = pnode->m_nData;
	//TODO :: 여기에 할당 해제 코드를 작성합니다.

	return data;
}
void viewNode(const node* cpnode) {
	if (isNonAlloc(cpnode)) return;
	printf("%5d", cpnode->m_nData);
}