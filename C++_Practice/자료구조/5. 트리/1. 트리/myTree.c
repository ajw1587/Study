#include "myTree.h"
/*
	트리의 모든 노드에 일정한 순서로 접근한다. : 트리의 순회 
	A { B { D, E, F{I, J} } , C {G, H}};
	A { B parent->Child		  J} Child 종료 / Sibling = NULL
	B { D parent->Child		F..} Child 종료 / Sibling = NULL
	F { I parent->Child		  H} Child 종료 / Sibling = NULL
	C { G parent->Child		C..} Child 종료 / Sibling = NULL
	, E	Sibling->Sibling
	, F	Sibling->Sibling
	, J	Sibling->Sibling
	, C	Sibling->Sibling
	, H Sibling->Sibling
*/
tree* initTree(const char* value) {
	tree* tempTree = (tree*)malloc(sizeof(tree));
	tempTree->root = initNode();
	setData(tempTree->root, value);
	tempTree->now = tempTree->root;
	
	return tempTree;
}



void addNode(tnode * block,const char * value) {
	tnode* cursor = NULL;
	cursor = block;

	if (cursor->m_pChild == NULL) {
		cursor->m_pChild = initNode();
		setData(cursor->m_pChild, value);
		cursor->m_pChild->m_pParent = cursor;
	}
	else {
		
		cursor = cursor->m_pChild;
		
		while (cursor->m_pSibling != NULL) {
			cursor = cursor->m_pSibling;
		}
		cursor->m_pSibling = initNode();
		setData(cursor->m_pSibling, value);
		cursor->m_pSibling->m_pParent = cursor->m_pParent;
	}
	
}




tnode* moveNode(tnode** pNow) {
	tnode* cursor = *pNow;
	int c = 0;
	
	do {
		printChildren(cursor);
		printf("\nselect > ");
		c = getch();
		switch (c) {
		case 'w':case 'W':
			if (cursor->m_pParent != NULL) {
				cursor = cursor->m_pParent;
				*pNow = cursor;
			}
			break;
		case 's':case 'S':
			if (cursor->m_pChild != NULL)
				cursor = cursor->m_pChild;
			break;
		case 'd':case 'D':
			if (cursor->m_pSibling != NULL)
				cursor = cursor->m_pSibling;
			break;
		case 'n':case 'N':
			*pNow = cursor;
			return cursor;
		}
		system("cls");
	} while (1);
	return cursor;
}

void printChildren(tnode* pNow) {
	tnode* cursor = pNow->m_pChild;
	printf("\n현재 노드"); viewData(pNow);
	if (cursor == NULL) printf("(NULL)\n");
	else {
		printf("\n내부 노드");
		while (cursor != NULL) {
			viewData(cursor);
			cursor = cursor->m_pSibling;
		}
	}
	return;
}

void innerViewer(const tnode* cursor) {
	viewData(cursor);
	if (cursor->m_pChild != NULL) {
		printf("{ ");
		innerViewer(cursor->m_pChild);
	}
	
	if (cursor->m_pSibling != NULL) {
		printf(", ");
		innerViewer(cursor->m_pSibling);
		printf("}");
	}

}
void viewElements(tree* pTree) {
	printf("\n<<트리의 모든 원소>>\n");
	innerViewer(pTree->root);
}