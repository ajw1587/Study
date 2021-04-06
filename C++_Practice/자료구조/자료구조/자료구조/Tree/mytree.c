#include "mytree.h"

tree* initTree(const char* value)
{
	tree* tempTree = (tree*)malloc(sizeof(tree));
	tempTree->root = NULL;
	setData(tempTree->root, value);
	tempTree->now = NULL;

	return tempTree;
}

void addNode(tree* block, char* value)
{
	tnode* cursor = NULL;
	cursor = block;

	if (cursor->m_pChild == NULL)
	{
		cursor->m_pChild = initNode();
		setData(cursor->m_pChild, value);
		cursor->m_pChild->m_pParent = cursor;
	}
	else
	{
		cursor = cursor->m_pChild;

		while (cursor->m_pSibling != NULL)
		{
			cursor = cursor->m_pSibling;
		}
		cursor->m_pSibling = initNode();
		setData(cursor->m_pSibling, value);
		cursor->m_pSibling->m_pParent = cursor->m_pParent;
	}
}

tnode* moveNode(tnode** pNow)
{
	char enter = 0;
	tnode* cursor = *pNow;
	printChildren(cursor);

	do
	{
		gechar(enter);
		switch (enter)
		{
		case 'd': case 'D':
			if (cursor->m_pSibling == NULL)
			{
				printf("�����Ͱ� �����ϴ�.");
				break;
			}
			else
			{
				cursor = cursor->m_pSibling;
				printf("%3s", cursor->m_sData);
				break;
			}
		case 's': case 'S':
			if (cursor->m_pChild == NULL)
			{
				printf("�����Ͱ� �����ϴ�.");
				break;
			}
			else
			{
				cursor = cursor->m_pChild;
				printf("%3s", cursor->m_sData);
				break;
			}
		case 'w': case 'W':
			if (cursor->m_pParent == NULL)
			{
				printf("�����Ͱ� �����ϴ�.");
				break;
			}
			else
			{
				cursor = cursor->m_pParent;
				printf("%3s", cursor->m_sData);
				break;
			}
		case 'm': case 'M':
			*pNow = cursor;
			break;
		}
	} while (enter == 'q' || enter == 'Q');

	return cursor;
}

void printChildren(tnode* pNow)
{
	tnode* cursor = pNow->m_pChild;
	printf("���� ���"); viewData(pNow);

	if (cursor == NULL) printf("\n");
	else
	{
		printf("���� ���");
		while (cursor->m_pSibling != NULL)
		{
			viewData(pNow);
			cursor = cursor->m_pSibling;
		}
	}
	return;
}

void innerViewer(const tnode* cursor)
{
	viewData(cursor);
	if (cursor->m_pChild != NULL)
	{
		printf("{");
		innerViewer(cursor->m_pChild);
	}
	printf(", ");
	if (cursor->m_pSibling != NULL)
	{
		innerViewer(cursor->m_pSibling);
		printf("}");
	}
}

void viewElements(tree* pTree)
{
	printf("\n << Ʈ���� ��� ����>>\n");
	innerViewer(pTree->root);
}

