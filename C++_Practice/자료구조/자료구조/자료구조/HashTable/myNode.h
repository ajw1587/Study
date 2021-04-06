#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct _Node
{
	char Name[30];
	struct _node* pNext;
	struct _node* pPrev;
}node;

node* initNode(void);
node* setNode(node* _block, const char* const _str);
void freeNode(node* block);
void viewData(const node* _cblock);

bool pushFile(FILE* fout, const node* _cblock);