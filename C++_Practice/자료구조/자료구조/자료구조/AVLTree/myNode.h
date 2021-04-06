#pragma once
#define _CRT_SECURE_NO_WARNINGS

// AVLtree�� ����Ž��Ʈ���� ������
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct _NODE
{
	int data;
	int height;
	struct _NODE* left;
	struct _NODE* right;
}node;

node* initNode(void);
void setData(node* block, int _key);
void viewData(const node* cpnode);