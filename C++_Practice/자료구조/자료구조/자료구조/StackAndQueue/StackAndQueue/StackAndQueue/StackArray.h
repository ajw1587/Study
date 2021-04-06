#pragma once

#include "../../LinkedList_0530/Practice_0530/framework.h"
#include <stdbool.h>

typedef struct StackArray
{
	char* Storage;
	int Top;
	int Capacity;
}s_array;

s_array* initArrayStack(int capacity);
bool isStackFull(const s_array* temp);
void push(s_array* temp, char data);
int pop(s_array* temp);