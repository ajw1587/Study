#pragma once

#include "../../LinkedList_0530/Practice_0530/framework.h"
#include <stdbool.h>

typedef struct Stackarr
{
	char* Storage;
	int Capacity;
	int front;
	int rear;
}a_queue;

a_queue* initArrayQueue(int capacity);
bool isQEmpty(a_queue* temp);
bool isQFull(a_queue* temp);
void enQueue(a_queue* temp, char data);
int deQueue(a_queue* temp);