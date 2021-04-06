#include "QueueArray.h"

a_queue* initArrayQueue(int capacity)
{
	a_queue* temp = (a_queue*)malloc(sizeof(a_queue));

	temp->Capacity = capacity;
	temp->front = 0;
	temp->rear = 0;
	temp->Storage = (char*)calloc(capacity, sizeof(char));

	return temp;
}

bool isQEmpty(a_queue* temp)
{
	if (temp->rear == temp->front) return true;
	else return false;
}

bool isQFull(a_queue* temp)
{
	if ((temp->rear + 1) % temp->Capacity == temp->front) return true;
	else return false;
}

void enQueue(a_queue* temp, char data)
{
	if (isQFull(temp)) return;
	int index = temp->rear;

	index = (index + 1) % temp->Capacity;
	temp->Storage[index] = data;
	temp->rear = index;
	return;
}

char deQueue(a_queue* temp)
{
	int data = 0;
	char index = 0;

	index = (temp->front + 1) % temp->Capacity;
	data = temp->Storage[index];

	temp->front = index;

	return data;
}