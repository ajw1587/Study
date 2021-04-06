#include "ArrayBaseQueue.h"

a_queue* initArrayQueue(int capacity) {
	a_queue* temp = (a_queue*)malloc(sizeof(a_queue));
	temp->Capacity = capacity;
	temp->Storage = (int*)calloc(capacity, sizeof(int));
	temp->rear = temp->front = 0;
	return temp;
}
bool isQEmpty(const a_queue* cpQ) {
	if (cpQ->rear == cpQ->front) return true;
	else return false;
}
bool isQFull(const a_queue* cpQ) {
	if ((cpQ->rear + 1) % cpQ->Capacity == cpQ->front) return true;
	else return false;
}

void enQueue(a_queue* pQ, int value) {
	int index = 0;
	if (isQFull(pQ)) return;
	index = pQ->rear;
	index = (index + 1) % pQ->Capacity;
	pQ->Storage[index] = value;
	pQ->rear = index;
	return;
}
int deQueue(a_queue* pQ) {
	int index = 0;
	int data = 0;
	if (isQEmpty(pQ)) return -1;

	index = pQ->front;

	index = (index + 1)%pQ->Capacity;
	data = pQ->Storage[index];
	pQ->front = index;
	return data;
}


int seekFront(const a_queue*cpQ){ return cpQ->Storage[cpQ->front];}
int seekRear(const a_queue* cpQ){ return cpQ->Storage[cpQ->rear];}