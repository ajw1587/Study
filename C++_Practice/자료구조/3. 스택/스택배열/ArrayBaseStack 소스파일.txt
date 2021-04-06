#include "ArrayBaseStack.h"

a_stack* initArrayStack(int _capacity) {
	a_stack* temp = (a_stack*)malloc(sizeof(a_stack));
	temp->storage = (int*)calloc(_capacity, sizeof(int));

	temp->Capacity = _capacity;
	temp->Top = -1;
	return temp;
}

bool isStackFull(const a_stack* _cpStack) {
	if (_cpStack->Top + 1 == _cpStack->Capacity) {
		printf("\a STACK OVERFLOW\n");
		return true;
	}
	else return false;
}

void push(a_stack* _pStack, int _value) {
	if (!isStackFull(_pStack)) {
		_pStack->Top += 1;
		_pStack->storage[_pStack->Top] = _value;
	}
	return;
}
bool isStackEmpty(const a_stack* _cpStack) {
	if (_cpStack->Top == -1) {
		printf("\aSTACK UNDERFLOW\n");
		return true;
	}
	return false;
}
int pop(a_stack* _pStack) {
	int data = 0;
	if (!isStackEmpty(_pStack)) {
		data = _pStack->storage[_pStack->Top];
		_pStack->Top -= 1;
		return data;
	}
	else
		return -1;
}

int Seek(const a_stack* _cpStack) {
	if (!isStackEmpty(_cpStack))
		return _cpStack->storage[_cpStack->Top];
	else return -1;
}