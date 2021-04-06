#include "StackArray.h"

s_array* initArrayStack(int capacity)
{
	s_array* temp = (s_array*)malloc(sizeof(s_array));

	temp->Storage = (char*)calloc(capacity, sizeof(char));
	temp->Capacity = capacity;
	temp->Top = -1;
}

bool isStackFull(const s_array* temp)
{
	if (temp->Top+1 == temp->Capacity) return true;
	else return false;
}

void push(s_array* temp, char data)
{
	if (!isStackFull(temp))
	{
		temp->Top += 1;
		temp->Storage[temp->Top] = data;
		return;
	}
	else return;
}
int pop(s_array* temp)
{
	if (temp->Top + 1 == temp->Capacity) return false;
	int data = 0;
	data = temp->Storage[temp->Top];
	temp->Top -= 1;
	return data;
}