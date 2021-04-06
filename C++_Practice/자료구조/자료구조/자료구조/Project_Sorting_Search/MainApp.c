#include "myArray.h"

int main(void)
{
	int* arr = setArray(sizeof(int), 10);
	arr = SetArrTange(arr, 10, 4, 26);

	printArray(arr, 10);
}