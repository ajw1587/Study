#include "mySort.h"

/*
 22  25  13   8  14  19   5  10  20   9
  5 8 9 10 13 14  19 20 22  25
   - Bubble Sort
   - CockTail Sort
   - Selection Sort
   - Insert Sort
   - Radix Sort
   - Merge Sort
   - Quick Sort
   ...
   ...
   ... etc
*/
int QuickFunc(const void* var1, const void* var2) {
    int x = *(int*)var1;
    int y = *(int*)var2;
    if (x - y > 0) return -1;
    else if (x - y < 0) return 1;
    else return 0;
}
int main() {
	int* arr = setArray(sizeof(int), 10);
	arr = SetArrRange(arr, 10, 4, 20);
	
    system("mode con cols=300 lines=100");
	printArray(arr, 10);
    qsort(arr, 10, sizeof(int), QuickFunc);
    //QuickSort(arr, 0, 9);
    
    printf("\n");
    printArray(arr, 10);
    system("pause");

}