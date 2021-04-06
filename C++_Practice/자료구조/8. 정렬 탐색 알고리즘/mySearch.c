#include "mySearch.h"

int binarySearch(const int* block, int front, int rear, int value) {
	//데이터가 정렬 되어있다는 가정 하에 사용되는 검색 알고리즘 
	if (front > rear || rear < 0) return -1;
	int mid = (front + rear) >> 1;
	int cmp = IntegerCmp(block[mid], value);
	if (cmp == 1) binarySearch(block, mid + 1, rear, value);
	else if (cmp == -1) binarySearch(block, front, mid - 1, value);
	else return mid;

}

int IntegerCmp(const int* lparam, const int* rparam) {
	if (*lparam > * rparam) return 1;
	else if (*lparam < *rparam) return -1;
	else return 0;
}
int binarySearch2(const int* block, int front, int rear, int value) {
	bsearch(&value, block, rear - front + 1, sizeof(int), IntegerCmp);
}