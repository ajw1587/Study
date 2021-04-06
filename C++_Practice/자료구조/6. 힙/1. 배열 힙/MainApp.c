/*
	우선순위 Queue --> 기본은 큐 구조와 동일 
	 - 입력된 순서를 기반으로하는 큐가 아닌
	   저장되어있는 우선순위 데이터가 빠른 쪽이 먼저 출력
*/

#include "myHeap.h"

int main() {
	int arr[15] = { 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 };
	heap* test = createHeap(4);
	for (int i = 0; i < 15; i++) {
		addData(test, arr[i]);
	}

	
	for (int i = 0; i < 15; i++) {
		printf("main>> %d\n", popData(test));
	}

}