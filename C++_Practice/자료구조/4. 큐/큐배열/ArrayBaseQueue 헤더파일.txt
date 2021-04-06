#pragma once
#include "../SingleLinkedList/framework.h"

typedef struct _ArrayBasedQueue {
	int* Storage;
	int Capacity;
	int front;// 앞 : 추출위치 인덱스
	int rear; // 뒤 : 입력위치 인덱스
}a_queue;

/*
	Qeueu 구조를 
	배열을 이용해서 직선형태로 구현할 경우 front / rear 가 각자 증가만 하게 구현하기 때문에
	1회용으로 종료되거나 Capacity를 늘려서 재할당한 뒤 사용하게 된다. 
	   * 메모리를 초과할 수 있다.
	   * 잉여 메모리(할당 되어있지만 사용하지 않는 메모리)가 증가 하게 된다.
	배열의 끝 인덱스와 첫 인덱스를 연결하는 형태로 배열을 사용한다.
	<환형 큐 구조> 
	마지막 인덱스 -> 첫 인덱스 --> (index + 1) % Size;
	큐 구조의 데이터 입력과정을 : enqueue();
	                추출과정  : dequeue();
	* 큐의 저장공간이 전부 사용중인 경우 : 
	   (rear + 1)%capacity 연산의 인덱스번호가 front 인덱스번호와 같다.
	* 큐의 저장공간에 데이터가 없는 경우 : 
		front인덱스번호와 rear의 번호가 같다.
*/

a_queue* initArrayQueue(int );

void enQueue(a_queue * , int );
int deQueue(a_queue *);
bool isQEmpty(const a_queue *);
bool isQFull(const a_queue*);
int seekFront(const a_queue*);
int seekRear(const a_queue*);