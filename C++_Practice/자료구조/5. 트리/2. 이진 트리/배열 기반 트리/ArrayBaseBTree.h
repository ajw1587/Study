#pragma once
/*배열 기반 이진트리 ADT*/
#include "framework.h"
#define TREE_SIZE	16	
/* 배열 기반 이진트리의 경우
[0] 인덱스가 Root 일 때  -> Left[1]   | Right[2]  --> x*2 + 1    |  (x+1)*2    
[1] 인덱스가 SubTreeRoot 일 때  -> Left[3]   | Right[4]  --> x*2 + 1    |  (x+1)*2
[1] 인덱스가 Root 일 때  -> Left[2]   | Right[3]  --> x*2    |  x*2+1
인덱스 연산을 간단히 하기위해 1번인덱스를 Root로 한다.  [0]은 사용 안함
Tree 구조 전체의 크기 --> 깊이가 0인 노드의 최대 개수 : 1	 트리 전체노드 개수 : 1	2^1 - 1 +1
                     --> 깊이가 1인 노드의 최대 개수 : 2   트리 전체노드 개수 : 3	2^2 - 1	+1
					 --> 깊이가 2인 노드의 최대 개수 : 4   트리 전체노드 개수 : 7	2^3 - 1	+1
					 --> 깊이가 3인 노드의 최대 개수 : 8   트리 전체노드 개수 : 15	2^4 - 1	+1
*/
typedef int Node;
typedef struct _BinaryTree {
	Node* storage;  // 1 ~ 15인덱스 까지 15개의 데이터를 저장
	int cur; // 마지막 입력된 인덱스
}btree;

btree* initBTree(int Size);
void addData(btree * block, Node value);
void erase_Tree(btree* block);
void FreeTree(btree* block);

int getLeftChild(int Pos);
int getRightChild(int Pos);

void PreOrder	(const btree* target, int index); /* 전위 순회   :  Root -> Left -> Right  */
void PostOrder	(const btree* target, int index); /* 후위 순회  :  Left -> Right-> Root   */
void InOrder	(const btree* target, int index); /* 중위 순회    :  Left -> Root -> Right  */