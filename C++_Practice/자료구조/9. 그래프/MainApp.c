#include "framework.h"
#include "AdjacencyMatrix.h"
/*
정점의 개수에 따라서 인접 행렬의 크기를 할당하고 
PPT 의 그래프 구조처럼 연결 하는 ADT 제작
*/

int main() {
	char Vertex[7]= { 'A', 'B', 'C', 'D', 'E', 'F', 'G' };
	AMGraph* graph = initMatrix(7);
	
	connectVertex(graph, getIndex(Vertex, 7, 'A') , getIndex(Vertex, 7, 'B'));
	connectVertex(graph, getIndex(Vertex, 7, 'A'), getIndex(Vertex, 7, 'D'));
	connectVertex(graph, getIndex(Vertex, 7, 'B'), getIndex(Vertex, 7, 'C'));
	connectVertex(graph, getIndex(Vertex, 7, 'C'), getIndex(Vertex, 7, 'D'));
	connectVertex(graph, getIndex(Vertex, 7, 'D'), getIndex(Vertex, 7, 'E'));
	connectVertex(graph, getIndex(Vertex, 7, 'D'), getIndex(Vertex, 7, 'F'));

	connectVertex(graph, getIndex(Vertex, 7, 'E'), getIndex(Vertex, 7, 'F'));
	connectVertex(graph, getIndex(Vertex, 7, 'F'), getIndex(Vertex, 7, 'G'));

	
	ShowGraphInfo(graph, Vertex, 7);

	printf("\n");
	printf("깊이 우선 탐색\n");
	DFSGraph(graph, Vertex, 7);

	return 0;
}