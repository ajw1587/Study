#pragma once
#include "framework.h"
#include "ListBaseStack.h"
#include "ListBaseQueue.h"
typedef struct _AdjacencyMatrix {
	int** matrix;
}AMGraph;

AMGraph * initMatrix(int _GraphSize);

void connectVertex(AMGraph* _block,int StartVertexIndex,int ArrivalVertexIndex);

void ShowGraphInfo(const AMGraph* _block, const char* VertexGroup,int _Size);

int getIndex(char* VertexGroup, int Size, char VertexName);
void DFSGraph(const AMGraph* _block, const char* VertexGroup, int _Size);
void BFSGraph(const AMGraph* _block, const char* VertexGroup, int _Size);