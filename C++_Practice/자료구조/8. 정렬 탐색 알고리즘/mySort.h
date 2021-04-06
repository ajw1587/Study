#pragma once
#include "myArray.h"
#define SWAP(x,y) int temp = x; x = y; y = temp

void BubbleSort(int * block,int Size);
void SelectSort(int* block, int Size);
void insertSort(int* block, int Size);
void QuickSort(int* block, int front, int rear);
void RadixSort(int* block, int Size, int Radix);