#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../StackAndQueue/StackAndQueue/StackAndQueue/QueueArray.h"

void* setArray(int Size, int Count);
int* SetArrRange(int* _block, int _Size, int _Min, int _Max);
void printArray(const int* _block, int _Count);