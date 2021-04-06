#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int factorial(int n) {
	return (n > 1) ? n * factorial(n - 1): n;

}

int factorial2(int n) {
	int m = 1;
	for (int i = n; i > 1; i--)
		m *= i;
	return m;

}