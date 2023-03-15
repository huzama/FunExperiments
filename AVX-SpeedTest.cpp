#include <immintrin.h>
#include<iostream>

int main()
{
	float lola[] = { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 };
	float BOBA[] = { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0 };

	/* Initialize the two argument vectors */
	__m256 evens = *(__m256*)&lola;
	__m256 odds = *(__m256*)&BOBA;

	/* Compute the difference between the two vectors */
	__m256 result = _mm256_sub_ps(evens, odds);

	/* Display the elements of the result vector */

	float* e = (float*)&evens;
	float* o = (float*)&odds;
	float* a = (float*)&result;

	for (int i = 0; i < 8; i++)
	{
		std::cout << e[i] << " - " << o[i] << " = " << a[i] << std::endl;
	}

	return 69;
}