#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

long long c_conv(int in_channel, int o_channel, int kernel_size, int stride, uint8_t *input_image);

void main()
{
	int height = 720;
	int width = 1280;
	int channel = 3;
	int o_channel = 3;
	int kernel_size = 3;
	int stride = 1;
	int i, j, k;
	int seed = time(NULL);
	srand(seed);
// init image: rgb
	uint8_t *imgr = (uint8_t **)malloc(sizeof(uint8_t *) * height * width);
	uint8_t *imgg = (uint8_t **)malloc(sizeof(uint8_t *) * height * width);
	uint8_t *imgb = (uint8_t **)malloc(sizeof(uint8_t *) * height * width);
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			*(imgr + i * width + j) = rand() % 255;
			*(imgg + i * width + j) = rand() % 255;
			*(imgb + i * width + j) = rand() % 255;
		}
	}
	for (i = 0; i < 11; i++) {
		clock_t begin, end;
		begin = clock();
		long long c1 = c_conv(3, pow(2,i), 3, 1, imgr);
		long long c2 = c_conv(3, pow(2, i), 3, 1, imgg);
		long long c3 = c_conv(3, pow(2, i), 3, 1, imgb);
		end = clock();
		//printf("%f", (float)CLOCKS_PER_SEC);
		printf("%d 's time is : %f \n", i, (float) ((double)(end-begin)/(double)CLOCKS_PER_SEC)); //(i, clock());
		printf("%lli operations \n", (long long)(c1+c2+c3+2*718*1278)); //extra adding times for 3 rgb channels
	}

	getchar();
	return;
}

long long c_conv(int in_channel, int o_channel, int kernel_size, int stride, uint8_t *input_image)
{
	long long count = 0;
	int K1[3][3] = { { -1, -1, -1 },{ 0, 0, 0 },{ 1, 1, 1 } };
	int d1 = 720;
	int d2 = 1280;
	int od1 = (d1 - kernel_size) / stride + 1;
	int od2 = (d2 - kernel_size) / stride + 1;
	int i, j;
	int p, q;
	int cn;
	int *output = (int **)malloc(sizeof(int *) * od1 * od2);
	for (cn = 0; cn < o_channel; cn++) {
		for (i = 0; i < kernel_size; i++) {
			for (j = 0; j < kernel_size; j++) {
				K1[i][j] = rand() % 3 - 1;
			}
		}
		for (i = 0; i < od1; i++) {
			for (j = 0; j < od2; j++) {
				for (p = 0; p < kernel_size; p++) {
					for (q = 0; q < kernel_size; q++) {
						*(output + i * od2 + j) += *(input_image + (i*stride + p)*d2 + j*stride + q) * K1[p][q];
						count = count + 2; //a x, a +
					}
				}
			}
		}
	}
	return count;
}