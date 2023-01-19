#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#define FILTER_WIDTH 3
__constant__ float dc_filterX[FILTER_WIDTH * FILTER_WIDTH];
__constant__ float dc_filterY[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) 
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) 
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void writeGrayscalePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");
}

void convertToGrayscaleByHost(uchar3 * inPixels, int width, int height, uint8_t * outPixels)
{
    // gray = 0.299 * red + 0.587 * green + 0.114 * blue  
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int i = r * width + c;
            outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
        }
    }
}

void edgeDetectionByHost(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix)
{

	int filterX[9] = {-1, 0, 1,
					  -2, 0, 2,
					  -1, 0, 1};

	int filterY[9] = {1, 2, 1,
					  0, 0, 0,
					 -1, -2, -1};
	int filterWidth = 3;

	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			float outPixelX = 0;
			float outPixelY = 0;
			for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					float filterValX = filterX[filterR*filterWidth + filterC];
					float filterValY = filterY[filterR*filterWidth + filterC];

					int inPixelsR = outPixelsR - filterWidth/2 + filterR;
					int inPixelsC = outPixelsC - filterWidth/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];

					outPixelX += inPixel * filterValX;
					outPixelY += inPixel * filterValY;
				}
			}
			energyMatrix[outPixelsR*width + outPixelsC] = abs(outPixelX) + abs(outPixelY); 
		}
	}
}

// Parallel code

__global__ void convertToGrayscaleKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
    // Reminder: gray = 0.299 * red + 0.587 * green + 0.114 * blue
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;	
	
    if (r < height && c < width)
    { 
        int i = r * width + c;
        outPixels[i] = 0.299f * inPixels[i].x + 0.587f * inPixels[i].y + 0.114f * inPixels[i].z;
    }	
}

void convertToGrayscaleByDevice(uchar3 * inPixels, int width, int height, uint8_t * outPixels, 
		dim3 blockSize=dim3(1))
{
	uchar3 * d_in;
	uint8_t * d_out;
	CHECK(cudaMalloc(&d_in, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_out, width * height * sizeof(uint8_t)));

	CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	convertToGrayscaleKernel<<<gridSize, blockSize>>>(d_in, width, height, d_out);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));

	CHECK(cudaMemcpy(outPixels, d_out, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}

__global__ void edgeDetectionKernel(uint8_t * inPixels, int width, int height, float * filterX,
        float * filterY, int filterWidth, uint8_t * energyMatrix)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) 
    {
        float outPixelX = 0;
        float outPixelY = 0;
        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                float filterValX = filterX[filterR*filterWidth + filterC];
                float filterValY = filterY[filterR*filterWidth + filterC];

                int inPixelsR = r - filterWidth/2 + filterR;
                int inPixelsC = c - filterWidth/2 + filterC;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR*width + inPixelsC];

                outPixelX += inPixel * filterValX;
                outPixelY += inPixel * filterValY;
                }
        }
        energyMatrix[r*width + c] = abs(outPixelX) + abs(outPixelY); 
    }
}

void edgeDetectionByDevice(uint8_t * inPixels, int width, int height, uint8_t * energyMatrix, dim3 blockSize=dim3(1))
{
    // X axis edge detect
    float filterX[9] = {-1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1};

	// Y axis edge dectect
	float filterY[9] = {1, 2, 1,
					   0, 0, 0,
					  -1, -2, -1};

	int filterWidth = 3;

	// Allocate device memories
	uint8_t * d_in, * d_energyMatrix;
    float * d_filterX, * d_filterY;
	CHECK(cudaMalloc(&d_in, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_energyMatrix, width * height * sizeof(uint8_t)));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Set grid size and call kernel
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    
    // Allocate device memories
    CHECK(cudaMalloc(&d_filterX, filterWidth * filterWidth * sizeof(float)));
    CHECK(cudaMalloc(&d_filterY, filterWidth * filterWidth * sizeof(float)));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_filterX, filterX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filterY, filterY, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

    edgeDetectionKernel<<<gridSize, blockSize>>>(d_in, width, height, d_filterX, d_filterY, 
            filterWidth, d_energyMatrix);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));

	CHECK(cudaMemcpy(energyMatrix, d_energyMatrix, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_filterX));
    CHECK(cudaFree(d_filterY));
    
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_energyMatrix));

}

__global__ void computeMinimumEnergyOnRowKernel(uint8_t * inPixelsRow, int width, int height, 
        int curRow, uint32_t * minimumEnergyRow, uint32_t * backtrack)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < width)
    {
        int minIdx = 0;
        
        if (c == 0)
        {
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;
            
            minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
        else if (c == width - 1)
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid; 
        }
        else 
        {
            int left = (curRow - 1) * width + c - 1;
            int mid = (curRow - 1) * width + c;
            int right = (curRow - 1) * width + c + 1;

            minIdx = left;
            if (minimumEnergyRow[mid] < minimumEnergyRow[minIdx]) minIdx = mid;
            if (minimumEnergyRow[right] < minimumEnergyRow[minIdx]) minIdx = right;
        }
		
        minimumEnergyRow[curRow * width + c] = inPixelsRow[curRow * width + c] + minimumEnergyRow[minIdx];
        backtrack[curRow * width + c] = minIdx;
    }
}

void findSeamPathByDevice(uint8_t * inPixels, int width, int height, uint32_t * seamPath, dim3 blockSize= dim3(1))
{
	uint32_t * minimumEnergy, * backtrack, * tmp;
	backtrack = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	tmp = (uint32_t *)malloc(height * sizeof(uint32_t));
	minimumEnergy = (uint32_t *)malloc(width * height * sizeof(uint32_t));

	// Top row 
	for (int c = 0; c < width; c++)
	{
		minimumEnergy[c] = inPixels[c];
	}

    uint32_t * d_backtrack;
    CHECK(cudaMalloc(&d_backtrack, width * height * sizeof(uint32_t)));

    uint32_t * d_minimumEnergy;
    uint8_t * d_in;

    CHECK(cudaMalloc(&d_minimumEnergy, width * height * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_in, width * height * sizeof(uint8_t)));

    CHECK(cudaMemcpy(d_minimumEnergy, minimumEnergy, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

    for (int r = 1; r < height; r++)
    {
        computeMinimumEnergyOnRowKernel<<<gridSize, blockSize>>>(d_in, width, height,
                    r, d_minimumEnergy, d_backtrack);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    }
   

    CHECK(cudaMemcpy(backtrack, d_backtrack, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_backtrack));

    CHECK(cudaMemcpy(minimumEnergy, d_minimumEnergy, width * height* sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_minimumEnergy));
    CHECK(cudaFree(d_in));

	uint32_t min = minimumEnergy[(height - 1) * width];
	uint32_t minIdx = 0;
	for (int c = 1; c < width; c++) 
	{
		if (minimumEnergy[(height - 1) * width + c] < min) 
		{
			min = minimumEnergy[(height - 1) * width + c];
			minIdx = (height - 1) * width + c;
		}
	}

	seamPath[0] = minIdx;
    int curIdx = minIdx;
	for (int r = 1; r < height; r++)
	{
        seamPath[r] = backtrack[curIdx];
        curIdx = backtrack[curIdx];
	}

	memcpy(tmp, seamPath, height * sizeof(uint32_t));
	int idx = 0;
	for (int i = height - 1; i >= 0; i--)
	{
		seamPath[idx] = tmp[i];
		idx++;
	}

	free(minimumEnergy);
	free(backtrack);
    free(tmp);
}

void seamCarvingByDevice(uchar3 * inPixels, int width, int height, uchar3 * outPixels, 
        int scale_width, dim3 blockSize=dim3(1))
{
    uchar3 * img = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(img, inPixels, (width * height * sizeof(uchar3)));
   

	for (int i = 0; i < width - scale_width; i++)
    {
        int curWidth = width - i;
        uint8_t * grayScaleImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));
        uint8_t * edgeDetectImg = (uint8_t *)malloc(curWidth * height * sizeof(uint8_t));

        convertToGrayscaleByDevice(img, curWidth, height, grayScaleImg, blockSize);
		
        edgeDetectionByDevice(grayScaleImg, curWidth, height, edgeDetectImg, blockSize);
        
        uint32_t * seamPath;
        uchar3 * temp;
        seamPath = (uint32_t *)malloc(height * sizeof(uint32_t));
        memset(seamPath, 0, height * sizeof(uint32_t));

        findSeamPathByDevice(edgeDetectImg, curWidth, height, seamPath, blockSize);
		
		temp = (uchar3 *)malloc((curWidth - 1) * height * sizeof(uchar3));

        int idx = 0;
        for (int r = 0; r < height; r++) 
        {
            for (int c = 0; c < curWidth; c++) 
            {
                int i = r * curWidth + c;
                if (i != seamPath[r])
                {
                    temp[idx] = img[i];
                    idx++;
                }
            }
        }

        img = (uchar3 *)realloc(img, (curWidth - 1) * height * sizeof(uchar3));
        memcpy(img, temp, (curWidth - 1) * height * sizeof(uchar3));
		
		free(grayScaleImg);
		free(edgeDetectImg);
        free(seamPath);
        free(temp);
    }

    memcpy(outPixels, img, scale_width * height * sizeof(uchar3));

    free(img);
}

void seamCarving(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int scale_width, 
        bool useDevice= false, dim3 blockSize= dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	
    seamCarvingByDevice(inPixels, width, height, outPixels, scale_width, blockSize);
	
	timer.Stop();
    float time = timer.Elapsed();
	printf("\nRun time: %f ms\n", time);
}


int main(int argc, char ** argv)
{
	printDeviceInfo();
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nInput image size (width x height): %i x %i\n", width, height);
    float scale_rate = 0.85;

    if (argc >= 4) 
    {
        scale_rate = atof(argv[3]);
    }
    int scale_width = width * scale_rate;
    printf("Output image size (width x height): %i x %i\n", scale_width, height);

    uint8_t * grayScaleImg = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t * edgeDetectImg = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    convertToGrayscaleByHost(inPixels, width, height, grayScaleImg);
		
    edgeDetectionByHost(grayScaleImg, width, height, edgeDetectImg);

    dim3 blockSize(32, 32);
	
	uchar3 * outPixelsByDevice = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixelsByDevice, scale_width, true, blockSize);
	
    char * outFileNameBase = strtok(argv[2], ".");
    writeGrayscalePnm(edgeDetectImg, 1, width, height, concatStr(outFileNameBase, "_edgeDetect.pnm"));
	writePnm(outPixelsByDevice, scale_width, height, concatStr(outFileNameBase, "_device.pnm"));

	free(inPixels);
    free(grayScaleImg);
    free(edgeDetectImg);
	free(outPixelsByDevice);
}