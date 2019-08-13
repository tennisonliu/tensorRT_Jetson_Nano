#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "tensorNet.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace nvuffparser;
using namespace nvinfer1;

static Logger gLogger;
static std::vector<void*> buffers;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;

#define MAX_WORKSPACE (1<<30)
#define MAX_BATCHSIZE 1
#define OUTPUT_LEN 1001

size_t getBufferSize(Dims d, nvinfer1::DataType t){
	size_t size = 1;
	for(size_t i=0; i<d.nbDims; i++) size*= d.d[i];
	switch(t){
		case nvinfer1::DataType::kFLOAT: return size*4;
		case nvinfer1::DataType::kHALF: return size*2;
		case nvinfer1::DataType::kINT8: return size*1;
	}
	assert(0);
	return 0;
}

void prepareBuffer(ICudaEngine* engine) {
	if(!engine){
		std::cout<<"Invalid Engine, create engine first" << std::endl;
		exit(0);
	}
	IExecutionContext* context = engine->createExecutionContext();
	
	int nbBindings = engine->getNbBindings();
	
	buffers.clear();
	buffers.reserve(nbBindings);
	
	for (int i = 0; i<nbBindings; i++){
		cudaMallocManaged(&buffers[i], getBufferSize(engine->getBindingDimensions(i), engine->getBindingDataType(i)));
	}
	std::cout << "Successfully created binding buffer" << std::endl;
}

// 1. convert image to the right size
// 2. convert to float
// 3. normalize for inception //
// 4. convert to flat vector, channels first
float * normalize_for_trt(const cv::cuda::GpuMat &img)
{
	cv::Size size(INPUT_W, INPUT_H);
	cv::cuda::GpuMat resizedMat;
	cv::resize(img, resizedMat, size, 0, 0, CV_INTER_LINEAR);
	cv::cuda::GpuMat flatData = resizedMat.reshape(1, 1);
	// cv::cuda::cvtColor(resizedMat, resizedMat, cv::COLOR_BGRA2RGB);

	unsigned volChl = INPUT_H * INPUT_W;

	float * data;
	cudaMalloc((void**)&data, INPUT_C * volChl * sizeof(float));
	data = (float *)flatData.data;


	// we treat the memory as if it's a one-channel, one row image
	// int rowSize = (int)resizedMat.step / (int)resizedMat.elemSize1();

	// Normalize to [-1, 1] for MobileNetV2

	// CUDA kernel to reshape the non-continuous GPU Mat structure and make it channel-first continuous
	//channelFirst(resizedMat.ptr<uint8_t>(), data, volChl, INPUT_C, INPUT_W * INPUT_C, rowSize);
	// cv::cuda::GpuMat flatData = data.reshape(1, 1);

	return data;
}

int main() {

	// create engine
	int size_of_single_input = INPUT_H * INPUT_W * 3 * sizeof(float);
	int size_of_single_output = OUTPUT_LEN * 1 * sizeof(float); 

	auto modelpath = "./model/uff_model.uff";
	auto parser = createUffParser();

	parser->registerInput("input", DimsCHW(3, INPUT_H, INPUT_W), UffInputOrder::kNHWC);
	parser->registerOutput("MobilenetV2/Predictions/Reshape_1");
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder-> createNetwork();
	
	if(!parser->parse(modelpath, *network, nvinfer1::DataType::kFLOAT)){
		std::cout << "Failed to parse UFF model" << modelpath <<std::endl;
		exit(0);
	}
	
	builder->setMaxBatchSize(MAX_BATCHSIZE);
	builder->setMaxWorkspaceSize(MAX_WORKSPACE);
	
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (!engine){
		std::cout << "Unable to create engine" << std::endl;
		exit(0);
	}
	network->destroy();
	builder->destroy();
	parser->destroy();
	std::cout << "Successfully created TensorRT engine from UFF" << modelpath << std::endl;

	//create context and perform inference
	IExecutionContext *context = engine->createExecutionContext();

	// read in image
	cv::cuda::GpuMat input = cv::cuda::GpuMat(imread("./elephant.jpg"));
	// cv::resize(input, input, cv::Size(3, INPUT_H, INPUT_W));
	float * data = normalize_for_trt(input);

	float output[OUTPUT_LEN];

	int inputIndex = engine->getBindingIndex("input");
	int outputIndex = engine->getBindingIndex("MobilenetV2/Predictions/Reshape_1");
	//prepareBuffer(engine);
	cudaMalloc(&buffers[inputIndex], MAX_BATCHSIZE * size_of_single_input);
	cudaMalloc(&buffers[outputIndex], MAX_BATCHSIZE * size_of_single_output);

	// this line needs some attention
	cudaMemcpy(buffers[inputIndex], data, MAX_BATCHSIZE * size_of_single_input, cudaMemcpyHostToDevice);
	context->execute(MAX_BATCHSIZE, &buffers[0]);
	cudaMemcpy(output, buffers[outputIndex], MAX_BATCHSIZE * size_of_single_output, cudaMemcpyDeviceToHost);

	float max_value = -100;
	int max_index = -1;

	for (int j = 0; j < OUTPUT_LEN; j++)
		if (output[j] > max_value){
			max_value = output[j];
			max_index = j;
		}
	std::cout << "Max Value = " << max_value << std::endl;
}


