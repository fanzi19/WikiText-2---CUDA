#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

#define MAX_SEQUENCE_LENGTH 1000
#define NUM_CHUNKS 10
#define NUM_EPOCHS 10
#define BATCH_SIZE 32
#define HIDDEN_SIZE 512

void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCudaErrors(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printTensorDescriptor(const cudnnTensorDescriptor_t& desc, const char* name) {
    int dims[4], strides[4];
    cudnnDataType_t dataType;
    int nbDims;
    
    checkCUDNN(cudnnGetTensorNdDescriptor(desc, 4, &dataType, &nbDims, dims, strides));
    
    printf("%s: dims[%d, %d, %d, %d], strides[%d, %d, %d, %d], dataType %d\n",
           name, dims[0], dims[1], dims[2], dims[3],
           strides[0], strides[1], strides[2], strides[3],
           dataType);
}

void printFilterDescriptor(const cudnnFilterDescriptor_t& desc, const char* name) {
    int dims[4];
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int nbDims;
    
    checkCUDNN(cudnnGetFilterNdDescriptor(desc, 4, &dataType, &format, &nbDims, dims));
    
    printf("%s: dims[%d, %d, %d, %d], dataType %d, format %d\n",
           name, dims[0], dims[1], dims[2], dims[3], dataType, format);
}

std::vector<std::vector<std::string>> readWordsFromCSVInChunks(const std::string& filename, int chunkSize, int numChunks) {
    std::vector<std::vector<std::string>> chunks;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return chunks;
    }
    
    std::vector<std::string> currentChunk;
    int chunkCount = 0;
    
    while (std::getline(file, line) && chunkCount < numChunks) {
        std::istringstream iss(line);
        std::string word;
        
        while (std::getline(iss, word, ',')) {
            currentChunk.push_back(word);
            
            if (currentChunk.size() >= chunkSize) {
                chunks.push_back(currentChunk);
                currentChunk.clear();
                chunkCount++;
                
                if (chunkCount >= numChunks) {
                    break;
                }
            }
        }
        
        if (chunkCount >= numChunks) {
            break;
        }
    }
    
    if (!currentChunk.empty()) {
        chunks.push_back(currentChunk);
    }
    
    return chunks;
}

std::vector<int> encodeWords(const std::vector<std::string>& words, std::unordered_map<std::string, int>& wordToInt, int& nextId) {
    std::vector<int> encoded;
    for (const auto& word : words) {
        if (wordToInt.find(word) == wordToInt.end()) {
            wordToInt[word] = nextId++;
        }
        encoded.push_back(wordToInt[word]);
    }
    return encoded;
}

__global__ void updateWeights(float* weights, float* gradients, int N, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        weights[idx] -= learningRate * gradients[idx];
    }
}

__global__ void sumColumns(const float* matrix, float* result, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < rows) {
        float sum = 0;
        for (int i = 0; i < cols; ++i) {
            sum += matrix[i * rows + col];
        }
        result[col] = sum;
    }
}

__global__ void softmaxCrossEntropyLossWithLabelSmoothingKernel(float* logits, int* targets, float* loss, float* dlogits, int batchSize, int vocabSize, float labelSmoothing) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        float maxLogit = -INFINITY;
        for (int i = 0; i < vocabSize; ++i) {
            maxLogit = max(maxLogit, logits[idx * vocabSize + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < vocabSize; ++i) {
            sum += expf(logits[idx * vocabSize + i] - maxLogit);
        }

        float logSum = logf(sum) + maxLogit;
        int targetIdx = targets[idx];

        float smoothProb = labelSmoothing / vocabSize;
        float targetProb = 1.0f - labelSmoothing + smoothProb;

        float sampleLoss = 0.0f;
        for (int i = 0; i < vocabSize; ++i) {
            float prob = (i == targetIdx) ? targetProb : smoothProb;
            sampleLoss -= prob * (logits[idx * vocabSize + i] - logSum);
            dlogits[idx * vocabSize + i] = expf(logits[idx * vocabSize + i] - logSum) - prob;
        }

        atomicAdd(loss, sampleLoss);
    }
}

__global__ void clipGradientsKernel(float* gradients, int size, float clipValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] = max(min(gradients[idx], clipValue), -clipValue);
    }
}

__global__ void updateWeightsAdamKernel(float* weights, float* m, float* v, int size, float learningRate, float beta1, float beta2, float epsilon, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = weights[idx];
        m[idx] = beta1 * m[idx] + (1 - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1 - beta2) * grad * grad;
        
        float m_hat = m[idx] / (1 - pow(beta1, step + 1));
        float v_hat = v[idx] / (1 - pow(beta2, step + 1));
        
        weights[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

void checkCuBLAS(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: ";
        switch(status) {
            case CUBLAS_STATUS_NOT_INITIALIZED: std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED: std::cerr << "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE: std::cerr << "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH: std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR: std::cerr << "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR: std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            case CUBLAS_STATUS_NOT_SUPPORTED: std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED"; break;
            case CUBLAS_STATUS_LICENSE_ERROR: std::cerr << "CUBLAS_STATUS_LICENSE_ERROR"; break;
            default: std::cerr << "unknown error";
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

class LanguageModel {
private:
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    cudnnRNNDescriptor_t rnnDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnDropoutDescriptor_t dropoutDesc;
    
    float *weights;
    size_t weightSize;
    
    float *outputWeights;
    float *outputBias;
    
    int vocabSize;
    int inputSize;
    int hiddenSize;
    int numLayers;
    float learningRate;
    float clipValue;
    float dropoutProb;

    float initialLearningRate;
    float beta1, beta2, epsilon;
    float *m, *v; // Adam optimizer parameters
    int step;
    
public:
    LanguageModel(int vocabSize, int hiddenSize, int numLayers, float lr = 0.001, float clip = 1.0, float dropout = 0.1) 
        : vocabSize(vocabSize), inputSize(vocabSize), hiddenSize(hiddenSize), numLayers(numLayers),
          initialLearningRate(lr), clipValue(clip), dropoutProb(dropout), beta1(0.9), beta2(0.999), epsilon(1e-8), step(0) {
        
        std::cout << "Initializing LanguageModel with:" << std::endl;
        std::cout << "  vocabSize: " << vocabSize << std::endl;
        std::cout << "  hiddenSize: " << hiddenSize << std::endl;
        std::cout << "  numLayers: " << numLayers << std::endl;

        checkCUDNN(cudnnCreate(&cudnnHandle));
        checkCuBLAS(cublasCreate(&cublasHandle));
        
        checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                            BATCH_SIZE, inputSize, 1, 1));
        
        void* states = nullptr;
        size_t stateSize = 0;
        checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

        if (stateSize > 0) {
            checkCudaErrors(cudaMalloc(&states, stateSize));
        }

        checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropoutProb, states, stateSize, 0));

        checkCUDNN(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                          rnnDesc,
                                          hiddenSize,
                                          numLayers,
                                          dropoutDesc,
                                          CUDNN_LINEAR_INPUT,
                                          CUDNN_UNIDIRECTIONAL,
                                          CUDNN_LSTM,
                                          CUDNN_RNN_ALGO_STANDARD,
                                          CUDNN_DATA_FLOAT));
        
        checkCUDNN(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc, &weightSize, CUDNN_DATA_FLOAT));

        checkCudaErrors(cudaMalloc(&weights, weightSize));
        checkCudaErrors(cudaMalloc(&outputWeights, hiddenSize * vocabSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&outputBias, vocabSize * sizeof(float)));

        initializeWeights();

        checkCudaErrors(cudaMalloc(&m, weightSize));
        checkCudaErrors(cudaMalloc(&v, weightSize));
        checkCudaErrors(cudaMemset(m, 0, weightSize));
        checkCudaErrors(cudaMemset(v, 0, weightSize));
    }
    
    ~LanguageModel() {
        cudnnDestroy(cudnnHandle);
        cublasDestroy(cublasHandle);
        cudnnDestroyRNNDescriptor(rnnDesc);
        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        cudaFree(weights);
        cudaFree(outputWeights);
        cudaFree(outputBias);
        cudaFree(m);
        cudaFree(v);
    }
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        
        std::vector<float> hostWeights(weightSize / sizeof(float));
        float scale = std::sqrt(6.0f / (inputSize + hiddenSize));
        for (auto& w : hostWeights) {
            w = d(gen) * scale;
        }
        checkCudaErrors(cudaMemcpy(weights, hostWeights.data(), weightSize, cudaMemcpyHostToDevice));

        std::vector<float> hostOutputWeights(hiddenSize * vocabSize);
        scale = std::sqrt(6.0f / (hiddenSize + vocabSize));
        for (auto& w : hostOutputWeights) {
            w = d(gen) * scale;
        }
        checkCudaErrors(cudaMemcpy(outputWeights, hostOutputWeights.data(), hiddenSize * vocabSize * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<float> hostOutputBias(vocabSize, 0.0f);
        checkCudaErrors(cudaMemcpy(outputBias, hostOutputBias.data(), vocabSize * sizeof(float), cudaMemcpyHostToDevice));
    }

    float getLearningRate(int step) {
        float warmup = 1000.0f; // warmup steps
        float factor = std::min(1.0f, (static_cast<float>(step) + 1) / warmup);
        float lr = factor * initialLearningRate / std::sqrt(std::max(static_cast<float>(step) + 1, warmup));
        return std::max(lr, 1e-5f); // Ensure minimum learning rate of 1e-5
    }
        
    float train(const std::vector<int>& encodedWords, int numEpochs) {
        const int seqLength = 128;
        const int batchSize = 32;
        const float clipValue = 1.0f;

        printf("Starting training function\n");

        // Create tensor descriptors
        std::vector<cudnnTensorDescriptor_t> xDesc(seqLength), yDesc(seqLength);
        for (int i = 0; i < seqLength; i++) {
            checkCUDNN(cudnnCreateTensorDescriptor(&xDesc[i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&yDesc[i]));
            checkCUDNN(cudnnSetTensor4dDescriptor(xDesc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                                  batchSize, inputSize, 1, 1));
            checkCUDNN(cudnnSetTensor4dDescriptor(yDesc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                                  batchSize, hiddenSize, 1, 1));
        }

        printf("Tensor descriptors created\n");

        cudnnTensorDescriptor_t hxDesc, cxDesc, hyDesc, cyDesc;
        checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(hxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                              numLayers, batchSize, hiddenSize, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(cxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                              numLayers, batchSize, hiddenSize, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(hyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                              numLayers, batchSize, hiddenSize, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(cyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                              numLayers, batchSize, hiddenSize, 1));

        printf("Additional tensor descriptors created\n");

        // Allocate memory
        float *x, *y, *hx, *cx, *hy, *cy, *dx, *dy, *dhx, *dcx, *dhy, *dcy, *logits;
        float *doutputWeights, *doutputBias;
        checkCudaErrors(cudaMalloc(&x, seqLength * batchSize * inputSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&y, seqLength * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&hx, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&cx, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&hy, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&cy, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dx, seqLength * batchSize * inputSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dy, seqLength * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dhx, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dcx, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dhy, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&dcy, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&logits, seqLength * batchSize * vocabSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&doutputWeights, hiddenSize * vocabSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&doutputBias, vocabSize * sizeof(float)));

        printf("Memory allocated\n");

        // Initialize hidden and cell states to zero
        checkCudaErrors(cudaMemset(hx, 0, numLayers * batchSize * hiddenSize * sizeof(float)));
        checkCudaErrors(cudaMemset(cx, 0, numLayers * batchSize * hiddenSize * sizeof(float)));

        printf("Hidden and cell states initialized\n");

        // Create filter descriptor
        cudnnFilterDescriptor_t wDesc;
        checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
        int filterDims[3] = {(int)(weightSize / sizeof(float)), 1, 1};
        checkCUDNN(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, filterDims));

        printf("Filter descriptor created\n");

        // Workspace and reserve space
        size_t workspaceSize, reserveSpaceSize;
        checkCUDNN(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc.data(), &workspaceSize));
        checkCUDNN(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc.data(), &reserveSpaceSize));

        void *workspace, *reserveSpace;
        checkCudaErrors(cudaMalloc(&workspace, workspaceSize));
        checkCudaErrors(cudaMalloc(&reserveSpace, reserveSpaceSize));

        printf("Workspace and reserve space allocated\n");

        // Training loop
        //for (int epoch = 0; epoch < numEpochs; epoch++) {
        float learningRate = getLearningRate(step);
        printf("Starting training with learning rate %f, Step: %d\n", learningRate, step);

        float totalLoss = 0.0f;
        int totalWords = 0;
        int batchCount = 0;

        for (int batchStart = 0; batchStart < encodedWords.size() - seqLength * batchSize; batchStart += seqLength * batchSize) {
            // Prepare input data
            std::vector<float> hostX(seqLength * batchSize * inputSize, 0.0f);
            std::vector<int> targets(seqLength * batchSize);
            for (int i = 0; i < seqLength; ++i) {
                for (int b = 0; b < batchSize; ++b) {
                    int wordIndex = batchStart + i * batchSize + b;
                    if (wordIndex < encodedWords.size() - 1) {
                        hostX[i * batchSize * inputSize + b * inputSize + encodedWords[wordIndex]] = 1.0f;
                        targets[i * batchSize + b] = encodedWords[wordIndex + 1];
                    }
                }
            }
            checkCudaErrors(cudaMemcpy(x, hostX.data(), hostX.size() * sizeof(float), cudaMemcpyHostToDevice));

            printf("Input data prepared for batch %d\n", batchCount + 1);

            // Forward pass
            checkCUDNN(cudnnRNNForwardTraining(
                cudnnHandle, rnnDesc, seqLength,
                xDesc.data(), x, 
                hxDesc, hx,
                cxDesc, cx,
                wDesc, weights,
                yDesc.data(), y,
                hyDesc, hy,
                cyDesc, cy,
                workspace, workspaceSize,
                reserveSpace, reserveSpaceSize
            ));

            printf("Forward pass completed for batch %d\n", batchCount + 1);

            // Compute loss and gradients
            float batchLoss = 0.0f;
            calculateLossAndGradients(logits, targets, dy, doutputWeights, doutputBias, &batchLoss, seqLength);

            // Backward pass
            checkCUDNN(cudnnRNNBackwardData(
                cudnnHandle, rnnDesc, seqLength,
                yDesc.data(), y, yDesc.data(), dy,
                hyDesc, dhy, cyDesc, dcy,
                wDesc, weights,
                hxDesc, hx, cxDesc, cx,
                xDesc.data(), dx, hxDesc, dhx, cxDesc, dcx,
                workspace, workspaceSize,
                reserveSpace, reserveSpaceSize
            ));

            checkCUDNN(cudnnRNNBackwardWeights(
                cudnnHandle, rnnDesc, seqLength,
                xDesc.data(), x,
                hxDesc, hx,
                yDesc.data(), y,
                workspace, workspaceSize,
                wDesc, weights,  // Use 'weights' instead of 'dweights'
                reserveSpace, reserveSpaceSize
            ));

            printf("Backward pass completed for batch %d\n", batchCount + 1);

            // Clip gradients
            clipGradients(weights, weightSize / sizeof(float), clipValue);
            clipGradients(doutputWeights, hiddenSize * vocabSize, clipValue);
            clipGradients(doutputBias, vocabSize, clipValue);

            // Update weights using Adam optimizer
            updateWeightsAdam(weights, weightSize / sizeof(float), learningRate);
            updateWeightsAdam(outputWeights, hiddenSize * vocabSize, learningRate);
            updateWeightsAdam(outputBias, vocabSize, learningRate);

            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            printf("Weights updated for batch %d\n", batchCount + 1);

            totalLoss += batchLoss;
            totalWords += seqLength * batchSize;
            batchCount++;
            step++;

            if (batchCount % 10 == 0) {
                float currentPerplexity = std::exp(totalLoss / totalWords);
                printf("Batch %d, Perplexity: %f\n", batchCount, currentPerplexity);
            }
        }

        float epochPerplexity = std::exp(totalLoss / totalWords);
        printf("Completed training, Perplexity: %f, Processed batches: %d\n", 
               epochPerplexity, batchCount);

        printf("Training loop completed\n");

        // Clean up
        for (auto& desc : xDesc) cudnnDestroyTensorDescriptor(desc);
        for (auto& desc : yDesc) cudnnDestroyTensorDescriptor(desc);
        cudnnDestroyTensorDescriptor(hxDesc);
        cudnnDestroyTensorDescriptor(cxDesc);
        cudnnDestroyTensorDescriptor(hyDesc);
        cudnnDestroyTensorDescriptor(cyDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudaFree(x);
        cudaFree(y);
        cudaFree(hx);
        cudaFree(cx);
        cudaFree(hy);
        cudaFree(cy);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dhx);
        cudaFree(dcx);
        cudaFree(dhy);
        cudaFree(dcy);
        cudaFree(logits);
        cudaFree(doutputWeights);
        cudaFree(doutputBias);
        cudaFree(workspace);
        cudaFree(reserveSpace);

        return totalLoss / totalWords;  // Return average loss
    }

private:
    void clipGradients(float* gradients, int size, float clipValue) {
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        clipGradientsKernel<<<gridDim, blockDim>>>(gradients, size, clipValue);
    }

    void calculateLossAndGradients(float* logits, const std::vector<int>& targets, float* dy, float* doutputWeights, float* doutputBias, float* loss, int seqLength) {
        int batchSize = BATCH_SIZE * seqLength;
        int* d_targets;
        float* d_loss;
        float* dlogits;

        checkCudaErrors(cudaMalloc(&d_targets, targets.size() * sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_loss, sizeof(float)));
        checkCudaErrors(cudaMalloc(&dlogits, batchSize * vocabSize * sizeof(float)));

        checkCudaErrors(cudaMemcpy(d_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(d_loss, 0, sizeof(float)));

        float labelSmoothing = 0.1;
        softmaxCrossEntropyLossWithLabelSmoothingKernel<<<(batchSize + 255) / 256, 256>>>(logits, d_targets, d_loss, dlogits, batchSize, vocabSize, labelSmoothing);

        float h_loss;
        checkCudaErrors(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        *loss = h_loss;

        // Compute gradients for output weights and bias
        float alpha = 1.0f / batchSize;
        float beta = 0.0f;
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    vocabSize, hiddenSize, batchSize,
                    &alpha, dlogits, vocabSize,
                    dy, hiddenSize,
                    &beta, doutputWeights, vocabSize);

        sumColumns<<<(vocabSize + 255) / 256, 256>>>(dlogits, doutputBias, vocabSize, batchSize);
        cublasSscal(cublasHandle, vocabSize, &alpha, doutputBias, 1);

        // Compute gradients for RNN
        cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    hiddenSize, batchSize, vocabSize,
                    &alpha, outputWeights, vocabSize,
                    dlogits, vocabSize,
                    &beta, dy, hiddenSize);

        // Clip gradients
        clipGradients(doutputWeights, hiddenSize * vocabSize, clipValue);
        clipGradients(doutputBias, vocabSize, clipValue);
        clipGradients(dy, batchSize * hiddenSize, clipValue);

        // Free temporary memory
        checkCudaErrors(cudaFree(d_targets));
        checkCudaErrors(cudaFree(d_loss));
        checkCudaErrors(cudaFree(dlogits));
    }

    void updateWeightsAdam(float* weights, int size, float learningRate) {
        dim3 blockDim(256);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
        updateWeightsAdamKernel<<<gridDim, blockDim>>>(weights, m, v, size, learningRate, beta1, beta2, epsilon, step);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [num_epochs] [hidden_size] [num_layers]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int numEpochs = (argc > 2) ? std::stoi(argv[2]) : NUM_EPOCHS;
    int hiddenSize = (argc > 3) ? std::stoi(argv[3]) : HIDDEN_SIZE;
    int numLayers = (argc > 4) ? std::stoi(argv[4]) : 4;

    std::vector<std::vector<std::string>> wordChunks;
    try {
        wordChunks = readWordsFromCSVInChunks(filename, MAX_SEQUENCE_LENGTH, NUM_CHUNKS);
    } catch (const std::exception& e) {
        std::cerr << "Error reading file: " << e.what() << std::endl;
        return 1;
    }

    if (wordChunks.empty() || wordChunks[0].empty()) {
        std::cerr << "No words read from file." << std::endl;
        return 1;
    }

    std::unordered_map<std::string, int> wordToInt;
    int nextId = 1;  // Start from 1, reserve 0 for unknown words
    std::vector<std::vector<int>> encodedChunks;

    for (const auto& chunk : wordChunks) {
        encodedChunks.push_back(encodeWords(chunk, wordToInt, nextId));
    }

    int vocabSize = nextId;

    // Split data into training and validation sets
    std::vector<int> trainingData;
    std::vector<int> validationData;
    
    for (const auto& chunk : encodedChunks) {
        size_t splitPoint = static_cast<size_t>(chunk.size() * 0.8);
        trainingData.insert(trainingData.end(), chunk.begin(), chunk.begin() + splitPoint);
        validationData.insert(validationData.end(), chunk.begin() + splitPoint, chunk.end());
    }

    float initialLearningRate = 0.001f;
    float dropoutRate = 0.1f;

    LanguageModel model(vocabSize, hiddenSize, numLayers, initialLearningRate, 1.0, dropoutRate);

    // Training loop with improved progress reporting
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs << std::endl;
        
        float epochLoss = model.train(trainingData, BATCH_SIZE);
        float perplexity = std::exp(epochLoss);
        
        std::cout << "Epoch " << (epoch + 1) << " completed. "
                  << "Loss: " << epochLoss << ", Perplexity: " << perplexity << std::endl;
    }

    return 0;
}
