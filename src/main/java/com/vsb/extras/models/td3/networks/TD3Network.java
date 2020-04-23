package com.vsb.extras.models.td3.networks;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface TD3Network {
    MultiLayerNetwork getNeuralNetwork();
    INDArray forwardPass(INDArray state);
    Gradient computeGradient(INDArray input, INDArray expectedOutput);
    void applyGradient(Gradient gradient, int batchSize);
}
