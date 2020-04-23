package com.vsb.extras.models.td3;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Utils {
    public static INDArray minQ(INDArray firstArray, INDArray secondArray) {
        return Nd4j.create(new double[]{1},new int[]{1,1});    }
}
