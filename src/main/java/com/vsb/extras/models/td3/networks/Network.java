package com.vsb.extras.models.td3.networks;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

public class Network implements NeuralNet, TD3Network {
    final protected MultiLayerNetwork mln;

    public Network (MultiLayerNetwork mln) {
        this.mln = mln;

    }

    @Override
    public NeuralNetwork[] getNeuralNetworks() {
      return new NeuralNetwork[] { mln };
    }

    @Override
    public boolean isRecurrent() {
        return false;
    }

    @Override
    public void reset() {

    }

    public INDArray output(INDArray batch) {
        return mln.output(batch);
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[] {output(batch)};
    }

    @Override
    public NeuralNet clone() {
        return new Network(mln);
    }


    @Override
    public void copy(NeuralNet from) {
    }

    public Gradient[] gradient(INDArray input, INDArray[] expectedOutput) {
        return gradient(input, expectedOutput[0]);
    }

    public Gradient[] gradient(INDArray input, INDArray expectedOutput) {
        mln.setInput(input);
        mln.setLabels(expectedOutput);
        mln.computeGradientAndScore();
        Collection<TrainingListener> iterationListeners = mln.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener l : iterationListeners) {
                l.onGradientCalculation(mln);
            }
        }
        //System.out.println("SCORE: " + mln.score());
        return new Gradient[] {mln.gradient()};
    }


    @Override
    public void fit(INDArray input, INDArray[] expectedOutput) {
        mln.fit(input, expectedOutput[0]);
    }

    public void fit(INDArray input, INDArray labels) {
        mln.fit(input, labels);
    }

    @Override
    public void applyGradient(Gradient[] gradients, int batchSize) {
        MultiLayerConfiguration mlnConf = mln.getLayerWiseConfigurations();
        int iterationCount = mlnConf.getIterationCount();
        int epochCount = mlnConf.getEpochCount();
        mln.getUpdater().update(mln, gradients[0], iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        mln.params().subi(gradients[0].gradient());
        Collection<TrainingListener> iterationListeners = mln.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener listener : iterationListeners) {
                listener.iterationDone(mln, iterationCount, epochCount);
            }
        }
        mlnConf.setIterationCount(iterationCount + 1);
    }

    @Override
    public double getLatestScore() {
        return mln.score();
    }

    @Override
    public void save(OutputStream os) throws IOException {
        ModelSerializer.writeModel(mln, os, true);
    }

    @Override
    public void save(String filename) throws IOException {
        ModelSerializer.writeModel(mln, filename, true);
    }

    @Override
    public MultiLayerNetwork getNeuralNetwork() {
        return mln;
    }

    @Override
    public INDArray forwardPass(INDArray batch) {
        return output(batch);
    }

    @Override
    public Gradient computeGradient(INDArray input, INDArray expectedOutput) {
        return gradient(input, expectedOutput)[0];
    }

    @Override
    public void applyGradient(Gradient gradient, int batchSize) {
        Gradient[] gradients = {gradient};
        applyGradient(gradients, batchSize);
    }
}
