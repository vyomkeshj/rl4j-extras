package com.vsb.extras.models.td3.hindsightReplay;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Experience {
    private INDArray state;
    private INDArray action;
    private INDArray nextState;
    private double reward;
    private int notDone;

    public Experience(INDArray state, INDArray action, INDArray nextState, double reward, int notDone) {
        this.state = state;
        this.action = action;
        this.nextState = nextState;
        this.reward = reward;
        this.notDone = notDone;
    }

    public INDArray getState() {
        return state;
    }

    public INDArray getAction() {
        return action;
    }

    public INDArray getNextState() {
        return nextState;
    }

    public double getReward() {
        return reward;
    }

    public int isNotDone() {
        return notDone;
    }
}
