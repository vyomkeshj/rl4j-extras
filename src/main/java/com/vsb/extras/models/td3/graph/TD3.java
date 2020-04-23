package com.vsb.extras.models.td3.graph;

import com.vsb.extras.models.td3.Utils;
import com.vsb.extras.models.td3.hindsightReplay.Experience;
import com.vsb.extras.models.td3.hindsightReplay.ReplayBuffer;
import com.vsb.extras.models.td3.networks.Network;
import com.vsb.extras.models.td3.networks.TD3Network;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TD3 {
    private TD3Network actor;
    private TD3Network actorTarget;
    private TD3Network criticA;
    private TD3Network criticATarget;
    private TD3Network criticB;
    private TD3Network criticBTarget;

    final ReplayBuffer<Experience> experienceReplayBuffer;
    final ReplayBuffer<Experience> localBatch;

    private double selfDiscount = 0.90d;
    private double tau = 0.7;
    public TD3(MultiLayerNetwork actorNetworkKind, MultiLayerNetwork criticNetworkKind) {
       initializeNetworks(actorNetworkKind, criticNetworkKind);
       experienceReplayBuffer = new ReplayBuffer<>();   //todo: replace with HER
       localBatch = new ReplayBuffer<>();
    }

    public void refreshLocalBatch(int batchSize) {
        localBatch.clearExperiences();
        localBatch.insertExperiences(experienceReplayBuffer.fetchExperienceBatch(batchSize));
    }

    public INDArray getActionForState(INDArray state) {
        return actorTarget.forwardPass(state);
    }

    public void updateCritics() {
        INDArray nextActions = actorTarget.forwardPass(localBatch.getNextStatesFromExperiences());  //fixme: is it an array or arrays of array
        //todo: add noise
        INDArray targetQ1 = criticATarget.forwardPass(nextActions.addColumnVector(localBatch.getNextStatesFromExperiences()));
        INDArray targetQ2 = criticBTarget.forwardPass(nextActions.addColumnVector(localBatch.getNextStatesFromExperiences()));

        INDArray targetQ = Utils.minQ(targetQ1, targetQ2);
        INDArray intermediateResult = localBatch.getCompletionStatusFromExperiences().muli(selfDiscount);
        intermediateResult = intermediateResult.muli(targetQ);
        targetQ = experienceReplayBuffer.getRewardsFromExperiences().add(intermediateResult);

        INDArray currentQ1 = criticA.forwardPass(localBatch.getActionsFromExperiences().addColumnVector(localBatch.getStatesFromExperiences()));
        INDArray currentQ2 = criticB.forwardPass(localBatch.getActionsFromExperiences().addColumnVector(localBatch.getStatesFromExperiences()));

        INDArray criticLoss; //todo: perform mse loss, back propagate this loss to the critics
    }

    public void updateActor() {
        INDArray actorLoss = criticA.forwardPass(localBatch.getStatesFromExperiences().addColumnVector(actor.forwardPass(localBatch.getStatesFromExperiences())));
        actorLoss.muli(-1);
    }

    public void updateCriticTargets() {
        INDArray criticAParams = criticA.getNeuralNetwork().params();
        INDArray criticBParams = criticB.getNeuralNetwork().params();

        INDArray criticATargetParams = criticATarget.getNeuralNetwork().params();
        INDArray criticBTargetParams = criticBTarget.getNeuralNetwork().params();

        INDArray targetParams = criticATargetParams.mul(tau);
        targetParams.add(criticAParams.mul(1-tau));

        criticATarget.getNeuralNetwork().setParams(targetParams);

        targetParams = criticBTargetParams.mul(tau);
        targetParams.add(criticBParams.mul(1-tau));

        criticBTarget.getNeuralNetwork().setParams(targetParams);   //fixme: or parameters?
    }

    public void updateActorTargets() {
        INDArray actorParams = actor.getNeuralNetwork().params();
        INDArray actorTargetParams = actorTarget.getNeuralNetwork().params();
        INDArray targetParams = actorTargetParams.mul(tau);
        targetParams.add(actorParams.mul(1-tau));
        actorTarget.getNeuralNetwork().setParams(targetParams);
    }

    private void initializeNetworks(MultiLayerNetwork actors, MultiLayerNetwork critics) {
        this.actor = new Network(actors.clone());
        this.actorTarget = new Network(actors.clone());
        this.criticA = new Network(critics.clone());
        this.criticATarget = new Network(critics.clone());
        this.criticB = new Network(critics.clone());
        this.criticBTarget = new Network(critics.clone());
    }

}
