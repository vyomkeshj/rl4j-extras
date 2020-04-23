package com.vsb.extras.models.td3.hindsightReplay;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class ReplayBuffer<T extends Experience> {
    ArrayList<T> experienceList = new ArrayList<>();

    public void insertExperience(T experience) {
        experienceList.add(experience);
    }

    public void insertExperiences(List<T> experiences) {
        experienceList.addAll(experiences);
    }

    public void clearExperiences() {
        experienceList.clear();
    }

    public ArrayList<T> fetchExperienceBatch(int batchSize) {
        //todo: randomly select elements from experienceList and return it

        return new ArrayList<T>();
    }

    public INDArray getStatesFromExperiences() {
        INDArray returnStates = null;
        for(T experience: experienceList) {
            if(returnStates == null) {
                returnStates = experience.getState();
            } else {
                returnStates = returnStates.addiRowVector(experience.getState());
            }
        }
        return returnStates;
    }

    public INDArray getActionsFromExperiences() {
        INDArray returnActions = null;
        for(T experience: experienceList) {
            if(returnActions == null) {
                returnActions = experience.getAction();
            } else {
                returnActions = returnActions.addiRowVector(experience.getAction());
            }
        }
      return returnActions;
    }

    public INDArray getNextStatesFromExperiences() {
        INDArray returnStates = null;
        for(T experience: experienceList) {
            if(returnStates == null) {
                returnStates = experience.getNextState();
            } else {
                returnStates = returnStates.addiRowVector(experience.getNextState());
            }
        }
        return returnStates;
    }

    public INDArray getRewardsFromExperiences() {
        INDArray returnStates = null;
        for(T experience: experienceList) {
            if(returnStates == null) {
                returnStates = Nd4j.create(new double[]{experience.getReward()},new int[]{1,1});
            } else {
                returnStates = returnStates.addiRowVector(Nd4j.create(new double[]{experience.getReward()},new int[]{1,1}));
            }
        }
        return returnStates;
    }

    public INDArray getCompletionStatusFromExperiences() {
        INDArray returnStates = null;
        for(T experience: experienceList) {
            if(returnStates == null) {
                returnStates = Nd4j.create(new double[]{experience.isNotDone()},new int[]{1,1});
            } else {
                returnStates = returnStates.addiRowVector(Nd4j.create(new double[]{experience.isNotDone()},new int[]{1,1}));
            }
        }
        return returnStates;
    }
}
