package com.matveyenko.neuron.model;

/**
 * Created by 1 on 05.11.2017.
 */
public interface NeuronNetwork {

    void train(double[] inputs, double[] targets);

    double[][] query(double[] trainSet);
}
