package com.matveyenko.neuron.base;

import com.matveyenko.neuron.model.NeuronNetwork;
import com.matveyenko.neuron.util.MatrixManager;
import lombok.Getter;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;

import static com.matveyenko.neuron.util.MatrixManager.applySigma;
import static com.matveyenko.neuron.util.MatrixManager.getError;
import static com.matveyenko.neuron.util.MatrixManager.multiplayByElem;
import static org.apache.commons.math3.linear.MatrixUtils.createRowRealMatrix;

/**
 * Created by 1 on 05.11.2017.
 */
@Getter
public class SimpleNetwork implements NeuronNetwork {

    private double learningFactor;
    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;
    private RealMatrix inputHiddenWeight;
    private RealMatrix hiddenOutputWeight;

    private InputHandler inputHandler = new InputHandler();

    public SimpleNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningFactor) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningFactor = learningFactor;
        this.inputHiddenWeight = MatrixUtils.createRealMatrix(MatrixManager.generate(hiddenNodes, inputNodes, hiddenNodes));
        this.hiddenOutputWeight = MatrixUtils.createRealMatrix(MatrixManager.generate(outputNodes, hiddenNodes, outputNodes));
    }

    public void train(double[] inputs, double[] targets) {
        query(inputs);

        double[] outputs = inputHandler.output.transpose().getData()[0];
        RealMatrix outputErrors = getError(createRowRealMatrix(outputs).transpose(), createRowRealMatrix(targets).transpose());
        RealMatrix hiddenErrors = hiddenOutputWeight.transpose().multiply(outputErrors);

        RealMatrix updateWeightForHidden = multiplayByElem(
                multiplayByElem(outputErrors, inputHandler.output),
                getOneMatrix(outputNodes).subtract(inputHandler.output)
        ).multiply(inputHandler.outputHiddenLayer.transpose()).scalarMultiply(learningFactor);
        hiddenOutputWeight = hiddenOutputWeight.add(updateWeightForHidden);

        RealMatrix updateWeightForInput = multiplayByElem(
                multiplayByElem(hiddenErrors, inputHandler.outputHiddenLayer),
                getOneMatrix(hiddenNodes).subtract(inputHandler.outputHiddenLayer)
        ).multiply(MatrixUtils.createRowRealMatrix(inputs)).scalarMultiply(learningFactor);
        inputHiddenWeight = inputHiddenWeight.add(updateWeightForInput);
    }

    private RealMatrix getOneMatrix(int size) {
        double[] one = new double[size];
        Arrays.fill(one, 1.0);

        return MatrixUtils.createColumnRealMatrix(one);
    }

    public double[][] query(double[] trainSet) {
        RealMatrix trainMatrix = createRowRealMatrix(trainSet).transpose();

        return inputHandler.handle(trainMatrix);
    }

    public class InputHandler {

        RealMatrix inputForHiddenLayer;
        RealMatrix outputHiddenLayer;
        RealMatrix inputForOutputLayer;
        RealMatrix output;

        double[][] handle(RealMatrix trainMatrix) {
            this.inputForHiddenLayer = inputHiddenWeight.multiply(trainMatrix);
            this.outputHiddenLayer = applySigma(inputForHiddenLayer);
            this.inputForOutputLayer = hiddenOutputWeight.multiply(outputHiddenLayer);
            this.output = applySigma(inputForOutputLayer);

            return output.getData();
        }
    }
}
