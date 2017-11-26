package com.matveyenko.neuron.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * Created by 1 on 06.11.2017.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class MatrixManager {

    private static final Sigmoid SIGMOID = new Sigmoid();

    public static double[][] generate(int rows, int columns, int countNodes) {
        Random random = new Random();
        double[][] matrix = new double[rows][columns];
        for (double[] d : matrix) {
            for (int i = 0; i < d.length; i++) {
                DoubleStream doubles = random.doubles(1, -1 * Math.pow(countNodes, -0.5), Math.pow(countNodes, -0.5));
                d[i] = doubles.findFirst().getAsDouble();
            }
        }

        return matrix;
    }

    public static RealMatrix applySigma(RealMatrix realMatrix) {
        double[][] data = realMatrix.getData();
        for (int i = 0; i < data.length; i++) {
            double[] datum = data[i];
            for (int j = 0; j < datum.length; j++) {
                data[i][j] = SIGMOID.value(datum[j]);
            }
        }

        return MatrixUtils.createRealMatrix(data);
    }

    public static RealMatrix getError(RealMatrix output, RealMatrix target) {
        return target.subtract(output);
    }

    public static RealMatrix applyError(RealMatrix weights, RealMatrix error) {
        return weights.transpose().multiply(error);
    }

    public static RealMatrix multiplayByElem(RealMatrix first, RealMatrix second) {
        RealMatrix matrix = MatrixUtils.createRealMatrix(first.getData());
        double[][] data = first.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                matrix.multiplyEntry(i, j, second.getEntry(i, j));
            }
        }

        return matrix;
    }
}
