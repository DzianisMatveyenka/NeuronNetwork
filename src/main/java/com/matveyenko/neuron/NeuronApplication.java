package com.matveyenko.neuron;

import com.matveyenko.neuron.base.SimpleNetwork;
import com.matveyenko.neuron.model.NeuronNetwork;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Integer.parseInt;

/**
 * Created by 1 on 05.11.2017.
 */
public class NeuronApplication {

    private static final String FILE_TRAIN_PATH = "csv/mnist_train.csv";
    private static final String FILE_TEST_PATH = "csv/mnist_test.csv";
    private static final String CSV_DELIMITER = ",";
    private static final double[] model = new double[10];
    private static final Map<Integer, double[]> map = new HashMap<>();

    static {
        Arrays.fill(model, 0.01);

        map.put(0, getModel(0).get());
        map.put(1, getModel(1).get());
        map.put(2, getModel(2).get());
        map.put(3, getModel(3).get());
        map.put(4, getModel(4).get());
        map.put(5, getModel(5).get());
        map.put(6, getModel(6).get());
        map.put(7, getModel(7).get());
        map.put(8, getModel(8).get());
        map.put(9, getModel(9).get());
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws IOException {
        int inputNodes = 784;
        int hiddenNodes = 250;
        int outputNodes = 10;
        double learningFactor = 0.11;

        NeuronNetwork neuronNetwork = new SimpleNetwork(inputNodes, hiddenNodes, outputNodes, learningFactor);

        for (int i = 0; i < 5; i++) {
            try (InputStream resourceAsStream = NeuronApplication.class.getClassLoader().getResourceAsStream(FILE_TRAIN_PATH)) {
                List<String> list = IOUtils.readLines(resourceAsStream);
                for (String row : list) {
                    String[] split = row.split(CSV_DELIMITER);
                    int[] image = Stream.of(split)
                            .mapToInt(Integer::parseInt)
                            .skip(1)
                            .toArray();

                    double[] imageDouble = IntStream.of(image)
                            .mapToDouble(v -> (v / 255.0 * 0.99) + 0.01)
                            .toArray();

                    neuronNetwork.train(imageDouble, map.get(parseInt(split[0])));
                }
            }
        }

        int ok = 0;
        int bad = 0;

        try (InputStream resourceAsStream = NeuronApplication.class.getClassLoader().getResourceAsStream(FILE_TEST_PATH)) {
            List<String> list = IOUtils.readLines(resourceAsStream);
            for (String row : list) {
                String[] split = row.split(CSV_DELIMITER);
                int[] image = Stream.of(split)
                        .mapToInt(Integer::parseInt)
                        .skip(1)
                        .toArray();

                double[] imageDouble = IntStream.of(image)
                        .mapToDouble(v -> (v / 255.0 * 0.99) + 0.01)
                        .toArray();

                int expectedValue = Integer.parseInt(split[0]);
                double[][] actualValue = neuronNetwork.query(imageDouble);
                Optional<Pair> found = Stream.of(new Pair(0, actualValue[0][0]),
                        new Pair(1, actualValue[1][0]),
                        new Pair(2, actualValue[2][0]),
                        new Pair(3, actualValue[3][0]),
                        new Pair(4, actualValue[4][0]),
                        new Pair(5, actualValue[5][0]),
                        new Pair(6, actualValue[6][0]),
                        new Pair(7, actualValue[7][0]),
                        new Pair(8, actualValue[8][0]),
                        new Pair(9, actualValue[9][0]))
                        .sorted(Comparator.comparingDouble(v -> ((Pair) v).value).reversed())
                        .findFirst();
                if (found.isPresent() && found.get().key == expectedValue) {
                    ok++;
                } else {
                    bad++;
                }
            }
        }

        System.out.println("OK: " + ok / 100.0);
        System.out.println("Bad: " + bad / 100.0);
    }

    private static Supplier<double[]> getModel(int index) {
        return () -> {
            double[] array = Arrays.copyOf(model, model.length);
            array[index] = 0.99;
            return array;
        };
    }

    public static class Pair {

        int key;
        double value;

        public Pair(int key, double value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public String toString() {
            return String.valueOf(key) + " -> " + value;
        }
    }
}
