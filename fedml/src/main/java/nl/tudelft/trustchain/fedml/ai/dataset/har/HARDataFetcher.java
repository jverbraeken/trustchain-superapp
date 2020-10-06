package nl.tudelft.trustchain.fedml.ai.dataset.har;


import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;


/**
 * Data fetcher for the HAL dataset
 */
public class HARDataFetcher extends BaseDataFetcher {
    public static final int NUM_EXAMPLES = 7352;
    public static final int NUM_EXAMPLES_TEST = 2947;
    public static final int NUM_ATTRIBUTES = 561;
    public static final int NUM_LABELS = 6;

    protected transient HARManager man;
    protected boolean train;
    protected int[] order;
    protected Random rng;

    public HARDataFetcher(File baseDirectory, boolean train, long rngSeed, int numExamples) throws IOException {
        File[] data = new File[9];
        File labels;
        if (train) {
            data[0] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_acc_x_train.txt").toFile();
            data[1] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_acc_y_train.txt").toFile();
            data[2] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_acc_z_train.txt").toFile();
            data[3] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_gyro_x_train.txt").toFile();
            data[4] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_gyro_y_train.txt").toFile();
            data[5] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "body_gyro_z_train.txt").toFile();
            data[6] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "total_acc_x_train.txt").toFile();
            data[7] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "total_acc_y_train.txt").toFile();
            data[8] = Paths.get(baseDirectory.getPath(), "train", "Inertial Signals", "total_acc_z_train.txt").toFile();
            labels = new File(new File(baseDirectory, "train"), "y_train.txt");
            totalExamples = NUM_EXAMPLES;
        } else {
            data[0] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_acc_x_test.txt").toFile();
            data[1] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_acc_y_test.txt").toFile();
            data[2] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_acc_z_test.txt").toFile();
            data[3] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_gyro_x_test.txt").toFile();
            data[4] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_gyro_y_test.txt").toFile();
            data[5] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "body_gyro_z_test.txt").toFile();
            data[6] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "total_acc_x_test.txt").toFile();
            data[7] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "total_acc_y_test.txt").toFile();
            data[8] = Paths.get(baseDirectory.getPath(), "test", "Inertial Signals", "total_acc_z_test.txt").toFile();
            labels = new File(new File(baseDirectory, "test"), "y_test.txt");
            totalExamples = NUM_EXAMPLES_TEST;
        }
        man = new HARManager(data, labels);

        numOutcomes = NUM_LABELS;
        cursor = 0;
        inputColumns = NUM_ATTRIBUTES;
        this.train = train;

        if (train) {
            order = new int[NUM_EXAMPLES];
        } else {
            order = new int[NUM_EXAMPLES_TEST];
        }
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        rng = new Random(rngSeed);
        reset(); //Shuffle order
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        MathUtils.shuffleArray(order, rng);
    }

    @Override
    public DataSet next() {
        return super.next();
    }

    private double[][][] featureData = null;  // samples => time step => feature

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more");
        }

        INDArray labels = Nd4j.zeros(DataType.FLOAT, numExamples, numOutcomes);

        if (featureData == null || featureData.length < numExamples) {
            featureData = new double[numExamples][128][9];
        }

        int actualExamples = 0;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            String[] entries = man.readEntryUnsafe(order[cursor]);
            int label = man.readLabel(order[cursor]);

            labels.put(actualExamples, label, 1.0f);
            double[][] features = new double[9][];
            for (int j = 0; j < entries.length; j++) {
                entries[j] = entries[j].trim();
                String[] parts = entries[j].split("\\s+");
                double[] doubles = Arrays.stream(parts).mapToDouble(Double::parseDouble).toArray();
                features[j] = doubles;
            }
            double[][] timesteps = transposeMatrix(features);
            featureData[actualExamples] = timesteps;
//            featureData[actualExamples] = Arrays.stream(entries).map(entry -> Arrays.stream(entry.trim().split("\\s+")).mapToDouble(Double::parseDouble).toArray()).flatMap(Stream::of).toArray();

            actualExamples++;
        }

        INDArray features;

        if (featureData.length == actualExamples) {
            features = Nd4j.create(featureData);
        } else {
            features = Nd4j.create(Arrays.copyOfRange(featureData, 0, actualExamples));
        }

        if (actualExamples < numExamples) {
            labels = labels.get(NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all());
        }

        curr = new DataSet(features, labels);
    }

    public static double[][] transposeMatrix(double[][] matrix){
        int m = matrix.length;
        int n = matrix[0].length;

        double[][] transposedMatrix = new double[n][m];

        for(int x = 0; x < n; x++) {
            for(int y = 0; y < m; y++) {
                transposedMatrix[x][y] = matrix[y][x];
            }
        }

        return transposedMatrix;
    }
}
