package nl.tudelft.trustchain.fedml.ai.dataset.mnist;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

import nl.tudelft.trustchain.fedml.Behaviors;


public class CustomMnistDataFetcher extends BaseDataFetcher {
    protected static final long CHECKSUM_TRAIN_FEATURES = 2094436111L;
    protected static final long CHECKSUM_TRAIN_LABELS = 4008842612L;
    protected static final long CHECKSUM_TEST_FEATURES = 2165396896L;
    protected static final long CHECKSUM_TEST_LABELS = 2212998611L;

    protected static final long[] CHECKSUMS_TRAIN = new long[]{CHECKSUM_TRAIN_FEATURES, CHECKSUM_TRAIN_LABELS};
    protected static final long[] CHECKSUMS_TEST = new long[]{CHECKSUM_TEST_FEATURES, CHECKSUM_TEST_LABELS};
    protected transient CustomMnistManager man;
    protected int[] order;
    protected Random rng;
    protected boolean oneIndexed = false;
    protected boolean fOrder = false; //MNIST is C order, EMNIST is F order
    private float[][] featureData = null;

    public CustomMnistDataFetcher(List<Integer> iteratorDistribution, int seed, DataSetType dataSetType, int maxTestSamples, Behaviors behavior) throws IOException {
        if (!mnistExists()) {
            new MnistFetcher().downloadAndUntar();
        }

        String MNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").getAbsolutePath();
        String images;
        String labels;
        long[] checksums;
        int maxExamples;
        if (dataSetType == DataSetType.TRAIN) {
            images = FilenameUtils.concat(MNIST_ROOT, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED);
            labels = FilenameUtils.concat(MNIST_ROOT, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED);
            maxExamples = MnistDataFetcher.NUM_EXAMPLES;
            checksums = CHECKSUMS_TRAIN;
            man = new CustomMnistManager(images, labels, maxExamples, iteratorDistribution, Integer.MAX_VALUE, seed, behavior);
        } else {
            images = FilenameUtils.concat(MNIST_ROOT, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED);
            labels = FilenameUtils.concat(MNIST_ROOT, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED);
            maxExamples = MnistDataFetcher.NUM_EXAMPLES_TEST;
            checksums = CHECKSUMS_TEST;
            man = new CustomMnistManager(images, labels, maxExamples, iteratorDistribution, maxTestSamples, seed, behavior);
        }
        String[] files = new String[]{images, labels};

        try {
            man = new CustomMnistManager(images, labels, maxExamples, iteratorDistribution, dataSetType == DataSetType.TRAIN ? Integer.MAX_VALUE : maxTestSamples, seed, behavior);
            validateFiles(files, checksums);
        } catch (Exception e) {
            try {
                FileUtils.deleteDirectory(new File(MNIST_ROOT));
            } catch (Exception e2) {
                // Ignore
            }
            new MnistFetcher().downloadAndUntar();
            man = new CustomMnistManager(images, labels, maxExamples, iteratorDistribution, dataSetType == DataSetType.TRAIN ? Integer.MAX_VALUE : maxTestSamples, seed, behavior);
            validateFiles(files, checksums);
        }
        totalExamples = man.getNumSamples();

        numOutcomes = 10;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();

        order = IntStream.range(0, totalExamples).toArray();
        rng = new Random(seed);
        reset(); //Shuffle order
    }

    private boolean mnistExists() {
        String MNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").getAbsolutePath();
        File f = new File(MNIST_ROOT, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED);
        return f.exists();
    }

    private void validateFiles(String[] files, long[] checksums) {
        try {
            for (int i = 0; i < files.length; i++) {
                File f = new File(files[i]);
                Checksum adler = new Adler32();
                long checksum = f.exists() ? FileUtils.checksum(f, adler).getValue() : -1;
                if (!f.exists() || checksum != checksums[i]) {
                    throw new IllegalStateException("Failed checksum: expected " + checksums[i] +
                        ", got " + checksum + " for file: " + f);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        INDArray labels = Nd4j.zeros(DataType.FLOAT, numExamples, numOutcomes);

        if (featureData == null || featureData.length < numExamples) {
            featureData = new float[numExamples][28 * 28];
        }

        int actualExamples = 0;
        byte[] working = null;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            byte[] img = man.readImageUnsafe(order[cursor]);

            if (fOrder) {
                //EMNIST requires F order to C order
                if (working == null) {
                    working = new byte[28 * 28];
                }
                for (int j = 0; j < 28 * 28; j++) {
                    working[j] = img[28 * (j % 28) + j / 28];
                }
                img = working;
            }

            int label = man.readLabel(order[cursor]);
            if (oneIndexed) {
                //For some inexplicable reason, Emnist LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
                // is indexed (0 to nClasses-1) :/
                label--;
            }

            labels.put(actualExamples, label, 1.0f);

            for (int j = 0; j < img.length; j++) {
                featureData[actualExamples][j] = ((int) img[j]) & 0xFF;
            }

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

        features.divi(255.0);

        curr = new DataSet(features, labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        MathUtils.shuffleArray(order, rng);
    }

    @NotNull
    public List<String> getLabels() {
        return man.getLabels();
    }
}
