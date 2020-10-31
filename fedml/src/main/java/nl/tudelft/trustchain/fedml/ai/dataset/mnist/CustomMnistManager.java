package nl.tudelft.trustchain.fedml.ai.dataset.mnist;


import org.deeplearning4j.datasets.mnist.MnistImageFile;
import org.deeplearning4j.datasets.mnist.MnistLabelFile;

import java.io.IOException;
import java.util.List;

import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager;


public class CustomMnistManager extends DatasetManager {
    private final MnistImageFile mnistImageFile;
    private final MnistLabelFile mnistLabelFile;
    private final byte[][] imagesArr;
    private final int[] labelsArr;

    public CustomMnistManager(String imagesFile, String labelsFile, int numExamples, List<Integer> iteratorDistribution, int maxSamples, int seed) throws IOException {
        mnistImageFile = new MnistImageFile(imagesFile, "r");
        mnistLabelFile = new MnistLabelFile(labelsFile, "r");
        final byte[][] tmpImagesArr = loadImages(mnistImageFile, numExamples);
        final int[] tmpLabelsArr = loadLabels(mnistLabelFile, numExamples);

        final int totalExamples = calculateTotalExamples(iteratorDistribution, maxSamples, tmpLabelsArr);
        final ImageLabelContainer res = sampleData(tmpImagesArr, tmpLabelsArr, totalExamples, iteratorDistribution, maxSamples, seed);
        imagesArr = res.images;
        labelsArr = res.labels;
    }

    private static int[] loadLabels(MnistLabelFile mnistLabelFile, int numExamples) throws IOException {
        return mnistLabelFile.readLabels(numExamples);
    }

    private static byte[][] loadImages(MnistImageFile mnistImageFile, int numExamples) throws IOException {
        return mnistImageFile.readImagesUnsafe(numExamples);
    }

    private static ImageLabelContainer sampleData(byte[][] tmpImagesArr, int[] tmpLabelsArr, int totalExamples, List<Integer> iteratorDistribution, int maxSamples, int seed) {
        final byte[][] imagesArr = new byte[totalExamples][tmpImagesArr[0].length];
        final int[] labelsArr = new int[totalExamples];
        int count = 0;
        for (int label = 0; label < iteratorDistribution.size(); label++) {
            final int maxSamplesOfLabel = iteratorDistribution.get(label);
            /*
            Absolute HORRIBLE code!! Seriously considering converting this java code to Kotlin
            Java does not support Math.min of 3 integers
            Java does not provide a simple function to shuffle an array
            Java does not provide streams for raw bytes
            Java does not provide a simple function to convert an int array to a list or the other way around
            ...
             */
            final int[] matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr);
            final int[] shuffledMatchingImageIndices = shuffle(matchingImageIndices, seed);
            for (int j = 0; j < min(shuffledMatchingImageIndices.length, maxSamplesOfLabel, maxSamples); j++) {
                imagesArr[count] = tmpImagesArr[shuffledMatchingImageIndices[j]];
                labelsArr[count] = tmpLabelsArr[shuffledMatchingImageIndices[j]];
                count++;
            }
        }
        return new ImageLabelContainer(imagesArr, labelsArr);
    }

    public byte[] readImageUnsafe(int i) {
        return imagesArr[i];
    }

    public int readLabel(int i) {
        return labelsArr[i];
    }

    public int getNumSamples() {
        assert imagesArr.length == labelsArr.length;
        return imagesArr.length;
    }

    /**
     * Get the underlying images file as {@link MnistImageFile}.
     *
     * @return {@link MnistImageFile}.
     */
    public MnistImageFile getImages() {
        return mnistImageFile;
    }

    /* Why??? does Java not support tuples??? */
    private static class ImageLabelContainer {
        private byte[][] images;
        private int[] labels;

        public ImageLabelContainer(byte[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }
}