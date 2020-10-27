package nl.tudelft.trustchain.fedml.ai.dataset.har;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager;


public class HARManager extends DatasetManager {
    private final String[][] dataArr;
    private final int[] labelsArr;

    public HARManager(File[] dataFiles, File labelsFile, List<Integer> iteratorDistribution, int maxSamples, int seed) throws IOException {
        final List<String>[] tmpDataArr = loadData(dataFiles);
        int[] tmpLabelsArr = loadLabels(labelsFile);

        int totalExamples = calculateTotalExamples(iteratorDistribution, maxSamples, tmpLabelsArr);
        final DataLabelContainer res = sampleData(tmpDataArr, tmpLabelsArr, totalExamples, iteratorDistribution, maxSamples, seed);
        dataArr = res.data;
        labelsArr = res.labels;
    }

    private static DataLabelContainer sampleData(List<String>[] tmpDataArr, int[] tmpLabelsArr, int totalExamples, List<Integer> iteratorDistribution, int maxSamples, int seed) {
        final String[][] dataArr = new String[totalExamples][tmpDataArr.length];
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
                final int index = j;
                dataArr[count] = Arrays.stream(tmpDataArr).map(dataFile -> dataFile.get(shuffledMatchingImageIndices[index])).toArray(String[]::new);
                labelsArr[count] = tmpLabelsArr[shuffledMatchingImageIndices[j]];
                count++;
            }
        }
        return new DataLabelContainer(dataArr, labelsArr);
    }

    private int[] loadLabels(File labelsFile) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(labelsFile))) {
            return reader.lines().mapToInt(i -> Integer.parseInt(i) - 1).toArray();  // labels start at 1 instead of 0
        }
    }

    private List<String>[] loadData(File[] dataFiles) throws IOException {
        final List<String>[] data = (List<String>[]) new List[dataFiles.length];
        for (int i = 0; i < dataFiles.length; i++) {
            try (BufferedReader reader = new BufferedReader(new FileReader(dataFiles[i]))) {
                data[i] = reader.lines().collect(Collectors.toList());
            }
        }
        return data;
    }

    public String[] readEntryUnsafe(int i) {
        return dataArr[i];
    }

    public int readLabel(int i) {
        return labelsArr[i];
    }

    public int getNumSamples() {
        assert dataArr.length == labelsArr.length;
        return dataArr.length;
    }

    /* Why??? does Java not support tuples??? */
    private static class DataLabelContainer {
        private String[][] data;
        private int[] labels;

        public DataLabelContainer(String[][] data, int[] labels) {
            this.data = data;
            this.labels = labels;
        }
    }
}
