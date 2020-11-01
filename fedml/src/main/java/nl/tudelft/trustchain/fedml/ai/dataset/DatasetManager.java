package nl.tudelft.trustchain.fedml.ai.dataset;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class DatasetManager {
    protected static int[] findMatchingImageIndices(int label, int[] tmpLabelsArr) {
        return IntStream.range(0, tmpLabelsArr.length).filter(j -> label == tmpLabelsArr[j]).toArray();
    }

    protected static int[] shuffle(int[] matchingImageIndices, int seed) {
        final List<Integer> tmp = Arrays.stream(matchingImageIndices.clone()).boxed().collect(Collectors.toList());
        Collections.shuffle(tmp, new Random(seed));
        return tmp.stream().mapToInt(j -> j).toArray();
    }

    protected static int min(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }

    public static int calculateTotalExamples(List<Integer> iteratorDistribution, int maxTestSamples, int[] labelsArray) {
        return IntStream.range(0, iteratorDistribution.size()).map(i -> min(iteratorDistribution.get(i),
            maxTestSamples, (int) Arrays.stream(labelsArray).filter(j -> j == i).count())).sum();
    }
}
