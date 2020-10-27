package nl.tudelft.trustchain.fedml.ai.dataset.cifar;


import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.IOException;

import nl.tudelft.trustchain.fedml.ai.IteratorDistributions;
import nl.tudelft.trustchain.fedml.ai.MaxTestSamples;

public class CustomCifar10DataSetIterator extends RecordReaderDataSetIterator {
    public CustomCifar10DataSetIterator(int batchSize, int[] imgDim, DataSetType set,
                                        ImageTransform imageTransform, long rngSeed, IteratorDistributions iteratorDistribution) throws IOException {
        super(new CustomCifar10Fetcher(iteratorDistribution, null).getRecordReader(rngSeed, imgDim, set, imageTransform), batchSize, 1, Cifar10Fetcher.NUM_LABELS);
    }

    public CustomCifar10DataSetIterator(int batchSize, int[] imgDim, DataSetType set,
                                        ImageTransform imageTransform, long rngSeed, IteratorDistributions iteratorDistribution, MaxTestSamples maxSamples) throws IOException {
        super(new CustomCifar10Fetcher(iteratorDistribution, maxSamples).getRecordReader(rngSeed, imgDim, set, imageTransform), batchSize, 1, Cifar10Fetcher.NUM_LABELS);
    }
}
