package nl.tudelft.trustchain.fedml.ai.dataset.har;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.File;
import java.io.IOException;

import nl.tudelft.trustchain.fedml.ai.IteratorDistributions;
import nl.tudelft.trustchain.fedml.ai.MaxTestSamples;

public class HARIterator extends BaseDatasetIterator {

    /**
     * Constructor to get the full HAL data set (either test or train sets) with shuffling based on a random seed.
     */
    public HARIterator(File baseDirectory, int batchSize, int seed, IteratorDistributions iteratorDistribution, DataSetType dataSetType) throws IOException {
        super(batchSize, -1, new HARDataFetcher(baseDirectory, seed, iteratorDistribution.getValue(), dataSetType, null));
    }

    /**
     * Constructor to get the full HAL data set (either test or train sets) with shuffling based on a random seed.
     */
    public HARIterator(File baseDirectory, int batchSize, int seed, IteratorDistributions iteratorDistribution, DataSetType dataSetType, MaxTestSamples maxTestSamples) throws IOException {
        super(batchSize, -1, new HARDataFetcher(baseDirectory, seed, iteratorDistribution.getValue(), dataSetType, maxTestSamples));
    }
}
