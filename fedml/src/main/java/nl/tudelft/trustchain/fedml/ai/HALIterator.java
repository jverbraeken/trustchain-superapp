package nl.tudelft.trustchain.fedml.ai;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.File;
import java.io.IOException;

public class HALIterator extends BaseDatasetIterator {

    /**
     * Constructor to get the full HAL data set (either test or train sets) with shuffling based on a random seed.
     *
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public HALIterator(File baseDirectory, int batchSize, boolean train, int seed) throws IOException {
        super(batchSize,
            (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST),
            new HALDataFetcher(baseDirectory, train, seed, (train ? MnistDataFetcher.NUM_EXAMPLES : MnistDataFetcher.NUM_EXAMPLES_TEST)));
    }
}
