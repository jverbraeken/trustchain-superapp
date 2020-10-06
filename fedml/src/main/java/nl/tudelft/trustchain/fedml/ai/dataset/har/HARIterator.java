package nl.tudelft.trustchain.fedml.ai.dataset.har;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.File;
import java.io.IOException;

public class HARIterator extends BaseDatasetIterator {

    /**
     * Constructor to get the full HAL data set (either test or train sets) with shuffling based on a random seed.
     *
     * @param batchSize
     * @param train
     * @throws IOException
     */
    public HARIterator(File baseDirectory, int batchSize, boolean train, int seed) throws IOException {
        super(batchSize,
            (train ? HARDataFetcher.NUM_EXAMPLES : HARDataFetcher.NUM_EXAMPLES_TEST),
            new HARDataFetcher(baseDirectory, train, seed, (train ? HARDataFetcher.NUM_EXAMPLES : HARDataFetcher.NUM_EXAMPLES_TEST)));
    }
}
