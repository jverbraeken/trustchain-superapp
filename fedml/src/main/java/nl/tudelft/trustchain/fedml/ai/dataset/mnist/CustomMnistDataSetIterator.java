package nl.tudelft.trustchain.fedml.ai.dataset.mnist;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.IOException;
import java.util.List;

import nl.tudelft.trustchain.fedml.ai.IteratorDistributions;
import nl.tudelft.trustchain.fedml.ai.MaxTestSamples;

public class CustomMnistDataSetIterator extends BaseDatasetIterator {
    public CustomMnistDataSetIterator(int batch, IteratorDistributions iteratorDistribution, int seed, DataSetType dataSetType) throws IOException {
        super(batch, -1, new CustomMnistDataFetcher(iteratorDistribution.getValue(), seed, dataSetType, Integer.MAX_VALUE));
    }
    public CustomMnistDataSetIterator(int batch, IteratorDistributions iteratorDistribution, int seed, DataSetType dataSetType, MaxTestSamples maxTestSamples) throws IOException {
        super(batch, -1, new CustomMnistDataFetcher(iteratorDistribution.getValue(), seed, dataSetType, maxTestSamples.getValue()));
    }
}
