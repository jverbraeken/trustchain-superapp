package nl.tudelft.trustchain.fedml.ai.dataset.cifar;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.nd4j.base.Preconditions;

import java.io.File;
import java.util.Random;

import nl.tudelft.trustchain.fedml.IteratorDistributions;

public class CustomCifar10Fetcher extends Cifar10Fetcher {
    private final IteratorDistributions iteratorDistribution;
    private final int maxTestSamples;

    public CustomCifar10Fetcher(IteratorDistributions iteratorDistribution, int maxTestSamples) {
        this.iteratorDistribution = iteratorDistribution;
        this.maxTestSamples = maxTestSamples;
    }

    @Override
    public RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType dataSetType, ImageTransform imageTransform) {
        Preconditions.checkState(imgDim == null || imgDim.length == 2, "Invalid image dimensions: must be null or lenth 2. Got: %s", imgDim);
        // check empty cache
        File localCache = getLocalCacheDir();
        deleteIfEmpty(localCache);

        try {
            if (!localCache.exists()) {
                downloadAndExtract();
            }
        } catch (Exception e) {
            throw new RuntimeException("Could not download CIFAR-10", e);
        }

        Random rng = new Random(rngSeed);
        File datasetPath;
        switch (dataSetType) {
            case TRAIN:
                datasetPath = new File(localCache, "/train/");
                break;
            case TEST:
                datasetPath = new File(localCache, "/test/");
                break;
            default:
                throw new IllegalArgumentException("You will need to manually create and iterate a validation directory, CIFAR-10 does not provide labels");
        }

        // dataSetType up file paths
        RandomPathFilter pathFilter = new RandomPathFilter(rng, BaseImageLoader.ALLOWED_FORMATS);
        FileSplit filesInDir = new FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, iteratorDistribution.getValue().stream().mapToDouble(i->i).toArray());

        int h = (imgDim == null ? Cifar10Fetcher.INPUT_HEIGHT : imgDim[0]);
        int w = (imgDim == null ? Cifar10Fetcher.INPUT_WIDTH : imgDim[1]);
        ImageRecordReader rr = new ImageRecordReader(h, w, Cifar10Fetcher.INPUT_CHANNELS, new ParentPathLabelGenerator(), imageTransform);

        try {
            rr.initialize(filesInDirSplit[0]);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return rr;
    }
}
