package nl.tudelft.trustchain.fedml.ai;



import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.index.Index;

import java.util.Iterator;

import static org.tensorflow.ndarray.index.Indices.range;

/**
 * Basic batch iterator across images presented in datset.
 */
public class ImageBatchIterator implements Iterator<ImageBatch> {

    @Override
    public boolean hasNext() {
        return batchStart < numImages;
    }

    @Override
    public ImageBatch next() {
        long nextBatchSize = Math.min(batchSize, numImages - batchStart);
        Index range = range(batchStart, batchStart + nextBatchSize);
        batchStart += nextBatchSize;
        return new ImageBatch(images.slice(range), labels.slice(range));
    }

    public ImageBatchIterator(int batchSize, ByteNdArray images, ByteNdArray labels) {
        this.batchSize = batchSize;
        this.images = images;
        this.labels = labels;
        this.numImages = images != null ? images.shape().size(0) : 0;
        this.batchStart = 0;
    }

    private final int batchSize;
    private final ByteNdArray images;
    private final ByteNdArray labels;
    private final long numImages;
    private int batchStart;
}
