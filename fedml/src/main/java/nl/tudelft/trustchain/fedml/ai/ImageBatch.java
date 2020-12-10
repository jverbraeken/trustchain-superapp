package nl.tudelft.trustchain.fedml.ai;

import org.tensorflow.ndarray.ByteNdArray;

/**
 * Batch of images for batch training.
 */
public class ImageBatch {

    public ByteNdArray images() {
        return images;
    }

    public ByteNdArray labels() {
        return labels;
    }

    public ImageBatch(ByteNdArray images, ByteNdArray labels) {
        this.images = images;
        this.labels = labels;
    }

    private final ByteNdArray images;
    private final ByteNdArray labels;
}
