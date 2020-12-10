package nl.tudelft.trustchain.fedml.ai;


import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

import static org.tensorflow.ndarray.index.Indices.from;
import static org.tensorflow.ndarray.index.Indices.to;

/**
 * Common loader and data preprocessor for MNIST and FashionMNIST datasets.
 */
public class MnistDataset {
    public static final int NUM_CLASSES = 10;

    public static MnistDataset create(int validationSize, String trainingImagesArchive, String trainingLabelsArchive,
                                      String testImagesArchive, String testLabelsArchive, File baseDirectory) {
        try {
            ByteNdArray trainingImages = readArchive(trainingImagesArchive, baseDirectory);
            ByteNdArray trainingLabels = readArchive(trainingLabelsArchive, baseDirectory);
            ByteNdArray testImages = readArchive(testImagesArchive, baseDirectory);
            ByteNdArray testLabels = readArchive(testLabelsArchive, baseDirectory);

            if (validationSize > 0) {
                return new MnistDataset(
                    trainingImages.slice(from(validationSize)),
                    trainingLabels.slice(from(validationSize)),
                    trainingImages.slice(to(validationSize)),
                    trainingLabels.slice(to(validationSize)),
                    testImages,
                    testLabels,
                    baseDirectory
                );
            }
            return new MnistDataset(trainingImages, trainingLabels, null, null, testImages, testLabels, baseDirectory);

        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    public Iterable<ImageBatch> trainingBatches(int batchSize) {
        return () -> new ImageBatchIterator(batchSize, trainingImages, trainingLabels);
    }

    public Iterable<ImageBatch> validationBatches(int batchSize) {
        return () -> new ImageBatchIterator(batchSize, validationImages, validationLabels);
    }

    public Iterable<ImageBatch> testBatches(int batchSize) {
        return () -> new ImageBatchIterator(batchSize, testImages, testLabels);
    }

    public ImageBatch testBatch() {
        return new ImageBatch(testImages, testLabels);
    }

    public long imageSize() {
        return imageSize;
    }

    public long numTrainingExamples() {
        return trainingLabels.shape().size(0);
    }

    public long numTestingExamples() {
        return testLabels.shape().size(0);
    }

    public long numValidationExamples() {
        return validationLabels.shape().size(0);
    }

    private static final int TYPE_UBYTE = 0x08;

    private final ByteNdArray trainingImages;
    private final ByteNdArray trainingLabels;
    private final ByteNdArray validationImages;
    private final ByteNdArray validationLabels;
    private final ByteNdArray testImages;
    private final ByteNdArray testLabels;
    private final long imageSize;
    private final File baseDirectory;

    private MnistDataset(
        ByteNdArray trainingImages,
        ByteNdArray trainingLabels,
        ByteNdArray validationImages,
        ByteNdArray validationLabels,
        ByteNdArray testImages,
        ByteNdArray testLabels,
        File baseDirectory) {
        this.trainingImages = trainingImages;
        this.trainingLabels = trainingLabels;
        this.validationImages = validationImages;
        this.validationLabels = validationLabels;
        this.testImages = testImages;
        this.testLabels = testLabels;
        this.imageSize = trainingImages.get(0).shape().size();
        this.baseDirectory = baseDirectory;
    }

    private static ByteNdArray readArchive(String archiveName, File baseDirectory) throws IOException {
        DataInputStream archiveStream = new DataInputStream(
            new GZIPInputStream(new FileInputStream(Paths.get(baseDirectory.getPath(), "mnist", archiveName).toFile()))
        );
        archiveStream.readShort(); // first two bytes are always 0
        byte magic = archiveStream.readByte();
        if (magic != TYPE_UBYTE) {
            throw new IllegalArgumentException("\"" + archiveName + "\" is not a valid archive");
        }
        int numDims = archiveStream.readByte();
        long[] dimSizes = new long[numDims];
        int size = 1;  // for simplicity, we assume that total size does not exceeds Integer.MAX_VALUE
        for (int i = 0; i < dimSizes.length; ++i) {
            dimSizes[i] = archiveStream.readInt();
            size *= dimSizes[i];
        }
        byte[] bytes = new byte[size];
        archiveStream.readFully(bytes);
        return NdArrays.wrap(DataBuffers.of(bytes, true, false), Shape.of(dimSizes));
    }
}
