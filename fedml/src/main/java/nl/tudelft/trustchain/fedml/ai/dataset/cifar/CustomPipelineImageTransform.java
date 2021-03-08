package nl.tudelft.trustchain.fedml.ai.dataset.cifar;


import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.*;

import org.bytedeco.opencv.opencv_core.*;

/**
 * Allows creation of image transform pipelines, either sequentially or randomly.
 *
 * You have the option of passing in multiple transforms as parameters. If you
 * want to create a more complex pipeline, you can pass in a pipeline that looks
 * like {@literal List<Pair<ImageTransform, Double>>}. The Double value is the probability that
 * particular element in the pipeline will be executed. This is helpful for creating
 * complex pipelines.
 *
 * The pipeline can also be randomly shuffled with each transform, further increasing
 * the available dataset.
 *
 * @author saudet
 * @author crockpotveggies
 */
public class CustomPipelineImageTransform extends BaseImageTransform<Mat> {

    protected List<Pair<ImageTransform, Double>> imageTransforms;
    protected boolean shuffle;
    protected org.nd4j.linalg.api.rng.Random rng;

    protected List<ImageTransform> currentTransforms = new ArrayList<>();

    public CustomPipelineImageTransform(ImageTransform... transforms) {
        super(null); // for perf reasons we ignore java Random, create our own

        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        for (ImageTransform transform : transforms) {
            pipeline.add(new Pair<>(transform, 0.5));
        }

        this.imageTransforms = pipeline;
        this.shuffle = true;
        this.rng = Nd4j.getRandom();
        rng.setSeed(1234);
    }

    /**
     * Takes an image and executes a pipeline of combined transforms.
     *
     * @param image to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */
    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (shuffle) {
            Collections.shuffle(imageTransforms);
        }

        currentTransforms.clear();

        // execute each item in the pipeline
        for (Pair<ImageTransform, Double> tuple : imageTransforms) {
            if (tuple.getSecond() == 1.0 || rng.nextDouble() < tuple.getSecond()) { // probability of execution
                currentTransforms.add(tuple.getFirst());
                image = random != null ? tuple.getFirst().transform(image, random)
                    : tuple.getFirst().transform(image);
            }
        }

        return image;
    }

    @Override
    public float[] query(float... coordinates) {
        for (ImageTransform transform : currentTransforms) {
            coordinates = transform.query(coordinates);
        }
        return coordinates;
    }
}
