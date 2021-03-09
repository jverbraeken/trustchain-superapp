package nl.tudelft.trustchain.fedml.ai.dataset.cifar;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;

public class CustomFlipImageTransform extends BaseImageTransform<Mat> {

    /**
     * the deterministic flip mode
     *                 {@code  0} Flips around x-axis.
     *                 {@code >0} Flips around y-axis.
     *                 {@code <0} Flips around both axes.
     */
    private final int flipMode;

    private int h;
    private int w;
    private int mode;

    /**
     * Calls {@code this(null)} and sets the flip mode.
     *
     * @param flipMode the deterministic flip mode
     *                 {@code  0} Flips around x-axis.
     *                 {@code >0} Flips around y-axis.
     *                 {@code <0} Flips around both axes.
     */
    public CustomFlipImageTransform(int flipMode, Random random) {
        super(random);
        converter = new OpenCVFrameConverter.ToMat();
        this.flipMode = flipMode;
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }

        Mat mat = converter.convert(image.getFrame());

        if(mat == null) {
            return null;
        }
        h = mat.rows();
        w = mat.cols();

        mode = random != null ? random.nextInt(4) - 2 : flipMode;

        Mat result = new Mat();
        if (mode < -1) {
            // no flip
            mat.copyTo(result);
        } else {
            flip(mat, result, mode);
        }

        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            float x = coordinates[i    ];
            float y = coordinates[i + 1];
            float x2 = w - x - 1;
            float y2 = h - y - 1;

            if (mode < -1) {
                transformed[i    ] = x;
                transformed[i + 1] = y;
            } else if (mode == 0) {
                transformed[i    ] = x;
                transformed[i + 1] = y2;
            } else if (mode > 0) {
                transformed[i    ] = x2;
                transformed[i + 1] = y;
            } else {
                transformed[i    ] = x2;
                transformed[i + 1] = y2;
            }
        }
        return transformed;
    }
}
