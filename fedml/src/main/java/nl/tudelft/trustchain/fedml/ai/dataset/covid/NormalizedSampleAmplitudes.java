package nl.tudelft.trustchain.fedml.ai.dataset.covid;
/**
 * Handles the wave data in amplitude-time domain.
 *
 * @author Jacquet Wong
 */
public class NormalizedSampleAmplitudes {

    private final Wave wave;
    private float[] normalizedAmplitudes; // normalizedAmplitudes[sampleNumber]=normalizedAmplitudeInTheFrame

    public NormalizedSampleAmplitudes(Wave wave) {
        this.wave = wave;
    }

    /**
     *
     * Get normalized amplitude of each frame
     *
     * @return	array of normalized amplitudes(signed 16 bit): normalizedAmplitudes[frame]=amplitude
     */
    public float[] getNormalizedAmplitudes() {

        if (normalizedAmplitudes == null) {

            boolean signed = true;

            // usually 8bit is unsigned
            if (wave.getWaveHeader().getBitsPerSample() == 8) {
                signed = false;
            }

            short[] amplitudes = wave.getSampleAmplitudes();
            int numSamples = amplitudes.length;
            int maxAmplitude = 1 << (wave.getWaveHeader().getBitsPerSample() - 1);

            if (!signed) { // one more bit for unsigned value
                maxAmplitude <<= 1;
            }

            normalizedAmplitudes = new float[numSamples];
            for (int i = 0; i < numSamples; i++) {
                normalizedAmplitudes[i] = (float) amplitudes[i] / maxAmplitude;
            }
        }
        return normalizedAmplitudes;
    }
}
