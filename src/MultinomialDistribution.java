import cern.jet.math.Mult;

import java.util.Random;

class MultinomialDistribution {
    private int size;
    private double[] probs;
    private double[] cumulative;

    MultinomialDistribution(double [] probs, int size) {
        this.size = size;
        this.probs = new double[size];
        cumulative = new double[size];
        System.arraycopy(probs, 0, this.probs, 0, size);
        for (int j = 0; j < size; j++) {
            cumulative[j] = probs[j];
            if (j > 0) {
                cumulative[j] += cumulative[j-1];
            }
        }
        assert cumulative[size-1] >= 1.0;
    }

    int sample(Random rand) {
        double u = rand.nextDouble();
        for (int j = 0; j < size; j++) {
            if (u <= cumulative[j]) {
                return j;
            }
        }
        return size-1;
    }

    double pdf(int k) {
        return probs[k];
    }
}
