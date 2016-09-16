
class Transformations {

    // Note: reals to simplex is not uniquely reversible; prob should just use this once...
    static double[] simplexToReals(double[] simplex, int size) {
        double reals[] = new double[size];
        for (int k = 0; k < size; k++) {
            reals[k] = Math.log(simplex[k]);
        }
        return reals;
    }

    static double[] realsToSimplex(double [] reals, int size) {
        double [] simplex = new double[size];
        double sum = 0;
        for (int k = 0; k < size; k++) {
            simplex[k] = Math.exp(reals[k]);
            sum += simplex[k];
        }
        for (int k = 0; k < size; k++) {
            simplex[k] = simplex[k] / sum;
        }
        return simplex;
    }

    static double[] cubeToReals(double p[], int size) {
        double reals[] = new double[size];
        for (int k = 0; k < size; k++) {
            reals[k] = Math.log(-Math.log(p[k]));
        }
        return reals;
    }

    static double[] realsToCube(double r[], int size) {
        double cube[] = new double[size];
        for (int k = 0; k < size; k++) {
            cube[k] = Math.exp(-Math.exp(r[k]));
        }
        return cube;
    }


}
