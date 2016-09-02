import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Array;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import com.oracle.javafx.jmx.json.JSONDocument;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.ranking.NaturalRanking;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import umontreal.ssj.randvarmulti.DirichletGen;
import umontreal.ssj.probdistmulti.DirichletDist;
import umontreal.ssj.rng.*;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.analysis.function.Sigmoid;


public class SamplerOld {

    private int nArticles;
    private int nAnnotators;
    private int nLabels;
    private int nTimes;
    private int nNewspapers;

    private static int firstYear = 1980;
    private static int nMonthsPerYear = 12;
    private static int nQuartersPerYear = 4;
    private static int nMonthsPerQuarter = 3;

    private HashMap<String, Integer> newspaperIndices;
    private HashMap<String, Integer> articleNameTime;
    private HashMap<String, Integer> articleNameNewspaper;
    private HashMap<Integer, Integer> articleTime;
    private HashMap<Integer, Integer> articleNewspaper;
    private HashMap<Integer, ArrayList<Integer>> timeArticles;
    //private HashMap<Integer, ArrayList<Integer>> newspaperArticles;
    private HashMap<String, Integer> annotatorIndices;
    private HashMap<Integer, ArrayList<Integer>> annotatorArticles;

    private ArrayList<String> articleNames;
    private ArrayList<HashMap<Integer, int[]>> annotations;

    private static double alphaTimes = 100;
    private static double lambdaTimes = 0.9;
    private static double lambdaArticles = 0.9;
    private static double betaShape = 10.0;
    private static double betaScale = 1;
    private static double zealShape = 17;
    private static double zealScale = 1;
    private static double biasShape = 3;
    private static double biasScale = 1;
    private static double zealMean = 17;
    private static double zealSigma = 1;
    private static double biasMean = 3;
    private static double biasSigma = 0.4;
    private ArrayList<Double> betas;

    private ArrayList<double[]> timeFramesSimplex;     // phi
    private ArrayList<double[]> timeFramesReals;
    private ArrayList<double[]> articleFramesSimplex;  // theta
    private ArrayList<double[]> articleFramesReals;
    private double[] zeal;
    private double[] bias;
    private double[] framesMeanSimplex;
    private double[] globalMean;

    private static double mhTimeStepSigma = 0.7 ;
    private static double mhArticleStepSigma = 3.0;
    private static double mhBetaScale = 0.1;
    private static double mhZealSigma = 0.01;
    private static double mhBiasSigma = 0.01;

    //private static double mhDirichletScale = 100.0;
    //private static double mhDirichletBias = 0.1;
    //private static double mhTimeFrameSigma = 0.05;
    //private static double mhArticleFrameSigma = 0.2;

    private static Random rand = new Random();
    private static RandomStream randomStream = new MRG32k3a();
    private static Sigmoid sigmoid = new Sigmoid();

    public SamplerOld(String inputFilename, String metadataFilename, String predictionsFilename) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        Path metadataPath = Paths.get(metadataFilename);
        JSONObject metadata = (JSONObject) parser.parse(new FileReader(metadataPath.toString()));

        nLabels = 15;

        // index newspapers
        Set<String> newspapers = gatherNewspapers(metadata);
        nNewspapers = newspapers.size();
        System.out.println(nNewspapers + " newspapers");
        newspaperIndices = new HashMap<>();
        int n = 0;
        for (String newspaper : newspapers) {
            newspaperIndices.put(newspaper, n++);
        }

        // record the time and paper of each article
        articleNameTime = new HashMap<>();
        articleNameNewspaper = new HashMap<>();
        nTimes = 0;
        for (Object articleName : metadata.keySet()) {
            JSONObject articleMetadata  = (JSONObject)metadata.get(articleName);
            String source = (String) articleMetadata.get("source");
            int year = ((Long) articleMetadata.get("year")).intValue();
            int month = ((Long) articleMetadata.get("month")).intValue() - 1;
            int quarter = (int) Math.floor((double) month / (double) nMonthsPerQuarter);
            int time = (year - firstYear) * nQuartersPerYear + quarter;
            if (time >= 0) {
                articleNameTime.put(articleName.toString(), time);
                articleNameNewspaper.put(articleName.toString(), newspaperIndices.get(source));
            }
            if (time >= nTimes) {
                nTimes = time + 1;
            }

        }

        System.out.println(nTimes + " time periods");

        timeArticles = new HashMap<>();
        for (int t = 0; t < nTimes; t++) {
            timeArticles.put(t, new ArrayList<>());
        }

        // index annotators
        Set<String> annotators = gatherAnnotators(data);
        nAnnotators = annotators.size();
        System.out.println(nAnnotators + " annotators");
        annotatorIndices = new HashMap<>();
        annotatorArticles = new HashMap<>();
        int j = 0;
        for (String annotator : annotators) {
            annotatorIndices.put(annotator, j);
            annotatorArticles.put(j, new ArrayList<>());
            j += 1;
        }
        System.out.println(annotatorIndices);

        // read in the annotations and build up all relevant information
        articleNames = new ArrayList<>();
        articleTime = new HashMap<>();
        articleNewspaper = new HashMap<>();
        annotations = new ArrayList<>();
        framesMeanSimplex = new double[nLabels];

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                JSONObject article = (JSONObject) data.get(articleName);
                JSONObject annotationsJson = (JSONObject) article.get("annotations");
                JSONObject framingAnnotations = (JSONObject) annotationsJson.get("framing");
                // make sure this article has annotations
                if (framingAnnotations.size() > 0) {
                    // get a new article id (i)
                    int i = articleNames.size();
                    // store the article name for future reference
                    articleNames.add(articleName.toString());
                    // get the timestep for this article (by name)
                    int time = articleNameTime.get(articleName.toString());
                    articleTime.put(i, time);
                    // get the newspaper for this article (by name)
                    articleNewspaper.put(i, articleNameNewspaper.get(articleName.toString()));
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                    // loop through the annotators for this article
                    for (Object annotator : framingAnnotations.keySet()) {
                        // for each one, create an array to hold the annotations, and set to zero
                        int annotationArray[] = new int[nLabels];
                        // get the anntoator name and index
                        String parts[] = annotator.toString().split("_");
                        int annotatorIndex = annotatorIndices.get(parts[0]);
                        // loop through this annotator's annotations
                        JSONArray annotatorAnnotations = (JSONArray) framingAnnotations.get(annotator);
                        for (Object annotation : annotatorAnnotations) {
                            // get the code
                            double realCode = (Double) ((JSONObject) annotation).get("code");
                            // subtract 1 for zero-based indexing
                            int code = (int) Math.round(realCode) - 1;
                            // record this code as being present
                            annotationArray[code] = 1;
                        }
                        // store the annotations for this annotator
                        articleAnnotations.put(annotatorIndex, annotationArray);
                        // store the total number of annotations for each label
                        annotatorArticles.get(annotatorIndex).add(i);
                        for (int k = 0; k < nLabels; k++) {
                            framesMeanSimplex[k] += (double) annotationArray[k];
                        }
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                    timeArticles.get(time).add(i);
                }
            }
        }

        /*

        // read in predictions and add to annotations
        Scanner scanner = new Scanner(new File(predictionsFilename));
        HashMap<String, int[]> predictions = new HashMap<>();
        scanner.useDelimiter(",");
        for (int i = 0; i < 17; i++) {
            String next = scanner.next();
        }
        String rowArticleName = "";
        int[] pArray = new int[nLabels];
        int iNext = 0;
        while (scanner.hasNext()) {
            String next = scanner.next();
            int k = iNext % 17;
            if (k == 1) {
                rowArticleName = next;
            }
            else if (k > 1) {
                pArray[k-2] = Integer.parseInt(next);
            }
            if (k == 16) {
                predictions.put(rowArticleName, pArray);
                pArray = new int[nLabels];
            }
            iNext += 1;
        }

        //Do not forget to close the scanner
        scanner.close();

        int annotatorIndex = nAnnotators;
        annotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : predictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                JSONObject article = (JSONObject) data.get(articleName);
                // only use if we have no annotations for this article
                if (!articleNames.contains(articleName)) {
                    // get a new article id (i)
                    int i = articleNames.size();
                    // store the article name for future reference
                    articleNames.add(articleName);
                    // get the timestep for this article (by name)
                    int time = articleNameTime.get(articleName);
                    articleTime.put(i, time);
                    // get the newspaper for this article (by name)
                    articleNewspaper.put(i, articleNameNewspaper.get(articleName));
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                    // treat predictions as coming from a separate anntoator
                    // loop through this annotator's annotations
                    articleAnnotations.put(annotatorIndex, predictions.get(articleName));
                    // store the total number of annotations for each label
                    annotatorArticles.get(annotatorIndex).add(i);
                    for (int k = 0; k < nLabels; k++) {
                        framesMeanSimplex[k] += (double) predictions.get(articleName)[k];
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                    timeArticles.get(time).add(i);
                }
            }
        }
        nAnnotators += 1;
        */

        // normalize the overall mean
        double frameMeanSum = 0.0;
        for (int k = 0; k < nLabels; k++) {
            frameMeanSum += framesMeanSimplex[k];
        }
        for (int k = 0; k < nLabels; k++) {
            //framesMeanSimplex[k] = framesMeanSimplex[k] / frameMeanSum;

            // a little bit of craziness here...
            framesMeanSimplex[k] = 1.0 / (double) nLabels;

        }
        System.out.println("Mean of anntoations:");
        for (int k = 0; k < nLabels; k++) {
            System.out.print(framesMeanSimplex[k] + " ");
        }

        nArticles = annotations.size();
        initialize();

    }


    private Set<String> gatherNewspapers(JSONObject metadata) {
        /*
        Read in the metadata and build up the set of newspapers for valid articles
         */
        Set <String> newspapers = new HashSet<>();
        for (Object key : metadata.keySet()) {
            JSONObject articleMetadata  = (JSONObject)metadata.get(key);
            String source = (String) articleMetadata.get("source");
            int year = ((Long) articleMetadata.get("year")).intValue();
            if (year >= firstYear) {
                newspapers.add(source);
            }
        }
        return newspapers;
    }

    private Set<String> gatherAnnotators(JSONObject data) {
        /*
        Read in the data and build up the set of annotators for valid articles
         */
        Set<String> annotators = new HashSet<>();

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                JSONObject article = (JSONObject) data.get(articleName);
                JSONObject annotations = (JSONObject) article.get("annotations");
                JSONObject framingAnnotations = (JSONObject) annotations.get("framing");
                for (Object annotator : framingAnnotations.keySet()) {
                    String parts[] = annotator.toString().split("_");
                    annotators.add(parts[0]);
                }
            }
        }
        return annotators;
    }

    /*
    Initialize random variables
     */
    private void initialize() {
        // initialize article frames based on mean of annotations
        articleFramesSimplex = new ArrayList<>();
        articleFramesReals = new ArrayList<>();
        for (int i = 0; i < nArticles; i++) {
            double frameDist[] = new double[nLabels];
            double frameDistReals[] = new double[nLabels];
            // start with the global mean
            for (int k = 0; k < nLabels; k++) {
                frameDist[k] += framesMeanSimplex[k];
            }
            // add the contributions of each annotator
            HashMap<Integer, int[]> articleAnnotations = annotations.get(i);
            double nAnnotators = (double) articleAnnotations.size();
            for (int annotator : articleAnnotations.keySet()) {
                int annotatorAnnotations[] = articleAnnotations.get(annotator);
                for (int k = 0; k < nLabels; k++) {
                    frameDist[k] += (double) annotatorAnnotations[k];
                }
            }
            // normalize
            double meanSum = 0;
            for (int k = 0; k < nLabels; k++) {
                meanSum += frameDist[k];
            }
            for (int k = 0; k < nLabels; k++) {

                // CRAZY experiment here...  but basically seems to work...
                // So, actually, this is necessary for this to work at all....
                frameDist[k] = 1.0 / (double) nLabels;

                //frameDist[k] = frameDist[k] / meanSum;
                // convert the point on the simplex to reals
                frameDistReals[k] = Math.log(frameDist[k]);
            }
            articleFramesSimplex.add(frameDist);
            articleFramesReals.add(frameDistReals);
        }

        // initialize timeFrames as the mean of the corresponding articles
        timeFramesSimplex = new ArrayList<>();
        timeFramesReals = new ArrayList<>();
        for (int t = 0; t < nTimes; t++) {
            double timeMean[] = new double[nLabels];
            // initialize everything with the global mean
            /*
            ArrayList<Integer> articles = timeArticles.get(t);
            double nTimeArticles = (double) articles.size();
            if (nTimeArticles == 0) {
                System.arraycopy(framesMeanSimplex, 0, timeMean, 0, nLabels);
            }
            else {
                for (int i : articles) {
                    double articleFrames[] = articleFramesSimplex.get(i);
                    for (int k = 0; k < nLabels; k++) {
                        timeMean[k] += articleFrames[k] / (nTimeArticles);
                    }
                }
            }
            */
            //System.arraycopy(framesMeanSimplex, 0, timeMean, 0, nLabels);
            for (int k = 0; k < nLabels; k++) {
                timeMean[k] = 1.0 / (double) nLabels;
            }
            double timeMeanReals[] = simplexToReals(timeMean, nLabels);
            timeFramesSimplex.add(timeMean);
            timeFramesReals.add(timeMeanReals);

        }

        System.out.println("");

        // intialize all betas to mean of prior
        betas = new ArrayList<>();
        for (int t = 0; t < nTimes; t++) {
            betas.add(betaShape);
        }

        // initialize annotator parameters to reasonable values
        zeal = new double[nAnnotators];
        bias = new double[nAnnotators];
        for (int j = 0; j < nAnnotators; j++) {
            zeal[j] = zealShape;
            bias[j] = biasShape;
        }
    }

    void sample(int nIter, int burnIn, int samplingPeriod, int printPeriod) throws  Exception {
        int nSamples = (int) Math.floor((nIter - burnIn) / (double) samplingPeriod);

        double timeFrameSamples [][][] = new double[nSamples][nTimes][nLabels];
        double betaSamples [][] = new double[nSamples][nTimes];
        double articleFrameSamples [][][] = new double[nSamples][nArticles][nLabels];
        double zealSamples [][] = new double[nSamples][nAnnotators];
        double biasSamples [][] = new double[nSamples][nAnnotators];

        double timeFrameRate = 0.0;
        double betasRate = 0;
        double articleFramesRate = 0;
        double zealRate = 0;
        double biasRate = 0;
        int s = 0;
        int i = 0;
        while (s < nSamples) {
            timeFrameRate += sampleTimeFrames();
            //betasRate += sampleBetas();
            articleFramesRate += sampleArticleFrames();

            // Not so bad to have one per annotator, but still seems to be better without
            // Also, high rejection rates for these... try Guassians?
            zealRate += sampleZeal();
            biasRate += sampleBias();

            globalMean = recomputeGlobalMean();


            // save samples
            if (i > burnIn && i % samplingPeriod == 0) {
                for (int t = 0; t < nTimes; t++) {
                    System.arraycopy(timeFramesSimplex.get(t), 0, timeFrameSamples[s][t], 0, nLabels);
                    betaSamples[s][t] = betas.get(t);
                }
                for (int a = 0; a < nArticles; a++) {
                    System.arraycopy(articleFramesSimplex.get(a), 0, articleFrameSamples[s][a], 0, nLabels);
                }
                for (int j = 0; j < nAnnotators; j++) {
                    for (int k = 0; k < nLabels; k++) {
                        zealSamples[s][j] = zeal[j];
                        biasSamples[s][j] = bias[j];
                    }
                }
                s += 1;
            }
            i += 1;
            if (i % printPeriod == 0) {
                System.out.print(i + ": ");
                for (int k = 0; k < nLabels; k++) {
                    System.out.print(globalMean[k] + " ");
                }
                System.out.println("");
            }
        }

        System.out.println(timeFrameRate / nIter);
        System.out.println(betasRate / nIter);
        System.out.println(articleFramesRate/ nIter);
        System.out.println(zealRate / nIter);
        System.out.println(biasRate / nIter);

        // save results

        Path output_path;
        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("timeFramesSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int t = 0; t < nTimes; t++) {
                        file.write(timeFrameSamples[s][t][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        output_path = Paths.get("betaSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (s=0; s < nSamples; s++) {
                for (int t = 0; t < nTimes; t++) {
                    file.write(betaSamples[s][t] + ",");
                }
                file.write("\n");
            }
        }

        output_path = Paths.get("zealSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (s = 0; s < nSamples; s++) {
                for (int j = 0; j < nAnnotators; j++) {
                    file.write(zealSamples[s][j] + ",");
                }
                file.write("\n");
            }
        }

        output_path = Paths.get("biasSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (s = 0; s < nSamples; s++) {
                for (int j = 0; j < nAnnotators; j++) {
                    file.write(biasSamples[s][j] + ",");
                }
                file.write("\n");
            }
        }

        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("articleSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int a = 0; a < nArticles; a++) {
                        file.write(articleFrameSamples[s][a][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        // What I actually want is maybe the annotation probabilities (?)

        output_path = Paths.get("articleMeans.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (int a = 0; a < nArticles; a++) {
                double mean [] = new double[nLabels];
                HashMap<Integer, int[]> articleAnnotations = annotations.get(a);
                for (int annotator : articleAnnotations.keySet()) {
                    for (int k = 0; k < nLabels; k++) {
                        mean[k] += (double) articleAnnotations.get(annotator)[k] / (double) articleAnnotations.size();
                    }
                }
                for (int k = 0; k < nLabels; k++) {
                    file.write(mean[k] + ",");
                }
                file.write("\n");
            }
        }

        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("articleProbs" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int a = 0; a < nArticles; a++) {
                        double pk = sigmoid.value(articleFrameSamples[s][a][k] * zealSamples[s][0] - biasSamples[s][0]);
                        file.write(pk + ",");
                    }
                    file.write("\n");
                }
            }
        }


    }

    private double sampleTimeFrames() {
        // sample the distribution over frames for the first time point

        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            // get the current distribution over frames
            double currentReals[] = timeFramesReals.get(t);
            double currentSimplex[] = timeFramesSimplex.get(t);
            // create a variable for a proposal
            double proposalReals[] = new double[nLabels];

            double[] step = dirichletStep(nLabels, mhTimeStepSigma);

            // apply the step to generate a proposal
            for (int k = 0; k < nLabels; k++) {
                proposalReals[k] = currentReals[k] + step[k];
            }

            // transform the proposal to the simplex
            double proposalSimplex[] = realsToSimplex(proposalReals, nLabels);

            // get the distribution over frames at the previous time point
            double previousSimplex[];
            if (t > 0) {
                previousSimplex = timeFramesSimplex.get(t-1);
            } else {
                // if t == 0, use the global mean
                previousSimplex = new double[nLabels];
                //for (int k = 0; k < nLabels; k++) {
                //    previousSimplex[k] = framesMeanSimplex[k];
                //}
            }

            // use this to compute a distribution over the current distribution over frames
            int nPrevious;
            if (t > 0) {
                nPrevious = timeArticles.get(t - 1).size();
            }
            else {
                nPrevious = 0;
            }
            //double lambdaPrevious = (double) nPrevious / (double) (nPrevious + 1);
            double previousDist[] = dirichletPrior(alphaTimes, lambdaTimes, previousSimplex, framesMeanSimplex, nLabels);

            // get the distribution over frames in the next time point
            double nextSimplex[] = new double[nLabels];
            if (t < nTimes-1) {
                nextSimplex = timeFramesSimplex.get(t + 1);
            }

            // compute a distribution over a new distribution over frames for the current distribution
            int nCurrent = timeArticles.get(t).size();
            //double lambdaCurrent = (double) nCurrent / (double) (nCurrent + 1);
            double currentDist[] = dirichletPrior(alphaTimes, lambdaTimes, currentSimplex, framesMeanSimplex, nLabels);

            // do the same for the proposal
            double proposalDist[] = dirichletPrior(alphaTimes, lambdaTimes, proposalSimplex, framesMeanSimplex, nLabels);

            // compute the probability of the current distribution over frames conditioned on the previous
            DirichletDist dirichletDistPrevious = new DirichletDist(previousDist);
            double pCurrentGivenPrev = dirichletDistPrevious.density(currentSimplex);
            double pProposalGivenPrev = dirichletDistPrevious.density(proposalSimplex);

            // do the same for the next time point conditioned on the current and proposal
            DirichletDist dirichletDistCurrent = new DirichletDist(currentDist);
            double pNextGivenCurrent = dirichletDistCurrent.density(nextSimplex);

            for (int k = 0; k < nLabels; k++) {
                if (proposalDist[k] <= 0) {
                   System.out.println("zero");
                }
            }

            DirichletDist dirichletDistProposal = new DirichletDist(proposalDist);
            double pNextGivenProposal = dirichletDistProposal.density(nextSimplex);

            double pLogCurrent = Math.log(pCurrentGivenPrev);
            if (t < t-1) {
                pLogCurrent += Math.log(pNextGivenCurrent);
            }
            double pLogProposal = Math.log(pProposalGivenPrev);
            if (t < t-1) {
                pLogProposal += Math.log(pNextGivenProposal);
            }

            double beta = betas.get(t);

            // compute distributions over distributions for articles
            //double lambdaArticle = (double) nCurrent / (double) (nCurrent + 1);
            double currentDistArticle[] = dirichletPrior(beta, lambdaArticles, currentSimplex, framesMeanSimplex, nLabels);
            DirichletDist dirichletDistArticleCurrent = new DirichletDist(currentDistArticle);

            double proposalDistArticle[] = dirichletPrior(beta, lambdaArticles, proposalSimplex, framesMeanSimplex, nLabels);
            DirichletDist dirichletDistArticleProposal = new DirichletDist(proposalDistArticle);

            // compute the probability of the article distributions for both current and proposal
            ArrayList<Integer> articles = timeArticles.get(t);
            for (int i : articles) {
                double articleDist[] = articleFramesSimplex.get(i);
                double pArticleCurrent = dirichletDistArticleCurrent.density(articleDist);
                pLogCurrent += Math.log(pArticleCurrent);
                double pArticleProposal= dirichletDistArticleProposal.density(articleDist);
                pLogProposal += Math.log(pArticleProposal);
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFramesSimplex.set(t, proposalSimplex);
                timeFramesReals.set(t, proposalReals);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }

    private double sampleBetas() {
        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            double current = betas.get(t);

            // generate a beta proposal (symmetric)
            GammaDistribution gamma = new GammaDistribution(current+1, mhBetaScale);
            double proposal = gamma.sample();
            double mhpProposal = gamma.density(proposal);

            if (proposal <= 0) {
                System.out.println("Error: beta proposal <= 0; current = " + current);
            }

            GammaDistribution reverseGamma = new GammaDistribution(proposal+1, mhBetaScale);
            double mhpReverse = reverseGamma.density(current);

            GammaDistribution betaPrior = new GammaDistribution(betaShape, betaScale);
            double pLogCurrent = Math.log(betaPrior.density(current));
            double pLogProposal = Math.log(betaPrior.density(proposal));

            // get the previous beta
            //double previous;
            //if (t > 0) {
            //    previous = betas.get(t-1);
            //} else {
            //    previous = betaShape0;
            //}

            // get beta in the next time point
            //double next = 1;
            //if (t < nTimes-1) {
            //    next = betas.get(t + 1);
            //}

            //GammaDistribution gammaPrevious = new GammaDistribution(previous, betaScale);
            //double pCurrentGivenPrev = gammaPrevious.density(current);
            //double pProposalGivenPrev = gammaPrevious.density(proposal);

            //GammaDistribution gammaCurrent = new GammaDistribution(current, betaScale);
            //GammaDistribution gammaProposal = new GammaDistribution(proposal, betaScale);

            //double pNextGivenCurrent = gammaCurrent.density(next);
            //double pNextGivenProposal = gammaProposal.density(next);

            //double pLogCurrent = Math.log(pCurrentGivenPrev);
            //if (t < t-1) {
            //    pLogCurrent += Math.log(pNextGivenCurrent);
            //}
            //double pLogProposal = Math.log(pProposalGivenPrev);
            //if (t < t-1) {
            //    pLogProposal += Math.log(pNextGivenProposal);
            //}

            // compute distributions over distributions for articles
            double timeFrameDist[] = timeFramesSimplex.get(t);

            double currentDistArticle[] = dirichletPrior(current, lambdaArticles, timeFrameDist, framesMeanSimplex, nLabels);
            DirichletDist dirichletDistArticleCurrent = new DirichletDist(currentDistArticle);

            double proposalDistArticle[] = dirichletPrior(proposal, lambdaArticles, timeFrameDist, framesMeanSimplex, nLabels);
            DirichletDist dirichletDistArticleProposal = new DirichletDist(proposalDistArticle);

            // compute the probability of the article disrtibutions for both current and proposal
            ArrayList<Integer> articles = timeArticles.get(t);
            for (int i : articles) {
                double articleDist[] = articleFramesSimplex.get(i);
                double pArticleCurrent = dirichletDistArticleCurrent.density(articleDist);
                pLogCurrent += Math.log(pArticleCurrent);
                double pArticleProposal= dirichletDistArticleProposal.density(articleDist);
                pLogProposal += Math.log(pArticleProposal);
            }

            double a = Math.exp(Math.log(mhpReverse) + pLogProposal - Math.log(mhpProposal) - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                betas.set(t, proposal);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }


    private double sampleArticleFrames() {
        double nAccepted = 0;

        // loop through all articles
        for (int i = 0; i < nArticles; i++) {

            // get the current article dist
            double currentSimplex[] = articleFramesSimplex.get(i);
            double currentReals[] = articleFramesReals.get(i);
            // create a variable for a proposal
            double proposalReals[] = new double[nLabels];

            double[] step = dirichletStep(nLabels, mhArticleStepSigma);

            // apply the step to generate a proposal
            for (int k = 0; k < nLabels; k++) {
                proposalReals[k] = currentReals[k] + step[k];
            }

            // transform the proposal to the simplex
            double proposalSimplex[] = realsToSimplex(proposalReals, nLabels);

            int time = articleTime.get(i);
            double beta = betas.get(time);
            double timeFrameSimplex[] = timeFramesSimplex.get(time);

            // compute distributions over distributions for articles
            int nCurrent = timeArticles.get(time).size();
            //double lambda = (double) nCurrent / (double) (nCurrent + 1);
            double articleDist[] = dirichletPrior(beta, lambdaArticles, timeFrameSimplex, framesMeanSimplex, nLabels);
            DirichletDist dirichletDistArticle = new DirichletDist(articleDist);

            // calcualte the probability of the article disrtibution conditioned on the time distribution
            double pLogCurrent = Math.log(dirichletDistArticle.density(currentSimplex));
            double pLogProposal = Math.log(dirichletDistArticle.density(proposalSimplex));

            // compute the probability of the labels for both current and proposal
            HashMap<Integer, int[]> articleAnnotations = annotations.get(i);
            for (int annotator : articleAnnotations.keySet()) {
                int labels[] = articleAnnotations.get(annotator);
                for (int k = 0; k < nLabels; k++) {
                    double pLabelCurrent =  sigmoid.value(currentSimplex[k] * zeal[annotator] - bias[annotator]);
                    double pLabelProposal =  sigmoid.value(proposalSimplex[k] * zeal[annotator] - bias[annotator]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                articleFramesSimplex.set(i, proposalSimplex);
                articleFramesReals.set(i, proposalReals);
                nAccepted += 1;
            }

        }
        return nAccepted / nArticles;
    }



    private double sampleZeal() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            ArrayList<Integer> articles = annotatorArticles.get(annotator);

            // get the current value
            double current = zeal[annotator];

            //GammaDistribution proposalDist = new GammaDistribution(current+1, 1.0);
            //double proposal = proposalDist.sample();
            //double mhpProposal = proposalDist.density(proposal);

            //GammaDistribution reverseDist = new GammaDistribution(proposal+1, 1.0);
            //double mhpReverse = reverseDist.density(current);

            double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhZealSigma);

            GammaDistribution zealPrior = new GammaDistribution(zealShape, zealScale);
            double pLogCurrent = Math.log(zealPrior.density(current));
            double pLogProposal = Math.log(zealPrior.density(proposal));

            // compute the probability of the annotations for all relevant articles
            for (int article : articles) {
                for (int k = 0; k < nLabels; k++) {
                    double articleDist[] = articleFramesSimplex.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * current - bias[annotator]);
                    double pLabelProposal = sigmoid.value(articleDist[k] * proposal - bias[annotator]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                zeal[annotator] = proposal;
                nAccepted += 1;
            }

        }
        return nAccepted / (nAnnotators);
    }

    private double sampleZealGaussian() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            ArrayList<Integer> articles = annotatorArticles.get(annotator);

            // get the current value
            double current = zeal[annotator];

            //GammaDistribution proposalDist = new GammaDistribution(current+1, 1.0);
            //double proposal = proposalDist.sample();
            //double mhpProposal = proposalDist.density(proposal);

            //GammaDistribution reverseDist = new GammaDistribution(proposal+1, 1.0);
            //double mhpReverse = reverseDist.density(current);

            double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhZealSigma);

            NormalDistribution zealPrior = new NormalDistribution(Math.log(zealMean), zealSigma);
            double pLogCurrent = Math.log(zealPrior.density(Math.log(current)));
            double pLogProposal = Math.log(zealPrior.density(Math.log(proposal)));

            // compute the probability of the annotations for all relevant articles

            for (int article : articles) {
                for (int k = 0; k < nLabels; k++) {
                    double articleDist[] = articleFramesSimplex.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * current - bias[annotator]);
                    double pLabelProposal = sigmoid.value(articleDist[k] * proposal - bias[annotator]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                zeal[annotator] = proposal;
                nAccepted += 1;
            }

        }
        return nAccepted / (nAnnotators);
    }

    private double sampleBias() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            ArrayList<Integer> articles = annotatorArticles.get(annotator);

            // get the current value
            double current = bias[annotator];
            // create a variable for a proposal
            //GammaDistribution proposalDist = new GammaDistribution(current, 1.0);
            //double proposal = proposalDist.sample();
            //double mhpProposal = proposalDist.density(proposal);

            //GammaDistribution reverseDist = new GammaDistribution(proposal, 1.0);
            //double mhpReverse = reverseDist.density(current);
            double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhBiasSigma);

            GammaDistribution biasPrior = new GammaDistribution(biasShape, biasScale);
            double pLogCurrent = Math.log(biasPrior.density(current));
            double pLogProposal = Math.log(biasPrior.density(proposal));

            // compute the probability of the annotations for all relevant articles
            for (int article : articles) {
                for (int k = 0; k < nLabels; k++) {
                    double articleDist[] = articleFramesSimplex.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * zeal[annotator] - current);
                    double pLabelProposal = sigmoid.value(articleDist[k] * zeal[annotator] - proposal);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                bias[annotator] = proposal;
                nAccepted += 1;
            }

        }
        return nAccepted / (nAnnotators);
    }

    private double[] recomputeGlobalMean() {
        double mean[] = new double[nLabels];
        // loop through all articles
        for (int i = 0; i < nArticles; i++) {
            // get the current article dist
            double distSimplex[] = articleFramesSimplex.get(i);
            for (int k = 0; k < nLabels; k++) {
                mean[k] += distSimplex[k];
            }
        }
        for (int k = 0; k < nLabels; k++) {
            mean[k] = mean[k] / (double) nArticles;
        }
        return mean;
    }

    private double[] simplexToReals(double p[], int size) {
        double reals[] = new double[size];
        for (int k = 0; k < size; k++) {
            reals[k] = Math.log(p[k]);
        }
        return reals;
    }

    private double[] realsToSimplex(double r[], int size) {
        double simplex[] = new double[size];
        double rSum = 0.0;
        for (int k = 0; k < size; k++) {
            rSum += Math.exp(r[k]);
        }
        for (int k = 0; k < size; k++) {
            simplex[k] = Math.exp(r[k]) / rSum;
        }
        return simplex;
    }

    private double[] dirichletStep(int size, double sigma) {
        double step[] = new double[size];
        // choose a direction to move for MH
        double stepDirection[] = new double[size];
        double mhProposalDist[] = new double[size];
        for (int k = 0; k < nLabels; k++) {
            mhProposalDist[k] = 1;
        }
        DirichletGen.nextPoint(randomStream, mhProposalDist, stepDirection);

        // choose a step size for MH
        double stepSize = rand.nextGaussian() * sigma;
        for (int k = 0; k < size; k++) {
            step[k] = stepDirection[k] * stepSize;
        }
        return step;
    }

    private double[] dirichletPrior(double alpha, double lambda, double p1[], double p2[], int size) {
        double prior[] = new double[size];
        for (int k = 0; k < size; k++) {
            prior[k] = (alpha + 1) * size * (lambda * p1[k] + (1-lambda) * p2[k]);
        }
        return prior;
    }


    /*
    private double sampleTimeFramesGuassian() {
        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            // get the distribution over frames at the previous time point
            double previous[];
            if (t > 0) {
                previous = timeFrames.get(t-1);
            } else {
                previous = new double[nLabels];
                for (int k = 0; k < nLabels; k++) {
                    previous[k] = 1.0 / nLabels;
                }
            }

            // use this to compute a distribution over the current distribution over frames
            double previousDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                previousDist[k] = previous[k] * nLabels * alpha + alpha0;
            }

            // get the current distribution over frames
            double current[] = timeFrames.get(t);
            // create a variable for a proposal
            double proposal[] = new double[nLabels];

            // get the distribution over frames in the next time point
            double next[] = new double[nLabels];
            if (t < nTimes-1) {
                next = timeFrames.get(t + 1);
            }


            // generate a proposal (symmetrically)
            double temp[] = new double[nLabels];
            double total = 0;
            for (int k = 0; k < nLabels; k++) {
                temp[k] = Math.exp(Math.log(current[k]) + rand.nextGaussian() * mhTimeFrameSigma);
                total += temp[k];
            }
            for (int k = 0; k < nLabels; k++) {
                proposal[k] = temp[k] / total;
            }


            / *
            // compute a distribution for generating a proposal
            double mhProposalDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                mhProposalDist[k] = mhDirichletScale * current[k] + mhDirichletBias;
            }

            // generate a point from this distribution
            DirichletGen.nextPoint(randomStream, mhProposalDist, proposal);

            // evaluate the probability of generating this new point from the proposal
            DirichletDist dirichletDist = new DirichletDist(mhProposalDist);
            double mhpProposal = dirichletDist.density(proposal);

            // compute the probability of the generating the current point from the proposal
            double reverseDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                reverseDist[k] = mhDirichletScale * proposal[k] + mhDirichletBias;
            }

            DirichletDist dirichletDistReverse = new DirichletDist(reverseDist);
            double mhpReverse = dirichletDist.density(current);
            * /

            // compute a distribution over a new distribution over frames for the current distribution
            double currentDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                currentDist[k] = current[k] * nLabels * alpha + alpha0;
            }

            // do the same for the proposal
            double proposalDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                proposalDist[k] = proposal[k] * nLabels * alpha + alpha0;
            }

            // compute the probability of the current distribution over frames conditioned on the previous
            DirichletDist dirichletDistPrevious = new DirichletDist(previousDist);
            double pCurrentGivenPrev = dirichletDistPrevious.density(current);
            double pProposalGivenPrev = dirichletDistPrevious.density(proposal);

            // do the same for the next time point conditioned on the current and proposal
            DirichletDist dirichletDistCurrent = new DirichletDist(currentDist);
            double pNextGivenCurrent = dirichletDistCurrent.density(next);

            DirichletDist dirichletDistProposal = new DirichletDist(proposalDist);
            double pNextGivenProposal = dirichletDistProposal.density(next);

            double pLogCurrent = Math.log(pCurrentGivenPrev);
            if (t < t-1) {
                pLogCurrent += Math.log(pNextGivenCurrent);
            }
            double pLogProposal = Math.log(pProposalGivenPrev);
            if (t < t-1) {
                pLogProposal += Math.log(pNextGivenProposal);
            }

            double beta = betas.get(t);
            // compute distributions over distributions for articles
            double currentDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                currentDistArticle[k] = current[k] * nLabels * beta + beta0;
            }

            DirichletDist dirichletDistArticleCurrent = new DirichletDist(currentDistArticle);

            double proposalDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                proposalDistArticle[k] = proposal[k] * nLabels * beta + beta0;
            }

            DirichletDist dirichletDistArticleProposal = new DirichletDist(proposalDistArticle);

            // compute the probability of the article disrtibutions for both current and proposal
            ArrayList<Integer> articles = timeArticles.get(t);
            for (int i : articles) {
                double articleDist[] = articleFrames.get(i);
                double pArticleCurrent = dirichletDistArticleCurrent.density(articleDist);
                pLogCurrent += Math.log(pArticleCurrent);
                double pArticleProposal= dirichletDistArticleProposal.density(articleDist);
                pLogProposal += Math.log(pArticleProposal);
            }

            double a = Math.exp(pLogProposal - pLogCurrent);

            double u = rand.nextDouble();

            if (u < a) {
                timeFrames.set(t, proposal);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }




    */
}
