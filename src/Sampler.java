import java.io.FileReader;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import com.oracle.javafx.jmx.json.JSONDocument;
import org.apache.commons.math3.linear.ArrayRealVector;
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


public class Sampler {

    private int nArticles;
    private int nAnnotators;
    private int nLabels;
    private int nTimes;
    private int nNewspapers;

    private static int firstYear = 1980;
    private static int nMonthsPerYear = 12;

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

    private static double alpha = 10.0;
    private static double alpha0 = 0.1;
    private static double beta0 = 0.1;
    private static double betaSigma = 0.1;
    private static double zealMean = 3;
    private static double zealSigma = 2;
    private static double biasMean = 0;
    private static double biasSigma = 2;
    private ArrayList<Double> betas;
    private ArrayList<double[]> timeFrames;     // phi
    private ArrayList<double[]> articleFrames;  // theta
    private double[][] zeal;
    private double[][] bias;

    private static double mhDirichletScale = 100.0;
    private static double mhDirichletBias = 0.1;
    private static double mhTimeFrameSigma = 0.09;
    private static double mhBetaSigma = 0.18;
    private static double mhArticleFrameSigma = 0.4;
    private static double mhZealSigma = 0.95;
    private static double mhBiasSigma = 0.22;

    private static Random rand = new Random();
    private static RandomStream randomStream = new MRG32k3a();

    private static Sigmoid sigmoid = new Sigmoid();


    public Sampler(String inputFilename, String metadataFilename) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        Path metadataPath = Paths.get(metadataFilename);
        JSONObject metadata = (JSONObject) parser.parse(new FileReader(metadataPath.toString()));

        nLabels = 15;

        // index newspapers
        Set<String> newspapers = gatherNewspapers(metadata);
        nNewspapers = newspapers.size();
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
            int time = (year - firstYear) * nMonthsPerYear + month;
            if (time >= 0) {
                articleNameTime.put(articleName.toString(), time);
                articleNameNewspaper.put(articleName.toString(), newspaperIndices.get(source));
            }
            if (time >= nTimes) {
                nTimes = time + 1;
            }

        }

        timeArticles = new HashMap<>();
        for (int t = 0; t < nTimes; t++) {
            timeArticles.put(t, new ArrayList<>());
        }

        // index annotators
        Set<String> annotators = gatherAnnotators(data);
        nAnnotators = annotators.size();
        annotatorIndices = new HashMap<>();
        annotatorArticles = new HashMap<>();
        int j = 0;
        for (String annotator : annotators) {
            annotatorIndices.put(annotator, j);
            annotatorArticles.put(j, new ArrayList<>());
            j += 1;
        }


        // read in the annotations and build up all relevant information
        articleNames = new ArrayList<>();
        articleTime = new HashMap<>();
        articleNewspaper = new HashMap<>();
        annotations = new ArrayList<>();

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
                        annotatorArticles.get(annotatorIndex).add(i);
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                    timeArticles.get(time).add(i);
                }
            }
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
        // generate timeFrames, each conditioned on the previous
        timeFrames = new ArrayList<>();
        // start with a uniform dirichlet
        double alphas[] = new double[nLabels];
        for (int k = 0; k < nLabels; k++) {
            alphas[k] = alpha + alpha0;
        }
        double p[] = new double[nLabels];
        for (int t = 0; t < nTimes; t++) {
            // generate a point from a dirichlet distribution parameterized by alpha
            DirichletGen.nextPoint(randomStream, alphas, p);
            // store the new point
            double frameDist[] = new double[nLabels];
            System.arraycopy(p, 0, frameDist, 0, nLabels);
            timeFrames.add(frameDist);
            // create a new (softened) prior based on the previous draw
            for (int k = 0; k < nLabels; k++) {
                alphas[k] = p[k] * nLabels * alpha + alpha0;
            }
            //DirichletDist dirichletDist = new DirichletDist(alphas);
            //double density = dirichletDist.density(p[0]);
            //System.out.println(density);
        }

        // generate betas, each conditioned on the previous
        betas = new ArrayList<>();
        betas.add(Math.exp(rand.nextGaussian() * betaSigma));
        for (int t = 1; t < nTimes; t++) {
            betas.add(Math.exp(Math.log(betas.get(t-1)) + rand.nextGaussian() * betaSigma));
        }

        // generate articleFrames, conditioned on timeFrames
        articleFrames = new ArrayList<>();
        for (int i = 0; i < nArticles; i++) {
            int time = articleTime.get(i);
            double timeFrameDist[] = timeFrames.get(time);
            double beta = betas.get(time);
            for (int k = 0; k < nLabels; k++) {
                alphas[k] = timeFrameDist[k] * beta * nLabels + beta0;
            }
            DirichletGen.nextPoint(randomStream, alphas, p);
            double frameDist[] = new double[nLabels];
            System.arraycopy(p, 0, frameDist, 0, nLabels);
            articleFrames.add(frameDist);
        }

        // initialize annotator parameters
        zeal = new double[nAnnotators][nLabels];
        bias = new double[nAnnotators][nLabels];
        for (int j = 0; j < nAnnotators; j++) {
            for (int k = 0; k < nLabels; k++) {
                zeal[j][k] = zealMean + rand.nextGaussian() * zealSigma;
                bias[j][k] = biasMean + rand.nextGaussian() * biasSigma;
            }
        }
    }

    void sample(int nIter, int burnIn, int samplingPeriod) {
        int nSamples = (int) Math.floor((nIter - burnIn) / (double) samplingPeriod);

        double timeFrameSamples [][][] = new double[nSamples][nTimes][nLabels];
        double betaSamples [][] = new double[nSamples][nTimes];
        double articleFrameSamples [][][] = new double[nSamples][nArticles][nLabels];
        double zealSamples [][][] = new double[nSamples][nAnnotators][nLabels];
        double biasSamples [][][] = new double[nSamples][nAnnotators][nLabels];

        double timeFrameRate = 0.0;
        double betasRate = 0;
        double articleFramesRate = 0;
        double zealRate = 0;
        double biasRate = 0;
        int s = 0;
        int i = 0;
        while (s < nSamples) {
            timeFrameRate += sampleTimeFramesGuassian();
            betasRate += sampleBetas();
            articleFramesRate += sampleArticleFrames();
            zealRate += sampleZeal();
            biasRate += sampleBias();

            // save samples
            if (i > burnIn && i % samplingPeriod == 0) {
                for (int t = 0; t < nTimes; t++) {
                    System.arraycopy(timeFrames.get(t), 0, timeFrameSamples[s][t], 0, nLabels);
                    betaSamples[s][t] = betas.get(t);
                }
                for (int a = 0; a < nArticles; a++) {
                    System.arraycopy(articleFrames.get(a), 0, articleFrameSamples[s][a], 0, nLabels);
                }
                for (int j = 0; j < nAnnotators; j++) {
                    for (int k = 0; k < nLabels; k++) {
                        zealSamples[s][j][k] = zeal[j][k];
                        biasSamples[s][j][k] = bias[j][k];
                    }
                }

            }
            i += 1;
        }

        System.out.println(timeFrameRate / nIter);
        System.out.println(betasRate / nIter);
        System.out.println(articleFramesRate/ nIter);
        System.out.println(zealRate / nIter);
        System.out.println(biasRate / nIter);

        

    }

    private double sampleTimeFrames() {
        // sample the distribution over frames for the first time point

        double nAccepted = 0;

        // loop through all time points
        for (int t = 1; t < nTimes; t++) {

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

            double a = Math.exp(Math.log(mhpReverse) + pLogProposal - Math.log(mhpProposal) - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFrames.set(t, proposal);
                nAccepted += 1;
            }

        }
        double acceptanceRate = nAccepted / nTimes;
        return acceptanceRate;
    }

    private double sampleTimeFramesGuassian() {
        double nAccepted = 0;

        // loop through all time points
        for (int t = 1; t < nTimes; t++) {

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


            /*
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
            */

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



    private double sampleBetas() {
        // sample the distribution over frames for the first time point

        double nAccepted = 0;

        // loop through all time points
        for (int t = 1; t < nTimes; t++) {

            // get the previous beta
            double previous;
            if (t > 0) {
                previous = betas.get(t-1);
            } else {
                previous = 1.0;
            }

            double current = betas.get(t);

            // get beta in the next time point
            double next = 1;
            if (t < nTimes-1) {
                next = betas.get(t + 1);
            }

            // generate a beta proposal (symmetric)
            double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhBetaSigma);

            NormalDistribution prevDist = new NormalDistribution(Math.log(previous), betaSigma);

            double pCurrentGivenPrev = prevDist.density(Math.log(current));
            double pProposalGivenPrev = prevDist.density(Math.log(proposal));

            NormalDistribution currentDist = new NormalDistribution(Math.log(current), betaSigma);
            NormalDistribution proposalDist = new NormalDistribution(Math.log(proposal), betaSigma);

            double pNextGivenCurrent = currentDist.density(next);
            double pNextGivenProposal = proposalDist.density(next);

            double pLogCurrent = Math.log(pCurrentGivenPrev);
            if (t < t-1) {
                pLogCurrent += Math.log(pNextGivenCurrent);
            }
            double pLogProposal = Math.log(pProposalGivenPrev);
            if (t < t-1) {
                pLogProposal += Math.log(pNextGivenProposal);
            }

            // compute distributions over distributions for articles

            double currentFrameDist[] = timeFrames.get(t);

            double currentDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                currentDistArticle[k] = currentFrameDist[k] * nLabels * current + beta0;
            }

            DirichletDist dirichletDistArticleCurrent = new DirichletDist(currentDistArticle);

            double proposalDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                proposalDistArticle[k] = currentFrameDist[k] * nLabels * proposal + beta0;
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
                betas.set(t, proposal);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }


    private double sampleArticleFrames() {
        double nAccepted = 0;

        // loop through all articles
        for (int i = 1; i < nArticles; i++) {

            // get the current article dist
            double current[] = articleFrames.get(i);
            // create a variable for a proposal
            double proposal[] = new double[nLabels];

            // generate a proposal (symmetrically)
            double temp[] = new double[nLabels];
            double total = 0;
            for (int k = 0; k < nLabels; k++) {
                temp[k] = Math.exp(Math.log(current[k]) + rand.nextGaussian() * mhArticleFrameSigma);
                total += temp[k];
            }
            for (int k = 0; k < nLabels; k++) {
                proposal[k] = temp[k] / total;
            }

            int time = articleTime.get(i);
            double beta = betas.get(time);
            double timeFrame[] = timeFrames.get(time);

            // compute distributions over distributions for articles
            double articleDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                articleDist[k] = timeFrame[k] * nLabels * beta + beta0;
            }

            DirichletDist dirichletDistArticle = new DirichletDist(articleDist);
            double pLogCurrent = Math.log(dirichletDistArticle.density(current));
            double pLogProposal = Math.log(dirichletDistArticle.density(proposal));

            // compute the probability of the labels for both current and proposal
            HashMap<Integer, int[]> articleAnnotations = annotations.get(i);

            for (int annotator : articleAnnotations.keySet()) {
                int labels[] = articleAnnotations.get(annotator);
                for (int k = 0; k < nLabels; k++) {
                    double pLabelCurrent =  sigmoid.value(current[k] * zeal[annotator][k] + bias[annotator][k]);
                    double pLabelProposal =  sigmoid.value(proposal[k] * zeal[annotator][k] + bias[annotator][k]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);

            double u = rand.nextDouble();

            if (u < a) {
                articleFrames.set(i, proposal);
                nAccepted += 1;
            }

        }
        return nAccepted / nArticles;
    }

    private double sampleZeal() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 1; annotator < nAnnotators; annotator++) {

            ArrayList<Integer> articles = annotatorArticles.get(annotator);

            for (int k = 0; k < nLabels; k++) {

                // get the current value
                double current = zeal[annotator][k];
                // create a variable for a proposal
                double proposal = current + rand.nextGaussian() * mhZealSigma;

                NormalDistribution zealPrior = new NormalDistribution(zealMean, zealSigma);

                double pLogCurrent = Math.log(zealPrior.density(current));
                double pLogProposal = Math.log(zealPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles

                for (int article : articles) {
                    double articleDist[] = articleFrames.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * current + bias[annotator][k]);
                    double pLabelProposal = sigmoid.value(articleDist[k] * proposal + bias[annotator][k]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);

                double u = rand.nextDouble();

                if (u < a) {
                    zeal[annotator][k] = proposal;
                    nAccepted += 1;
                }

            }
        }
        return nAccepted / (nAnnotators * nLabels);
    }

    private double sampleBias() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 1; annotator < nAnnotators; annotator++) {

            ArrayList<Integer> articles = annotatorArticles.get(annotator);

            for (int k = 0; k < nLabels; k++) {

                // get the current value
                double current = bias[annotator][k];
                // create a variable for a proposal
                double proposal = current + rand.nextGaussian() * mhBiasSigma;

                NormalDistribution biasPrior = new NormalDistribution(biasMean, biasSigma);

                double pLogCurrent = Math.log(biasPrior.density(current));
                double pLogProposal = Math.log(biasPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles
                for (int article : articles) {
                    double articleDist[] = articleFrames.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * zeal[annotator][k]+ current);
                    double pLabelProposal = sigmoid.value(articleDist[k] * zeal[annotator][k]+ proposal);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);

                double u = rand.nextDouble();

                if (u < a) {
                    bias[annotator][k] = proposal;
                    nAccepted += 1;
                }

            }
        }
        return nAccepted / (nAnnotators * nLabels);
    }

}
