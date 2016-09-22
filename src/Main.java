public class Main {

    public static void main(String[] args) throws Exception {

        String documents = "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json";
        String metadata = "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json";
        String predictions = "/Users/dcard/Desktop/uncorrected.csv";
        String predictionsWithTone = "/Users/dcard/Desktop/uncorrected_with_tone.csv";
        String mood = "/Users/dcard/Dropbox/CMU/ARK/compuframes/Analysis/Amber/data/quarter_smoothed_dummy.csv";
        //Sampler sampler = new Sampler(documents, metadata, predictions);
        CombinedModel sampler = new CombinedModel(documents, metadata, predictionsWithTone, mood);

        sampler.run(1000, 500, 10, 100);

    }
}
