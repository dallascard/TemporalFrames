public class Main {

    public static void main(String[] args) throws Exception {

        String documents = "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json";
        String metadata = "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json";
        String predictions = "/Users/dcard/Desktop/uncorrected.csv";
        Sampler sampler = new Sampler(documents, metadata, predictions);

        sampler.sample(5000, 4000, 10, 100);

    }
}
