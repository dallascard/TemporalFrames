public class Main {

    public static void main(String[] args) throws Exception {

        String documents = "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json";
        String metadata = "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json";
        Sampler sampler = new Sampler(documents, metadata);

        sampler.sample();

    }
}
