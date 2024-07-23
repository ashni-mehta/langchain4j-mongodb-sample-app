import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.model.CreateCollectionOptions;
//import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.mongodb.IndexMapping;
import dev.langchain4j.store.embedding.mongodb.MongoDbEmbeddingStore;
import org.bson.conversions.Bson;
//import org.testcontainers.containers.MongoDBContainer;

import java.util.List;

public class MongoDBEmbeddingStoreSample {
    public static void main(String[] args) {
        // MongoDBContainer mongodb = new MongoDBContainer("mongo:7.0.0");
        // mongodb.start();

        MongoClient mongoClient = MongoClients.create("URI");
        String databaseName = "sample-data";
        String collectionName = "semantic-searching";
        String indexName = "embedding";
        Long maxResultRatio = 10L;
        CreateCollectionOptions createCollectionOptions = new CreateCollectionOptions();
        Bson filter = null;
        IndexMapping indexMapping = new IndexMapping();
        Boolean createIndex = false;

        EmbeddingStore<TextSegment> embeddingStore = new MongoDbEmbeddingStore(
                mongoClient,
                databaseName,
                collectionName,
                indexName,
                maxResultRatio,
                createCollectionOptions,
                filter,
                indexMapping,
                createIndex
        );

        //  String apiKey = "demo";
        //  OpenAiChatModel model = OpenAiChatModel.withApiKey(apiKey);
        //  String answer = model.generate("What is MongoDB?");
        //  System.out.println(answer);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        TextSegment segment1 = TextSegment.from("I like football.");
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);

        TextSegment segment2 = TextSegment.from("The weather is good today.");
        Embedding embedding2 = embeddingModel.embed(segment2).content();
        embeddingStore.add(embedding2, segment2);

        Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);

        System.out.println(embeddingMatch.score()); // 0.8144289255142212
        System.out.println(embeddingMatch.embedded().text()); // I like football.
    }
}