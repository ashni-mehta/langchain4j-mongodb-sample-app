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
        String databaseName = "sample_mflix";
        String collectionName = "embedded_movies";
        String indexName = "embedding_index2";
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
        Embedding queryEmbedding = embeddingModel.embed("Find me a movie that takes place in Ancient Rome").content();
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);
        System.out.println(embeddingMatch.score());
        System.out.println(embeddingMatch.embedded().text());
    }
}