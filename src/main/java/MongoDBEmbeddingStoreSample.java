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

import java.util.ArrayList;
import java.util.List;

public class MongoDBEmbeddingStoreSample {
    public static void main(String[] args) {
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

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ArrayList<String> texts = new ArrayList<String>();
        texts.add("The weather is pretty bad today.");
        texts.add("Emperor penguins are the tallest and heaviest of all penguin species, standing up to 4 feet.");
        texts.add("The word \"pomme\" (apple) was historically used to describe many round fruits, and when potatoes were introduced to Europe, they were named pommes de terre because they grow underground, distinguishing them from regular apples that grow on trees.");

        for (String text : texts) {
            TextSegment segment = TextSegment.from(text);
            Embedding embedding = embeddingModel.embed(segment).content();
            embeddingStore.add(embedding, segment);
        }

        Embedding queryEmbedding = embeddingModel.embed("What is a penguin?").content();
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
        EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);

        System.out.println(embeddingMatch.score());
        System.out.println(embeddingMatch.embedded().text());
    }
}