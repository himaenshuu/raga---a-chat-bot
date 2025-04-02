import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
// import { OpenAI } from "openai";
import { pipeline } from "@xenova/transformers"; // Hugging Face Embedding Model

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import "dotenv/config";

type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  // OPEN_API_KEY,
} = process.env;

// const openai = new OpenAI({ apiKey: OPEN_API_KEY });

const f1Data = [
  "https://en.wikipedia.org/wiki/Formula_One",
  "https://en.wikipedia.org/wiki/Origin_of_language",
  "https://en.wikipedia.org/wiki/Forbes",
  "https://www.who.int/",
  "https://www.unicef.org/",
  "https://olympics.com/",
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const hfEmbedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
); // added

const collections = await db.listCollections();
const collectionNames = collections.map((col) => col.name); // Extract names
if (!collectionNames.includes(ASTRA_DB_COLLECTION)) {
  console.log("Creating collection...");
  await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: { dimension: 384, metric: "dot_product" },
  });
} else {
  console.log("Collection already exists.");
}

const ensureCollectionExists = async (
  similarityMetric: SimilarityMetric = "dot_product"
) => {
  await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: { dimension: 384, metric: similarityMetric }, // Reduced from 1536 to 384
  });
};

// const createCollection = async (
//   similarityMetric: SimilarityMetric = "dot_product"
// ) => {
//   const res = await db.createCollection(ASTRA_DB_COLLECTION, {
//     vector: {
//       dimension: 1536,
//       metric: similarityMetric,
//     },
//   });
//   console.log(res);
// };

// const loadSampleData = async () => {
//   const collection = await db.createCollection(ASTRA_DB_COLLECTION);

//   for await (const url of f1Data) {
//     const content = await scrapPage(url);
//     const chunks = await splitter.splitText(content);
//     for await (const chunk of chunks) {
//       const embedding = await openai.embeddings.create({
//         model: "text-embedding-3-small",
//         input: chunk,
//         encoding_format: "float",
//       });
//       const vector = embedding.data[0].embedding;

//       const res = await collection.insertOne({
//         $vector: vector,
//         text: chunk,
//       });
//       console.log(res);
//     }
//   }
// };

const getEmbedding = async (text: string) => {
  const embedding = await hfEmbedder(text, {
    pooling: "mean",
    normalize: true,
  });
  return embedding.data; // Returns the float array
};

const loadSampleData = async () => {
  try {
    const collection = db.collection(ASTRA_DB_COLLECTION);
    await ensureCollectionExists();

    await Promise.all(
      f1Data.map(async (url) => {
        try {
          console.log(`Scraping: ${url}`);
          const content = await scrapPage(url);
          const chunks = await splitter.splitText(content);

          await Promise.all(
            chunks.map(async (chunk) => {
              try {
                const vector = await getEmbedding(chunk);
                await collection.insertOne({ $vector: vector, text: chunk });
                console.log(`Inserted chunk from ${url}`);
              } catch (err) {
                console.error("Embedding/Insert Error:", err);
              }
            })
          );
        } catch (err) {
          console.error(`Scraping failed for ${url}:`, err);
        }
      })
    );

    console.log("All data loaded successfully.");
  } catch (err) {
    console.error("Error in loading data:", err);
  }
};

const scrapPage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });

  return (await loader.scrape())?.replace(/<[^>]*>?/gm, "");
};

// createCollection().then(() => loadSampleData());

ensureCollectionExists("cosine").then(loadSampleData);
