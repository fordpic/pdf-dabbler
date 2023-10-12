import { NextRequest } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { VectorDBQAChain } from 'langchain/chains';
import { StreamingTextResponse, LangChainStream } from 'ai';
import { CallbackManager } from 'langchain/callbacks';

export async function POST(request: NextRequest) {
	// parse request
	const body = await request.json();

	// set up stream w vercel
	const { stream, handlers } = LangChainStream();

	// init pinecone
	const pineconeClient = new Pinecone({
		apiKey: process.env.PINECONE_API_KEY || '',
		environment: 'us-central1',
	});

	const pineconeIndex = pineconeClient.Index(
		process.env.PINECONE_INDEX_NAME as string
	);

	// init the vector store
	const vectorStore = await PineconeStore.fromExistingIndex(
		new OpenAIEmbeddings(),
		{ pineconeIndex }
	);

	// init model + turn on streaming
	const model = new OpenAI({
		modelName: 'gpt-3.5-turbo',
		streaming: true,
		callbacks: CallbackManager.fromHandlers(handlers),
	});
}
