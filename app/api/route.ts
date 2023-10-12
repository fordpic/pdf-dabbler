import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { NextRequest, NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';

export async function POST(request: NextRequest) {
	// extract formData from request
	const data = await request.formData();

	// extract uploaded file from formData
	const file: File | null = data.get('file') as unknown as File;

	if (!file) {
		return NextResponse.json({ success: false, error: 'No file found' });
	}

	if (file.type !== 'application/pdf') {
		return NextResponse.json({ success: false, error: 'Invalid file type' });
	}

	// load the pdf, split into smaller docs
	const pdfLoader = new PDFLoader(file);
	const splitDocs = await pdfLoader.loadAndSplit();

	// init Pinecode
	const pineconeClient = new Pinecone({
		apiKey: process.env.PINECONE_API_KEY ?? '',
		environment: 'us-central1',
	});

	const pineconeIndex = pineconeClient.Index(
		process.env.PINECONE_INDEX_NAME as string
	);

	// use langchain integration w pinecone to store the docs
	await PineconeStore.fromDocuments(splitDocs, new OpenAIEmbeddings(), {
		pineconeIndex,
	});

	return NextResponse.json({ success: true });
}
