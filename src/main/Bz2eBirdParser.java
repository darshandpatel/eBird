package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

public class Bz2eBirdParser {

	public static void main(String[] args) {
		
		System.out.println(args.length);
		if (args.length < 2) {
			System.out.println("Input bz2 file required on command line.");
			System.exit(1);
		}

		BufferedReader reader = null;
		try {
			File inputFile = new File(args[0]);
			if (!inputFile.exists() || inputFile.isDirectory() || !inputFile.getName().endsWith(".bz2")) {
				System.out.println("Input File does not exist or not bz2 file: " + args[0]);
				System.exit(1);
			}

			BZip2CompressorInputStream inputStream = new BZip2CompressorInputStream(new FileInputStream(inputFile));
			reader = new BufferedReader(new InputStreamReader(inputStream));
			String line;
			Random rand = new Random();
			int lineCounter = 1;
			long totalNumberOfRecords = 0;
			while ((line = reader.readLine()) != null) {
				//String[] ls = line.split("");
				totalNumberOfRecords++;
				/*
				int p = rand.nextInt(10000);
				if(lineCounter == 1)
					System.out.println("First Line : "+line);
				else if(p == 0){
					System.out.println("Second Line :" + line);
					System.out.println("------------------------------");
					System.out.println("------------------------------");
					System.out.println("------------------------------");
				}
				*/
				lineCounter++;
			}
			System.out.println("Total number of records :" + totalNumberOfRecords);

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try { reader.close(); } catch (IOException e) {}
		}
	}

}