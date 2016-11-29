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
		if (args.length != 1) {
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
			while ((line = reader.readLine()) != null) {
				String[] ls = line.split("");
				int p = rand.nextInt(10000)+1;
				if(p == 10)
					System.out.println("SSSS"+line+"SSSS");				
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try { reader.close(); } catch (IOException e) {}
		}
	}

}